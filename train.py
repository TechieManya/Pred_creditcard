"""
train.py – Training script with MLflow experiment tracking & Kaggle submission generation
Fraud Detection – Kaggle Playground Series S6E11
"""

import argparse
import os
import sys
import warnings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import joblib
import mlflow
import mlflow.sklearn
try:
    import mlflow.lightgbm
except Exception:
    pass
try:
    import mlflow.xgboost
except Exception:
    pass
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join("data", "train.csv")
TEST_PATH = os.path.join("data", "test.csv")
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
ENCODERS_PATH = os.path.join(MODEL_DIR, "encoders.joblib")
FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "feature_cols.joblib")

# ── Feature lists ──────────────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "amount_usd", "hours_since_last_txn", "txn_count_last_24h",
    "distance_from_home_km", "card_age_months", "customer_age",
    "account_balance_usd", "cvv_retry_count", "velocity_score",
    "time_of_day_hour", "day_of_week", "merchant_risk_score", "prior_disputes",
]

CATEGORICAL_FEATURES = [
    "merchant_category", "card_type", "auth_method", "channel", "device_type",
]

BOOLEAN_FEATURES = [
    "is_foreign_transaction", "is_new_merchant", "used_vpn",
    "ip_country_mismatch", "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
]

TARGET = "is_fraud"


def generate_synthetic_data(n_samples: int = 100) -> pd.DataFrame:
    """Generate minimal synthetic dataset when data/train.csv is absent (e.g. on CI)."""
    np.random.seed(42)
    data = {
        "amount_usd": np.random.uniform(5, 5000, n_samples),
        "hours_since_last_txn": np.random.uniform(0.1, 72, n_samples),
        "txn_count_last_24h": np.random.randint(1, 20, n_samples),
        "distance_from_home_km": np.random.uniform(1, 500, n_samples),
        "card_age_months": np.random.randint(1, 120, n_samples),
        "customer_age": np.random.randint(18, 80, n_samples),
        "account_balance_usd": np.random.uniform(100, 50000, n_samples),
        "cvv_retry_count": np.random.randint(0, 4, n_samples),
        "velocity_score": np.random.uniform(0, 10, n_samples),
        "time_of_day_hour": np.random.randint(0, 24, n_samples),
        "day_of_week": np.random.randint(0, 7, n_samples),
        "merchant_risk_score": np.random.uniform(0, 1, n_samples),
        "prior_disputes": np.random.randint(0, 3, n_samples),
        "merchant_category": np.random.choice(["Groceries", "Online Retail", "Electronics"], n_samples),
        "card_type": np.random.choice(["Visa", "Mastercard", "Amex"], n_samples),
        "auth_method": np.random.choice(["OTP", "PIN", "3D Secure"], n_samples),
        "channel": np.random.choice(["POS", "Online", "In-App"], n_samples),
        "device_type": np.random.choice(["iPhone", "Android Phone", "Windows PC"], n_samples),
        "is_foreign_transaction": np.random.choice([False, True], n_samples),
        "is_new_merchant": np.random.choice([False, True], n_samples),
        "used_vpn": np.random.choice([False, True], n_samples),
        "ip_country_mismatch": np.random.choice([False, True], n_samples),
        "billing_shipping_mismatch": np.random.choice([False, True], n_samples),
        "is_ai_generated_scam_attempt": np.random.choice([False, True], n_samples),
        "is_fraud": np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
    }
    return pd.DataFrame(data)


def load_and_preprocess(path: str, sample_size: int = None):
    """Load data, engineer features, encode categoricals, scale numerics."""
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        print(f"Notice: '{path}' not found. Generating synthetic dataset for CI/pipeline...")
        df = generate_synthetic_data(n_samples=sample_size or 100)
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    # ── Feature engineering ────────────────────────────────────────────────
    df["total_spending"] = df["amount_usd"] * df["txn_count_last_24h"]
    df["spending_to_balance_ratio"] = df["amount_usd"] / (df["account_balance_usd"] + 1)
    df["is_high_risk_hour"] = df["time_of_day_hour"].apply(
        lambda h: 1 if h < 6 or h >= 22 else 0
    )
    df["risk_flags_sum"] = (
        df["is_foreign_transaction"].astype(int)
        + df["is_new_merchant"].astype(int)
        + df["used_vpn"].astype(int)
        + df["ip_country_mismatch"].astype(int)
        + df["billing_shipping_mismatch"].astype(int)
        + df["is_ai_generated_scam_attempt"].astype(int)
    )
    df["cvv_velocity_interaction"] = df["cvv_retry_count"] * df["velocity_score"]

    engineered = [
        "total_spending", "spending_to_balance_ratio",
        "is_high_risk_hour", "risk_flags_sum", "cvv_velocity_interaction",
    ]

    # ── Encode categoricals ────────────────────────────────────────────────
    encoders = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # ── Booleans to int ────────────────────────────────────────────────────
    for col in BOOLEAN_FEATURES:
        df[col] = df[col].astype(int)

    # ── Feature matrix ─────────────────────────────────────────────────────
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES + engineered
    X = df[feature_cols]
    y = df[TARGET] if TARGET in df.columns else None

    # ── Scale ──────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols, index=X.index)

    return X_scaled, y, scaler, encoders, feature_cols


def load_and_preprocess_test(path: str, scaler, encoders, feature_cols):
    """Preprocess test set using fitted scaler and encoders."""
    df = pd.read_csv(path)

    # ── Feature engineering ────────────────────────────────────────────────
    df["total_spending"] = df["amount_usd"] * df["txn_count_last_24h"]
    df["spending_to_balance_ratio"] = df["amount_usd"] / (df["account_balance_usd"] + 1)
    df["is_high_risk_hour"] = df["time_of_day_hour"].apply(
        lambda h: 1 if h < 6 or h >= 22 else 0
    )
    df["risk_flags_sum"] = (
        df["is_foreign_transaction"].astype(int)
        + df["is_new_merchant"].astype(int)
        + df["used_vpn"].astype(int)
        + df["ip_country_mismatch"].astype(int)
        + df["billing_shipping_mismatch"].astype(int)
        + df["is_ai_generated_scam_attempt"].astype(int)
    )
    df["cvv_velocity_interaction"] = df["cvv_retry_count"] * df["velocity_score"]

    for col in CATEGORICAL_FEATURES:
        le = encoders[col]
        class_map = {cls: idx for idx, cls in enumerate(le.classes_)}
        df[col] = df[col].astype(str).map(class_map).fillna(-1).astype(int)

    for col in BOOLEAN_FEATURES:
        df[col] = df[col].astype(int)

    X = df[feature_cols]
    X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_cols, index=X.index)
    return df["id"], X_scaled


def get_model(algorithm: str, **kwargs):
    """Return a model instance based on string name."""
    if algorithm == "logistic_regression":
        return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42, **kwargs)
    elif algorithm == "random_forest":
        rf_kwargs = {"class_weight": "balanced", "random_state": 42, "n_jobs": -1}
        rf_kwargs.update(kwargs)
        return RandomForestClassifier(**rf_kwargs)
    elif algorithm == "gradient_boosting":
        gb_kwargs = {"random_state": 42}
        gb_kwargs.update(kwargs)
        return GradientBoostingClassifier(**gb_kwargs)
    elif algorithm == "lightgbm":
        if not HAS_LIGHTGBM:
            raise ImportError("lightgbm is not installed.")
        lgb_kwargs = {
            "n_estimators": 100,
            "learning_rate": 0.08,
            "num_leaves": 31,
            "random_state": 42,
            "verbose": -1,
            "n_jobs": -1,
        }
        lgb_kwargs.update(kwargs)
        return LGBMClassifier(**lgb_kwargs)
    elif algorithm == "xgboost":
        if not HAS_XGBOOST:
            raise ImportError("xgboost is not installed.")
        xgb_kwargs = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42,
            "eval_metric": "auc",
            "n_jobs": -1,
        }
        xgb_kwargs.update(kwargs)
        return XGBClassifier(**xgb_kwargs)
    else:
        valid = ["logistic_regression", "random_forest", "gradient_boosting", "lightgbm", "xgboost"]
        raise ValueError(f"Unknown algorithm: {algorithm}. Choose from {valid}")


def train_and_log(algorithm: str, X, y, **kwargs):
    """Train a model with cross-validation and log to MLflow."""
    model = get_model(algorithm, **kwargs)

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1 if algorithm in ["lightgbm", "xgboost", "random_forest"] else 1)

    # Final fit on full train set
    model.fit(X, y)
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    # Metrics
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, y_proba),
        "cv_roc_auc_mean": cv_scores.mean(),
        "cv_roc_auc_std": cv_scores.std(),
    }

    # MLflow logging
    try:
        with mlflow.start_run(run_name=algorithm):
            mlflow.log_param("algorithm", algorithm)
            for k, v in kwargs.items():
                mlflow.log_param(k, v)
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
            if algorithm == "lightgbm" and hasattr(mlflow, "lightgbm"):
                mlflow.lightgbm.log_model(model, "model")
            elif algorithm == "xgboost" and hasattr(mlflow, "xgboost"):
                mlflow.xgboost.log_model(model, "model")
            else:
                mlflow.sklearn.log_model(model, "model")
    except Exception as e:
        print(f"Warning: MLflow logging failed ({e}), continuing...")

    print(f"\n{'='*60}")
    print(f"Algorithm: {algorithm}")
    for k, v in metrics.items():
        print(f"  {k:25s}: {v:.4f}")
    print(f"{'='*60}")

    return model, metrics


def generate_submission(model, test_ids, X_test, output_path="submission.csv"):
    """Generate Kaggle submission file with validation checks."""
    test_proba = model.predict_proba(X_test)[:, 1]
    submission = pd.DataFrame({
        "id": test_ids,
        "is_fraud": test_proba,
    })
    submission.to_csv(output_path, index=False)
    print(f"\n[SUCCESS] Kaggle submission successfully generated!")
    print(f"   Saved to: {output_path}")
    print(f"   Shape: {submission.shape}")
    print(f"   Null count: {submission.isnull().sum().sum()}")
    print("   Prediction Statistics:")
    print(submission["is_fraud"].describe().to_string())
    return submission


def main():
    parser = argparse.ArgumentParser(description="Train fraud detection model")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="lightgbm",
        choices=["logistic_regression", "random_forest", "gradient_boosting", "lightgbm", "xgboost"],
    )
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=0.08)
    parser.add_argument("--run_all", action="store_true", help="Train baseline, tree and boosting models to pick the best")
    parser.add_argument("--sample_size", type=int, default=None, help="Sample size for fast training / tests")
    parser.add_argument("--generate_submission", action="store_true", default=True, help="Generate submission.csv")
    parser.add_argument("--test_path", type=str, default=TEST_PATH)
    parser.add_argument("--submission_output", type=str, default="submission.csv")
    args = parser.parse_args()

    # ── MLflow setup ───────────────────────────────────────────────────────
    try:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("fraud_detection")
    except Exception as e:
        print(f"Notice: MLflow tracking initialization: {e}")

    # ── Load data ──────────────────────────────────────────────────────────
    print("Loading and preprocessing data...")
    X, y, scaler, encoders, feature_cols = load_and_preprocess(DATA_PATH, sample_size=args.sample_size)
    print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
    print(f"Fraud rate: {y.mean()*100:.2f}%")

    # ── Save preprocessing artifacts ───────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(encoders, ENCODERS_PATH)
    joblib.dump(feature_cols, FEATURE_COLS_PATH)

    best_model = None
    best_score = -1
    best_algo = ""

    if args.run_all:
        configs = [
            ("logistic_regression", {}),
            ("random_forest", {"n_estimators": 100, "max_depth": 10}),
        ]
        if HAS_LIGHTGBM:
            configs.append(("lightgbm", {"n_estimators": args.n_estimators, "learning_rate": args.learning_rate}))
        else:
            configs.append(("gradient_boosting", {"n_estimators": args.n_estimators, "max_depth": args.max_depth, "learning_rate": args.learning_rate}))
    else:
        extra = {}
        if args.algorithm in ("random_forest", "gradient_boosting", "lightgbm", "xgboost"):
            extra["n_estimators"] = args.n_estimators
        if args.algorithm in ("random_forest", "gradient_boosting", "xgboost"):
            extra["max_depth"] = args.max_depth
        if args.algorithm in ("gradient_boosting", "lightgbm", "xgboost"):
            extra["learning_rate"] = args.learning_rate
        configs = [(args.algorithm, extra)]

    for algo, kwargs in configs:
        model, metrics = train_and_log(algo, X, y, **kwargs)
        score = metrics["cv_roc_auc_mean"] if not np.isnan(metrics["cv_roc_auc_mean"]) else metrics.get("roc_auc", 0)
        if best_model is None or score > best_score:
            best_score = score
            best_model = model
            best_algo = algo

    # ── Save best model ────────────────────────────────────────────────────
    joblib.dump(best_model, MODEL_PATH)
    print(f"\n[SUCCESS] Best model saved: {best_algo} (CV ROC-AUC: {best_score:.4f})")
    print(f"   -> {MODEL_PATH}")

    # ── Generate submission if test data exists ────────────────────────────
    if args.generate_submission and os.path.exists(args.test_path):
        print(f"\nPreprocessing test set ({args.test_path})...")
        test_ids, X_test = load_and_preprocess_test(args.test_path, scaler, encoders, feature_cols)
        generate_submission(best_model, test_ids, X_test, output_path=args.submission_output)


if __name__ == "__main__":
    main()
