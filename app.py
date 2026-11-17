"""
app.py – FastAPI service for Fraud Detection
Serves predictions via POST /predict with Pydantic validation.
"""

import os
import logging
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

# ── Logging setup ──────────────────────────────────────────────────────────────
logger = logging.getLogger("fraud_predictor")
logger.setLevel(logging.INFO)

# ── Load model and artifacts at startup (not per request) ──────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
model = joblib.load(os.path.join(MODEL_DIR, "model.joblib"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.joblib"))
feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.joblib"))

# ── Pydantic Schema ───────────────────────────────────────────────────────────

MERCHANT_CATEGORIES = [
    "Groceries", "Online Retail", "Utilities", "Electronics", "Healthcare",
    "Restaurants", "Fuel", "Streaming", "Crypto Exchange", "Gaming",
    "Travel", "Gift Cards",
]
CARD_TYPES = ["Mastercard", "Visa", "Amex", "RuPay", "Discover"]
AUTH_METHODS = ["OTP", "PIN", "3D Secure", "Biometric", "No Authentication"]
CHANNELS = ["POS", "In-App", "Online", "ATM", "Contactless"]
DEVICE_TYPES = [
    "iPhone", "Android Phone", "Tablet", "Windows PC",
    "POS Terminal", "Mac", "ATM Machine", "Smart Watch",
]


class TransactionInput(BaseModel):
    """Pydantic model enforcing the data contract."""
    amount_usd: float = Field(..., ge=0, le=100000, description="Transaction amount in USD")
    hours_since_last_txn: float = Field(..., ge=0, le=10000, description="Hours since last transaction")
    txn_count_last_24h: int = Field(..., ge=0, le=500, description="Transactions in last 24 hours")
    distance_from_home_km: float = Field(..., ge=0, le=50000, description="Distance from home in km")
    card_age_months: int = Field(..., ge=0, le=1200, description="Card age in months")
    customer_age: int = Field(..., ge=0, le=120, description="Customer age")
    account_balance_usd: float = Field(..., ge=0, le=10000000, description="Account balance in USD")
    cvv_retry_count: int = Field(..., ge=0, le=20, description="CVV retry count")
    velocity_score: float = Field(..., ge=0, le=100, description="Transaction velocity score")
    time_of_day_hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Mon, 6=Sun)")
    merchant_risk_score: float = Field(..., ge=0, le=100, description="Merchant risk score (0-100)")
    prior_disputes: int = Field(..., ge=0, le=100, description="Number of prior disputes")
    merchant_category: str = Field(..., description="Merchant category")
    card_type: str = Field(..., description="Card type")
    auth_method: str = Field(..., description="Authentication method")
    channel: str = Field(..., description="Transaction channel")
    device_type: str = Field(..., description="Device type")
    is_foreign_transaction: bool = Field(..., description="Is foreign transaction")
    is_new_merchant: bool = Field(..., description="Is new merchant")
    used_vpn: bool = Field(..., description="Was VPN used")
    ip_country_mismatch: bool = Field(..., description="IP country mismatch")
    billing_shipping_mismatch: bool = Field(..., description="Billing/shipping address mismatch")
    is_ai_generated_scam_attempt: bool = Field(..., description="AI-generated scam attempt flag")

    @field_validator("merchant_category")
    @classmethod
    def validate_merchant_category(cls, v):
        if v not in MERCHANT_CATEGORIES:
            raise ValueError(f"Must be one of: {MERCHANT_CATEGORIES}")
        return v

    @field_validator("card_type")
    @classmethod
    def validate_card_type(cls, v):
        if v not in CARD_TYPES:
            raise ValueError(f"Must be one of: {CARD_TYPES}")
        return v

    @field_validator("auth_method")
    @classmethod
    def validate_auth_method(cls, v):
        if v not in AUTH_METHODS:
            raise ValueError(f"Must be one of: {AUTH_METHODS}")
        return v

    @field_validator("channel")
    @classmethod
    def validate_channel(cls, v):
        if v not in CHANNELS:
            raise ValueError(f"Must be one of: {CHANNELS}")
        return v

    @field_validator("device_type")
    @classmethod
    def validate_device_type(cls, v):
        if v not in DEVICE_TYPES:
            raise ValueError(f"Must be one of: {DEVICE_TYPES}")
        return v


class PredictionOutput(BaseModel):
    fraud_probability: float
    is_fraud: bool
    risk_level: str
    tips: list[str]


# ── FastAPI App ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Fraud Detection API",
    description="Predict transaction fraud probability – CSL236 End-Term Project",
    version="1.0.0",
)
STATIC_DIR = os.path.join(BASE_DIR, "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def preprocess_input(data: TransactionInput) -> pd.DataFrame:
    """Transform a single input into the feature vector the model expects."""
    d = data.model_dump()

    # Feature engineering (same as train.py)
    d["total_spending"] = d["amount_usd"] * d["txn_count_last_24h"]
    d["spending_to_balance_ratio"] = d["amount_usd"] / (d["account_balance_usd"] + 1)
    d["is_high_risk_hour"] = 1 if d["time_of_day_hour"] < 6 or d["time_of_day_hour"] >= 22 else 0
    d["risk_flags_sum"] = sum([
        int(d["is_foreign_transaction"]),
        int(d["is_new_merchant"]),
        int(d["used_vpn"]),
        int(d["ip_country_mismatch"]),
        int(d["billing_shipping_mismatch"]),
        int(d["is_ai_generated_scam_attempt"]),
    ])
    d["cvv_velocity_interaction"] = d["cvv_retry_count"] * d["velocity_score"]

    # Encode categoricals
    for col in ["merchant_category", "card_type", "auth_method", "channel", "device_type"]:
        le = encoders[col]
        d[col] = le.transform([d[col]])[0]

    # Booleans to int
    for col in ["is_foreign_transaction", "is_new_merchant", "used_vpn",
                "ip_country_mismatch", "billing_shipping_mismatch",
                "is_ai_generated_scam_attempt"]:
        d[col] = int(d[col])

    df = pd.DataFrame([d])[feature_cols]
    df_scaled = pd.DataFrame(scaler.transform(df), columns=feature_cols)
    return df_scaled


def generate_tips(data: TransactionInput, prob: float) -> list[str]:
    """Generate 2-3 rule-based tips for the user."""
    tips = []
    if prob > 0.5:
        tips.append("🚨 HIGH RISK: This transaction has a high fraud probability. Consider blocking it.")
    if data.cvv_retry_count >= 3:
        tips.append("⚠️ Multiple CVV retries detected — this is a common fraud pattern.")
    if data.used_vpn:
        tips.append("🔒 VPN usage detected. Verify the customer's identity before approving.")
    if data.ip_country_mismatch:
        tips.append("🌍 IP country does not match billing country — elevated risk.")
    if data.billing_shipping_mismatch:
        tips.append("📦 Billing and shipping addresses differ — watch for reshipping fraud.")
    if data.is_ai_generated_scam_attempt:
        tips.append("🤖 AI-generated scam attempt flagged — treat with extreme caution.")
    if data.amount_usd > 5000:
        tips.append("💰 Large transaction amount — ensure the customer profile supports this.")
    if data.time_of_day_hour < 6 or data.time_of_day_hour >= 22:
        tips.append("🌙 Transaction at unusual hour — nighttime fraud rates are elevated.")
    if prob <= 0.1:
        tips.append("✅ Low risk transaction. Standard processing recommended.")

    return tips[:3]  # Return at most 3 tips


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionOutput)
def predict(data: TransactionInput):
    X = preprocess_input(data)
    prob = float(model.predict_proba(X)[0, 1])
    is_fraud = prob >= 0.5
    risk_level = "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.3 else "LOW"
    tips = generate_tips(data, prob)

    # Log prediction
    import json
    logger.info(json.dumps({
        "timestamp": datetime.now().isoformat(),
        "input": data.model_dump(),
        "fraud_probability": prob,
        "is_fraud": is_fraud,
    }))

    return PredictionOutput(
        fraud_probability=round(prob, 4),
        is_fraud=is_fraud,
        risk_level=risk_level,
        tips=tips,
    )


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    html_path = os.path.join(BASE_DIR, "static", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()
