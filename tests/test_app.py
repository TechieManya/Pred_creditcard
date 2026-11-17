"""
test_app.py – Pytest suite for fraud detection API (minimum 4 tests)
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app, raise_server_exceptions=True)

# ── Sample valid input ─────────────────────────────────────────────────────────
VALID_INPUT = {
    "amount_usd": 150.0,
    "hours_since_last_txn": 12.5,
    "txn_count_last_24h": 3,
    "distance_from_home_km": 15.0,
    "card_age_months": 36,
    "customer_age": 35,
    "account_balance_usd": 5000.0,
    "cvv_retry_count": 0,
    "velocity_score": 2.5,
    "time_of_day_hour": 14,
    "day_of_week": 2,
    "merchant_risk_score": 0.15,
    "prior_disputes": 0,
    "merchant_category": "Groceries",
    "card_type": "Visa",
    "auth_method": "OTP",
    "channel": "POS",
    "device_type": "iPhone",
    "is_foreign_transaction": False,
    "is_new_merchant": False,
    "used_vpn": False,
    "ip_country_mismatch": False,
    "billing_shipping_mismatch": False,
    "is_ai_generated_scam_attempt": False,
}

HIGH_RISK_INPUT = {
    "amount_usd": 9999.0,
    "hours_since_last_txn": 0.1,
    "txn_count_last_24h": 50,
    "distance_from_home_km": 15000.0,
    "card_age_months": 1,
    "customer_age": 22,
    "account_balance_usd": 100.0,
    "cvv_retry_count": 5,
    "velocity_score": 90.0,
    "time_of_day_hour": 3,
    "day_of_week": 6,
    "merchant_risk_score": 0.95,
    "prior_disputes": 5,
    "merchant_category": "Crypto Exchange",
    "card_type": "Amex",
    "auth_method": "No Authentication",
    "channel": "Online",
    "device_type": "Windows PC",
    "is_foreign_transaction": True,
    "is_new_merchant": True,
    "used_vpn": True,
    "ip_country_mismatch": True,
    "billing_shipping_mismatch": True,
    "is_ai_generated_scam_attempt": True,
}


# ── Test 1: Health check ───────────────────────────────────────────────────────
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


# ── Test 2: UI loads ───────────────────────────────────────────────────────────
def test_ui_loads():
    response = client.get("/")
    assert response.status_code == 200
    assert "Fraud Detection" in response.text


# ── Test 3: Domain-logic test (high-risk > low-risk) ──────────────────────────
def test_domain_logic_high_risk_beats_low_risk():
    """A clearly suspicious transaction should have higher fraud probability
    than a normal one."""
    resp_low = client.post("/predict", json=VALID_INPUT)
    resp_high = client.post("/predict", json=HIGH_RISK_INPUT)

    assert resp_low.status_code == 200
    assert resp_high.status_code == 200

    prob_low = resp_low.json()["fraud_probability"]
    prob_high = resp_high.json()["fraud_probability"]

    assert prob_high > prob_low, (
        f"High-risk transaction ({prob_high:.4f}) should have higher "
        f"fraud probability than low-risk ({prob_low:.4f})"
    )


# ── Test 4: Boundary test – invalid input returns 422 ─────────────────────────
def test_invalid_input_returns_422():
    """Invalid merchant_category should return HTTP 422."""
    bad_input = VALID_INPUT.copy()
    bad_input["merchant_category"] = "INVALID_CATEGORY"
    response = client.post("/predict", json=bad_input)
    assert response.status_code == 422


# ── Test 5: Out-of-range numeric returns 422 ──────────────────────────────────
def test_out_of_range_numeric_returns_422():
    """Negative amount should return HTTP 422."""
    bad_input = VALID_INPUT.copy()
    bad_input["amount_usd"] = -100.0
    response = client.post("/predict", json=bad_input)
    assert response.status_code == 422


# ── Test 6: Valid prediction has correct structure ─────────────────────────────
def test_prediction_structure():
    response = client.post("/predict", json=VALID_INPUT)
    assert response.status_code == 200
    data = response.json()
    assert "fraud_probability" in data
    assert "is_fraud" in data
    assert "risk_level" in data
    assert "tips" in data
    assert isinstance(data["tips"], list)
    assert 0 <= data["fraud_probability"] <= 1
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
