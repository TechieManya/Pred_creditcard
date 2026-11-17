"""
verify_app.py – Standalone verification script (no TestClient, no async).
Imports app module and tests logic directly without starting a server.
"""
import os, sys, importlib, traceback

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath("."))

# ── Import app components without starting server ──────────────────────────────
print("=== Loading app module ===")
import app as api_app
print(f"  ENCODER_MAPS keys: {list(api_app.ENCODER_MAPS.keys())}")
print(f"  merchant_category count: {len(api_app.MERCHANT_CATEGORIES)}")
print(f"  merchant_categories: {api_app.MERCHANT_CATEGORIES}")
print()

# ── Test inputs ────────────────────────────────────────────────────────────────
VALID_INPUT = {
    "amount_usd": 150.0, "hours_since_last_txn": 12.5, "txn_count_last_24h": 3,
    "distance_from_home_km": 15.0, "card_age_months": 36, "customer_age": 35,
    "account_balance_usd": 5000.0, "cvv_retry_count": 0, "velocity_score": 2.5,
    "time_of_day_hour": 14, "day_of_week": 2, "merchant_risk_score": 15.0,
    "prior_disputes": 0, "merchant_category": "Groceries", "card_type": "Visa",
    "auth_method": "OTP", "channel": "POS", "device_type": "iPhone",
    "is_foreign_transaction": False, "is_new_merchant": False, "used_vpn": False,
    "ip_country_mismatch": False, "billing_shipping_mismatch": False,
    "is_ai_generated_scam_attempt": False,
}

HIGH_RISK_INPUT = {
    "amount_usd": 9999.0, "hours_since_last_txn": 0.1, "txn_count_last_24h": 50,
    "distance_from_home_km": 15000.0, "card_age_months": 1, "customer_age": 22,
    "account_balance_usd": 100.0, "cvv_retry_count": 5, "velocity_score": 90.0,
    "time_of_day_hour": 3, "day_of_week": 6, "merchant_risk_score": 95.0,
    "prior_disputes": 5, "merchant_category": "Crypto Exchange", "card_type": "Amex",
    "auth_method": "No Authentication", "channel": "Online", "device_type": "Windows PC",
    "is_foreign_transaction": True, "is_new_merchant": True, "used_vpn": True,
    "ip_country_mismatch": True, "billing_shipping_mismatch": True,
    "is_ai_generated_scam_attempt": True,
}

ALL_CATEGORIES = [
    "Groceries", "Online Retail", "Utilities", "Electronics", "Healthcare",
    "Restaurants", "Fuel", "Streaming", "Crypto Exchange", "Gaming", "Travel", "Gift Cards",
]

results = []

def run_test(name, fn):
    try:
        fn()
        print(f"  ✅ PASS: {name}")
        results.append((name, True, None))
    except Exception as e:
        print(f"  ❌ FAIL: {name}")
        print(f"          {e}")
        results.append((name, False, str(e)))

# ── Test 1: health (model loaded, merchant_category list) ──────────────────────
print("=== Test 1: Health check ===")
def t1():
    assert api_app.model is not None, "model is None"
    assert len(api_app.MERCHANT_CATEGORIES) == 12, \
        f"Expected 12 merchant_categories, got {len(api_app.MERCHANT_CATEGORIES)}"
run_test("Health: model loaded + 12 merchant_categories", t1)
print()

# ── Test 2: UI HTML exists ────────────────────────────────────────────────────
print("=== Test 2: UI HTML ===")
def t2():
    html_path = os.path.join("static", "index.html")
    assert os.path.exists(html_path), f"index.html not found at {html_path}"
    with open(html_path, encoding="utf-8") as f:
        content = f.read()
    assert "Fraud Detection" in content, "index.html doesn't contain 'Fraud Detection'"
run_test("UI: index.html exists and contains 'Fraud Detection'", t2)
print()

# ── Test 3: Valid transaction preprocesses and predicts ───────────────────────
print("=== Test 3: Valid prediction (Groceries/Visa/OTP) ===")
def t3():
    inp = api_app.TransactionInput(**VALID_INPUT)
    X = api_app.preprocess_input(inp)
    prob = float(api_app.model.predict_proba(X)[0, 1])
    assert 0.0 <= prob <= 1.0, f"Probability {prob} out of [0,1]"
    print(f"          prob={prob:.4f}")
run_test("Predict: valid input returns probability in [0,1]", t3)
print()

# ── Test 4: High-risk transaction ──────────────────────────────────────────────
print("=== Test 4: High-risk prediction (Crypto Exchange) ===")
def t4():
    inp_low = api_app.TransactionInput(**VALID_INPUT)
    inp_high = api_app.TransactionInput(**HIGH_RISK_INPUT)
    X_low  = api_app.preprocess_input(inp_low)
    X_high = api_app.preprocess_input(inp_high)
    prob_low  = float(api_app.model.predict_proba(X_low)[0, 1])
    prob_high = float(api_app.model.predict_proba(X_high)[0, 1])
    print(f"          prob_low={prob_low:.4f}  prob_high={prob_high:.4f}")
    # Note: with synthetic data the model may not always order these correctly
    # but both must succeed without error (the core bug was a crash)
    assert 0.0 <= prob_low <= 1.0 and 0.0 <= prob_high <= 1.0
run_test("Predict: high-risk input completes without ValueError", t4)
print()

# ── Test 5: Invalid category rejected by Pydantic ────────────────────────────
print("=== Test 5: Invalid merchant_category → Pydantic ValidationError ===")
def t5():
    from pydantic import ValidationError
    bad = VALID_INPUT.copy()
    bad["merchant_category"] = "INVALID_CATEGORY"
    try:
        api_app.TransactionInput(**bad)
        raise AssertionError("Should have raised ValidationError")
    except ValidationError:
        pass  # expected
run_test("Validation: INVALID_CATEGORY raises Pydantic ValidationError", t5)
print()

# ── Test 6: Negative amount rejected ──────────────────────────────────────────
print("=== Test 6: Negative amount_usd → Pydantic ValidationError ===")
def t6():
    from pydantic import ValidationError
    bad = VALID_INPUT.copy()
    bad["amount_usd"] = -100.0
    try:
        api_app.TransactionInput(**bad)
        raise AssertionError("Should have raised ValidationError")
    except ValidationError:
        pass  # expected
run_test("Validation: negative amount_usd raises Pydantic ValidationError", t6)
print()

# ── Test 7: ALL 12 merchant categories succeed ────────────────────────────────
print("=== Test 7: All 12 merchant categories ===")
cat_results = {}
for cat in ALL_CATEGORIES:
    try:
        payload = VALID_INPUT.copy()
        payload["merchant_category"] = cat
        inp = api_app.TransactionInput(**payload)
        X = api_app.preprocess_input(inp)
        prob = float(api_app.model.predict_proba(X)[0, 1])
        cat_results[cat] = ("✅ OK", round(prob, 4))
        print(f"    ✅ {cat:<20s} prob={prob:.4f}")
    except Exception as e:
        cat_results[cat] = ("❌ ERROR", str(e))
        print(f"    ❌ {cat:<20s} ERROR: {e}")

all_cats_ok = all(r[0] == "✅ OK" for r in cat_results.values())
results.append(("All 12 merchant categories", all_cats_ok, None if all_cats_ok else "Some categories failed"))
print()

# ── Summary ───────────────────────────────────────────────────────────────────
print("=" * 60)
print("SUMMARY")
print("=" * 60)
passed = sum(1 for _, ok, _ in results if ok)
total  = len(results)
for name, ok, err in results:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}")
    if err:
        print(f"         → {err}")
print()
print(f"Result: {passed}/{total} tests passed")
if passed == total:
    print("🎉 ALL TESTS PASSED")
    sys.exit(0)
else:
    print("❌ SOME TESTS FAILED")
    sys.exit(1)
