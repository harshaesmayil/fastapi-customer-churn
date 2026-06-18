import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "models" / "churn_classifier.pkl"

pipeline = joblib.load(MODEL_PATH)

FEATURE_ORDER = [
    "tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen",
    "gender", "Partner", "Dependents",
    "InternetService", "Contract", "PaymentMethod", "OnlineSecurity"
]


def _build_row(customer: dict) -> dict:
    """Map API field names → training column names."""
    return {
        "tenure":          customer["tenure"],
        "MonthlyCharges":  customer["monthly_charges"],
        "TotalCharges":    customer["total_charges"],
        "SeniorCitizen":   customer["senior_citizen"],
        "gender":          customer["gender"],
        "Partner":         customer["partner"],
        "Dependents":      customer["dependents"],
        "InternetService": customer["internet_service"],
        "Contract":        customer["contract"],
        "PaymentMethod":   customer["payment_method"],
        "OnlineSecurity":  customer["online_security"],
    }


def _make_prediction(prob: float) -> dict:
    will_churn = prob >= 0.35

    if prob >= 0.65:
        risk = "High"
        action = "Offer retention discount immediately"
    elif prob >= 0.35:
        risk = "Medium"
        action = "Schedule a follow-up call"
    else:
        risk = "Low"
        action = "No action needed"

    return {
        "prediction":         "Will Churn" if will_churn else "Will Not Churn",
        "probability":        round(float(prob), 4),
        "risk_level":         risk,
        "recommended_action": action,
    }


def predict_single(customer: dict) -> dict:
    """Predict churn for one customer. Called by POST /predict."""
    row = _build_row(customer)
    df = pd.DataFrame([row])[FEATURE_ORDER]
    prob = pipeline.predict_proba(df)[0][1]
    return _make_prediction(prob)


def predict_batch(customers: list[dict]) -> list[dict]:
    """Predict churn for multiple customers. Called by POST /predict/batch."""
    rows = [_build_row(c) for c in customers]
    df = pd.DataFrame(rows)[FEATURE_ORDER]
    probs = pipeline.predict_proba(df)[:, 1]
    return [_make_prediction(p) for p in probs]