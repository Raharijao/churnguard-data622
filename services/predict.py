import numpy as np
import pandas as pd
import joblib
import os

# Load ONLY sklearn model (no xgboost dependency)
MODEL_PATH = os.path.join("models", "logistic_model.pkl")

model = joblib.load(MODEL_PATH)

def predict_churn(input_df: pd.DataFrame):
    """
    Safe HF prediction (no xgboost required)
    """
    probs = model.predict_proba(input_df)[:, 1]
    preds = (probs > 0.5).astype(int)

    return preds, probs


def explain_customer(row):
    """
    Simple deterministic explanation (HF-safe fallback)
    """
    reasons = []

    if row.get("tenure", 0) < 12:
        reasons.append("Short customer tenure increases churn risk")

    if row.get("monthly_charges", 0) > 80:
        reasons.append("High monthly charges may lead to churn")

    if row.get("contract", 0) == "Month-to-month":
        reasons.append("Month-to-month contracts have higher churn rates")

    if len(reasons) == 0:
        reasons.append("Customer profile appears stable")

    return reasons