import os
import joblib
import pandas as pd
import numpy as np

# ----------------------------
# PATHS
# ----------------------------
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(BASE_DIR, "models", "logistic_model.pkl")
XGB_MODEL_PATH = os.path.join(BASE_DIR, "models", "xgboost_model.pkl")

model = joblib.load(MODEL_PATH)
xgb_model = joblib.load(XGB_MODEL_PATH)


# ----------------------------
# PREDICT CHURN (DataFrame in, DataFrame out)
# ----------------------------
def predict_churn(input_df: pd.DataFrame) -> pd.DataFrame:

    probabilities = model.predict_proba(input_df)[:, 1]
    predicted_classes = model.predict(input_df)

    def get_tier(p):
        if p >= 0.7:
            return "High"
        elif p >= 0.4:
            return "Medium"
        return "Low"

    def get_recommendation(p):
        if p >= 0.7:
            return "Immediate outreach required"
        elif p >= 0.4:
            return "Monitor closely"
        return "No action needed"

    results = pd.DataFrame({
        "probability": probabilities,
        "predicted_class": predicted_classes,
        "risk_tier": [get_tier(p) for p in probabilities],
        "recommendation": [get_recommendation(p) for p in probabilities],
    })

    return results


# ----------------------------
# EXPLAIN CUSTOMER (single-row DataFrame in, DataFrame out)
# ----------------------------
def explain_customer(customer_df: pd.DataFrame) -> pd.DataFrame:

    try:
        import shap
        explainer = shap.Explainer(xgb_model, customer_df)
        shap_values = explainer(customer_df)
        vals = shap_values.values[0]
    except Exception:
        # Fallback: use raw feature importances
        vals = xgb_model.feature_importances_

    feature_names = customer_df.columns.tolist()

    explanation = pd.DataFrame({
        "Feature": feature_names,
        "Impact": vals
    }).sort_values("Impact", key=abs, ascending=False)

    return explanation
