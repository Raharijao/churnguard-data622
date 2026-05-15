# predict.py
# Loads saved XGBoost pipeline, generates churn predictions,
# and explains customers using SHAP-style contributions.

import joblib
import pandas as pd
import numpy as np
import xgboost as xgb


# ----------------------------
# LOAD MODEL
# ----------------------------
model = joblib.load("models/xgboost_model.pkl")


# ----------------------------
# SCHEMA ALIGNMENT (FINAL FIX)
# ----------------------------
def align_to_model_schema(df):
    """
    Forces input dataframe to match exactly what the model expects.
    Prevents ALL missing column errors.
    """

    expected_cols = model.named_steps["preprocessor"].feature_names_in_

    df = df.copy()

    # Add missing columns as 0
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    # Keep correct order only
    df = df[expected_cols]

    return df


# ----------------------------
# MAIN PREDICTION FUNCTION
# ----------------------------
def predict_churn(input_df):

    # Align schema BEFORE anything else
    input_df = align_to_model_schema(input_df)

    probabilities = model.predict_proba(input_df)[:, 1]
    predicted_classes = model.predict(input_df)

    output_rows = []

    for i, prob in enumerate(probabilities):

        # Risk tier
        if prob > 0.7:
            risk_tier = "High"
        elif prob >= 0.4:
            risk_tier = "Medium"
        else:
            risk_tier = "Low"

        customer = input_df.iloc[i]

        # Safe rule logic (no KeyErrors possible)
        num_products = customer.get("NumOfProducts", 0)
        is_active = customer.get("IsActiveMember", 0)
        balance = customer.get("Balance", 0)

        if risk_tier == "High" and num_products <= 1:
            recommendation = "Recommend cross-selling additional products"

        elif risk_tier == "High" and is_active == 0:
            recommendation = "Recommend customer engagement campaign (email/call)"

        elif risk_tier == "High" and balance > 50000:
            recommendation = "Offer retention incentive or personalized offer"

        elif risk_tier == "Medium":
            recommendation = "Monitor customer and consider light engagement"

        else:
            recommendation = "No immediate action required"

        output_rows.append({
            "probability": prob,
            "predicted_class": predicted_classes[i],
            "risk_tier": risk_tier,
            "recommendation": recommendation
        })

    return pd.DataFrame(output_rows)


# ----------------------------
# FEATURE NAMES (SHAP)
# ----------------------------
def _get_feature_names():
    preprocessor = model.named_steps["preprocessor"]

    numeric_cols = list(preprocessor.transformers_[0][2])
    categorical_cols = list(preprocessor.transformers_[1][2])

    ohe = preprocessor.named_transformers_["cat"]
    encoded_cat_cols = list(ohe.get_feature_names_out(categorical_cols))

    return numeric_cols + encoded_cat_cols


# ----------------------------
# DRIVER FORMATTER
# ----------------------------
def _format_driver(feature_name, value):
    direction = "positive" if value > 0 else "negative"
    return f"{feature_name} ({direction})"


# ----------------------------
# SHAP EXPLANATION
# ----------------------------
def explain_customer(input_df, top_n=3):

    if isinstance(input_df, pd.Series):
        input_df = input_df.to_frame().T

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    # ALIGN schema BEFORE transform
    input_df = align_to_model_schema(input_df)

    X_processed = preprocessor.transform(input_df)

    feature_names = _get_feature_names()

    dmatrix = xgb.DMatrix(X_processed, feature_names=feature_names)

    shap_values = classifier.get_booster().predict(
        dmatrix,
        pred_contribs=True
    )

    shap_values = shap_values[:, :-1]  # remove bias term

    probabilities = model.predict_proba(input_df)[:, 1]

    output_rows = []

    for i, prob in enumerate(probabilities):

        if prob > 0.7:
            risk_tier = "High"
        elif prob >= 0.4:
            risk_tier = "Medium"
        else:
            risk_tier = "Low"

        row = shap_values[i]
        top_idx = np.argsort(np.abs(row))[-top_n:][::-1]

        drivers = [_format_driver(feature_names[j], row[j]) for j in top_idx]

        while len(drivers) < 3:
            drivers.append(None)

        output_rows.append({
            "churn_probability": prob,
            "risk_tier": risk_tier,
            "top_driver_1": drivers[0],
            "top_driver_2": drivers[1],
            "top_driver_3": drivers[2]
        })

    return pd.DataFrame(output_rows)


# ----------------------------
# OPTIONAL LOADER
# ----------------------------
def load_shap_explanations():
    return pd.read_csv("models/shap_customer_explanations.csv")


# ----------------------------
# TEST BLOCK
# ----------------------------
if __name__ == "__main__":

    sample_data = pd.DataFrame([{
        "customer_id": 15634602,
        "credit_score": 650,
        "country": "France",
        "gender": "Male",
        "age": 40,
        "tenure": 3,
        "balance": 60000,
        "products_number": 2,
        "credit_card": 1,
        "active_member": 1,
        "estimated_salary": 50000
    }])

    print(predict_churn(sample_data))
    print(explain_customer(sample_data))