# predict.py
# Loads saved XGBoost pipeline, generates churn predictions,
# and can explain one selected customer using SHAP-style XGBoost contributions.

import joblib
import pandas as pd
import numpy as np
import xgboost as xgb


# LOAD MODEL
# This should be the saved full pipeline:
# preprocessor + XGBoost classifier
model = joblib.load("models/xgboost_model.pkl")


def predict_churn(input_df):
    """
    Generate churn predictions and recommendations for new customer data.

    Parameters:
        input_df (pd.DataFrame): New customer data with the same columns
                                 used during model training.

    Returns:
        pd.DataFrame: probability, predicted_class, risk_tier, recommendation
    """

    # Churn probability
    probabilities = model.predict_proba(input_df)[:, 1]

    # Predicted class: 0 = not churn, 1 = churn
    predicted_classes = model.predict(input_df)

    # Store output rows
    output_rows = []

    for i, prob in enumerate(probabilities):

        # Assign risk tier
        if prob > 0.7:
            risk_tier = "High"
        elif prob >= 0.4:
            risk_tier = "Medium"
        else:
            risk_tier = "Low"

        # Get original customer feature values
        customer = input_df.iloc[i]

        # Rule-based recommendation logic
        if risk_tier == "High" and customer["NumOfProducts"] <= 1:
            recommendation = "Recommend cross-selling additional products"

        elif risk_tier == "High" and customer["IsActiveMember"] == 0:
            recommendation = "Recommend customer engagement campaign (email/call)"

        elif risk_tier == "High" and customer["Balance"] > 50000:
            recommendation = "Offer retention incentive or personalized offer"

        elif risk_tier == "Medium":
            recommendation = "Monitor customer and consider light engagement"

        elif risk_tier == "Low":
            recommendation = "No immediate action required"

        else:
            recommendation = "Review customer profile"

        # Add row to output
        output_rows.append({
            "probability": prob,
            "predicted_class": predicted_classes[i],
            "risk_tier": risk_tier,
            "recommendation": recommendation
        })

    results = pd.DataFrame(output_rows)

    return results


def _get_feature_names():
    """
    Get feature names after preprocessing.

    This assumes the pipeline has:
    - a ColumnTransformer named 'preprocessor'
    - numeric transformer named 'num'
    - categorical transformer named 'cat'
    """

    preprocessor = model.named_steps["preprocessor"]

    numeric_cols = list(preprocessor.transformers_[0][2])
    categorical_cols = list(preprocessor.transformers_[1][2])

    ohe = preprocessor.named_transformers_["cat"]
    encoded_cat_cols = list(ohe.get_feature_names_out(categorical_cols))

    feature_names = numeric_cols + encoded_cat_cols

    return feature_names


def _format_driver(feature_name, contribution_value, customer_row):
    """
    Format a SHAP driver into readable text.

    Example outputs:
    - Age (positive)
    - NumOfProducts (negative)
    - Geography_Germany (positive)
    """

    direction = "positive" if contribution_value > 0 else "negative"

    # For one-hot encoded categorical features, keep the encoded feature name.
    # Example: Geography_Germany means the Germany category affected the prediction.
    return f"{feature_name} ({direction})"


def explain_customer(input_df, top_n=3):
    """
    Generate live SHAP-style explanation for one selected customer or a small batch.

    This function should be used when the app user clicks on one customer
    and wants to see the top drivers behind that customer's churn prediction.

    Parameters:
        input_df (pd.DataFrame or pd.Series): One customer row, or a small DataFrame.
        top_n (int): Number of top drivers to return.

    Returns:
        pd.DataFrame: churn_probability, risk_tier, top_driver_1, top_driver_2, top_driver_3
    """

    # Allow a single row passed as a Series
    if isinstance(input_df, pd.Series):
        input_df = input_df.to_frame().T

    # Safety check
    if not isinstance(input_df, pd.DataFrame):
        raise ValueError("input_df must be a pandas DataFrame or Series.")

    # Get pipeline parts
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    # Transform raw customer data using the same preprocessing used during training
    X_processed = preprocessor.transform(input_df)

    # Get processed feature names
    feature_names = _get_feature_names()

    # Use XGBoost's built-in SHAP contribution method.
    # This avoids compatibility issues that can happen with shap.TreeExplainer.
    dmatrix = xgb.DMatrix(X_processed, feature_names=feature_names)

    shap_contribs = classifier.get_booster().predict(
        dmatrix,
        pred_contribs=True
    )

    # Last column is the bias/base value, not a feature contribution
    shap_values = shap_contribs[:, :-1]

    # Get prediction probabilities
    probabilities = model.predict_proba(input_df)[:, 1]

    output_rows = []

    for i, prob in enumerate(probabilities):

        # Assign risk tier
        if prob > 0.7:
            risk_tier = "High"
        elif prob >= 0.4:
            risk_tier = "Medium"
        else:
            risk_tier = "Low"

        # Find top features by absolute SHAP contribution
        row_shap_values = shap_values[i]
        top_indices = np.argsort(np.abs(row_shap_values))[-top_n:][::-1]

        drivers = []

        for idx in top_indices:
            driver_text = _format_driver(
                feature_names[idx],
                row_shap_values[idx],
                input_df.iloc[i]
            )
            drivers.append(driver_text)

        # Make sure output always has top_driver_1 through top_driver_3
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


def load_shap_explanations():
    """
    Optional helper: load precomputed SHAP explanations if using the CSV approach.
    Not required for live one-customer SHAP.
    """
    return pd.read_csv("models/shap_customer_explanations.csv")


# TEST EXAMPLE
# Only run this part when testing predict.py directly
if __name__ == "__main__":

    # Example input.
    # Replace these columns/values with the exact columns from your training data.
    sample_data = pd.DataFrame([{
        "CreditScore": 650,
        "Geography": "France",
        "Gender": "Male",
        "Age": 40,
        "Tenure": 3,
        "Balance": 60000,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 50000
    }])

    predictions = predict_churn(sample_data)

    print("\nPrediction results:")
    print(predictions)

    explanation = explain_customer(sample_data)

    print("\nSHAP explanation:")
    print(explanation)
