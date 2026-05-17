import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "logistic_model.pkl"
)

XGB_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "xgboost_model.pkl"
)

model = joblib.load(MODEL_PATH)
xgb_model = joblib.load(XGB_MODEL_PATH)


def predict_churn(input_data):

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }


def explain_customer(input_data):

    input_df = pd.DataFrame([input_data])

    prediction = xgb_model.predict(input_df)

    return prediction.tolist()
