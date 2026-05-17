import os
import joblib
import pandas as pd
import xgboost as xgb
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGISTIC_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "logistic_model.pkl"
)
XGBOOST_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "xgboost_model.pkl"
)
logistic_model = joblib.load(LOGISTIC_MODEL_PATH)
xgb_model = joblib.load(XGBOOST_MODEL_PATH)
def predict_churn(input_data):
    input_df = pd.DataFrame([input_data])
    prediction = logistic_model.predict(input_df)[0]
    probability = logistic_model.predict_proba(input_df)[0][1]
    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }
def explain_customer(input_data):
    input_df = pd.DataFrame([input_data])
    dmatrix = xgb.DMatrix(input_df)
    prediction = xgb_model.predict(dmatrix)
    return prediction.tolist()
