import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# -----------------------
# Load data
# -----------------------
df = pd.read_csv("data/Bank Customer Churn Prediction.csv")
print("Data loaded:", df.shape)

# -----------------------
# Target
# -----------------------
target = "churn"

# Remove customer_id
X = df.drop(columns=[target, "customer_id"])
y = df[target]

# -----------------------
# Convert text columns to numbers
# -----------------------
X = pd.get_dummies(X, drop_first=True)

print("Features ready:", X.shape)

# -----------------------
# Train/test split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------
# Logistic Regression
# -----------------------
log_model = LogisticRegression(max_iter=2000)
log_model.fit(X_train, y_train)

# -----------------------
# XGBoost
# -----------------------
xgb_model = XGBClassifier(
    eval_metric="logloss"
)

xgb_model.fit(X_train, y_train)

# -----------------------
# Save models
# -----------------------
os.makedirs("models", exist_ok=True)

joblib.dump(
    log_model,
    "models/logistic_model.pkl"
)

joblib.dump(
    xgb_model,
    "models/xgboost_model.pkl"
)

print("Models saved successfully.")