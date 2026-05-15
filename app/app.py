from shiny import App, ui, render
import pandas as pd
import os
import matplotlib.pyplot as plt

from services.predict import predict_churn, explain_customer


# ----------------------------
# LOAD DATA
# ----------------------------
def load_data(file):
    if file is not None:
        return pd.read_csv(file[0]["datapath"])
    else:
        sample_path = os.path.join(
            os.path.dirname(__file__),
            "../data/sample.csv"
        )
        return pd.read_csv(sample_path)


# ----------------------------
# UI
# ----------------------------
app_ui = ui.page_fluid(

    ui.h2("ChurnGuard - Customer Churn Prediction Dashboard"),

    ui.p("Upload CSV or use sample dataset."),

    ui.input_file("file", "Upload CSV File"),

    ui.output_text("status"),

    ui.h4("Data Preview"),
    ui.output_table("preview"),

    ui.h4("Top High-Risk Customers"),
    ui.output_table("predictions"),

    ui.h4("Risk Distribution"),
    ui.output_plot("risk_distribution"),

    ui.h4("Customer Explanation"),
    ui.output_table("feature_importance")
)


# ----------------------------
# SERVER
# ----------------------------
def server(input, output, session):

    # ----------------------------
    # STATUS
    # ----------------------------
    @output
    @render.text
    def status():
        if input.file() is None:
            return "Using sample dataset."
        return "Custom file loaded."

    # ----------------------------
    # PREVIEW
    # ----------------------------
    @output
    @render.table
    def preview():
        try:
            df = load_data(input.file())
            return df.head()
        except Exception as e:
            return pd.DataFrame({"Error": [str(e)]})

    # ----------------------------
    # PREDICTIONS (NO TRANSFORMATION)
    # ----------------------------
    @output
    @render.table
    def predictions():
        try:
            df = load_data(input.file())

            # ONLY REMOVE TARGET COLUMN
            X = df.drop(columns=["churn"]) if "churn" in df.columns else df.copy()

            results = predict_churn(X)

            df["risk_score"] = (results["probability"] * 100).round(2).astype(str) + "%"
            df["predicted_churn"] = results["predicted_class"]
            df["risk_tier"] = results["risk_tier"]
            df["recommendation"] = results["recommendation"]

            df["_risk"] = results["probability"]
            df = df.sort_values("_risk", ascending=False)

            cols = [
                "customer_id",
                "risk_score",
                "risk_tier",
                "predicted_churn",
                "recommendation"
            ]

            cols = [c for c in cols if c in df.columns]

            return df[cols].head(15)

        except Exception as e:
            return pd.DataFrame({"Error": [str(e)]})

    # ----------------------------
    # RISK DISTRIBUTION
    # ----------------------------
    @output
    @render.plot
    def risk_distribution():
        try:
            df = load_data(input.file())

            X = df.drop(columns=["churn"]) if "churn" in df.columns else df.copy()

            results = predict_churn(X)

            plt.figure()
            plt.hist(results["probability"].dropna(), bins=10)
            plt.xlabel("Churn Probability")
            plt.ylabel("Customers")
            plt.title("Risk Distribution")

            return plt.gcf()

        except Exception as e:
            plt.figure()
            plt.text(0.1, 0.5, str(e))
            return plt.gcf()

    # ----------------------------
    # EXPLANATION
    # ----------------------------
    @output
    @render.table
    def feature_importance():
        try:
            df = load_data(input.file())

            X = df.drop(columns=["churn"]) if "churn" in df.columns else df.copy()

            customer = X.iloc[[0]]

            return explain_customer(customer)

        except Exception as e:
            return pd.DataFrame({"Error": [str(e)]})


# ----------------------------
# RUN APP
# ----------------------------
app = App(app_ui, server)