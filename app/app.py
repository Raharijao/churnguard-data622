import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from shiny import App, ui, render
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from services.predict import predict_churn, explain_customer


# ----------------------------
# LOAD DATA
# ----------------------------
def load_data(file):
    if file is not None:
        return pd.read_csv(file[0]["datapath"])
    sample_path = os.path.join(
        os.path.dirname(__file__),
        "../data/sample.csv"
    )
    return pd.read_csv(sample_path)


# ----------------------------
# GET PREDICTIONS (helper used everywhere)
# ----------------------------
def get_predictions(df):
    X = (
        df.drop(columns=["churn"])
        if "churn" in df.columns
        else df.copy()
    )
    return predict_churn(X)


# ----------------------------
# KPI CARD
# ----------------------------
def create_kpi_card(title, value):
    return ui.card(
        ui.card_header(title),
        ui.h2(str(value))
    )


# ----------------------------
# UI
# ----------------------------
app_ui = ui.page_fluid(

    ui.h2("ChurnGuard - Customer Churn Prediction Dashboard"),

    ui.p("AI-powered customer churn prediction and retention analytics."),

    ui.input_file("file", "Upload CSV File"),

    ui.output_text("status"),

    ui.hr(),

    # KPI CARDS
    ui.row(
        ui.column(3, ui.output_ui("kpi_total")),
        ui.column(3, ui.output_ui("kpi_high_risk")),
        ui.column(3, ui.output_ui("kpi_avg_risk")),
        ui.column(3, ui.output_ui("kpi_churn_rate")),
    ),

    ui.hr(),

    ui.h4("Data Preview"),
    ui.output_table("preview"),

    ui.hr(),

    ui.h4("Top High-Risk Customers"),
    ui.output_table("predictions"),
    ui.download_button("download_preds", "Download Predictions"),

    ui.hr(),

    ui.h4("Model Evaluation Metrics"),
    ui.output_table("metrics_table"),

    ui.hr(),

    ui.h4("Confusion Matrix"),
    ui.output_plot("confusion_matrix_plot"),

    ui.hr(),

    ui.h4("Risk Distribution"),
    ui.output_plot("risk_distribution"),

    ui.hr(),

    ui.h4("Customer Explanation"),
    ui.input_numeric("customer_row", "Customer Row", value=0, min=0),
    ui.output_table("feature_importance"),
)


# ----------------------------
# SERVER
# ----------------------------
def server(input, output, session):

    # STATUS
    @output
    @render.text
    def status():
        if input.file() is None:
            return "Using sample dataset."
        return "Custom dataset loaded."

    # KPI TOTAL
    @output
    @render.ui
    def kpi_total():
        df = load_data(input.file())
        return create_kpi_card("Total Customers", len(df))

    # KPI HIGH RISK
    @output
    @render.ui
    def kpi_high_risk():
        try:
            df = load_data(input.file())
            results = get_predictions(df)
            high_risk = int((results["risk_tier"] == "High").sum())
            return create_kpi_card("High Risk Customers", high_risk)
        except Exception as e:
            return create_kpi_card("High Risk Customers", f"Error: {e}")

    # KPI AVG RISK
    @output
    @render.ui
    def kpi_avg_risk():
        try:
            df = load_data(input.file())
            results = get_predictions(df)
            avg_risk = round(float(results["probability"].mean()) * 100, 2)
            return create_kpi_card("Average Risk Score", f"{avg_risk}%")
        except Exception as e:
            return create_kpi_card("Average Risk Score", f"Error: {e}")

    # KPI CHURN RATE
    @output
    @render.ui
    def kpi_churn_rate():
        try:
            df = load_data(input.file())
            results = get_predictions(df)
            churn_rate = round(float(results["predicted_class"].mean()) * 100, 2)
            return create_kpi_card("Predicted Churn Rate", f"{churn_rate}%")
        except Exception as e:
            return create_kpi_card("Predicted Churn Rate", f"Error: {e}")

    # DATA PREVIEW
    @output
    @render.table
    def preview():
        try:
            df = load_data(input.file())
            return df.head()
        except Exception as e:
            return pd.DataFrame({"Error": [str(e)]})

    # PREDICTIONS TABLE
    @output
    @render.table
    def predictions():
        try:
            df = load_data(input.file())
            results = get_predictions(df)

            out = df.copy()
            out["risk_score"] = (results["probability"] * 100).round(2).astype(str) + "%"
            out["predicted_churn"] = results["predicted_class"].values
            out["risk_tier"] = results["risk_tier"].values
            out["recommendation"] = results["recommendation"].values
            out["_risk"] = results["probability"].values

            out = out.sort_values("_risk", ascending=False)

            cols = ["risk_score", "risk_tier", "predicted_churn", "recommendation"]
            if "customer_id" in out.columns:
                cols = ["customer_id"] + cols

            return out[cols].head(15)

        except Exception as e:
            return pd.DataFrame({"Error": [str(e)]})

    # DOWNLOAD
    @render.download(filename="churn_predictions.csv")
    def download_preds():
        df = load_data(input.file())
        results = get_predictions(df)
        export_df = df.copy()
        export_df["risk_score"] = (results["probability"] * 100).round(2)
        export_df["risk_tier"] = results["risk_tier"].values
        export_df["recommendation"] = results["recommendation"].values
        yield export_df.to_csv(index=False)

    # METRICS TABLE
    @output
    @render.table
    def metrics_table():
        try:
            df = load_data(input.file())

            if "churn" not in df.columns:
                return pd.DataFrame({"Message": ["No actual churn column found. Upload a dataset with a 'churn' column to see metrics."]})

            results = get_predictions(df)
            y_true = df["churn"]
            y_pred = results["predicted_class"]
            y_prob = results["probability"]

            return pd.DataFrame({
                "Metric": ["Accuracy", "Precision", "Recall", "ROC-AUC"],
                "Value": [
                    round(accuracy_score(y_true, y_pred), 3),
                    round(precision_score(y_true, y_pred, zero_division=0), 3),
                    round(recall_score(y_true, y_pred, zero_division=0), 3),
                    round(roc_auc_score(y_true, y_prob), 3),
                ]
            })
        except Exception as e:
            return pd.DataFrame({"Error": [str(e)]})

    # CONFUSION MATRIX
    @output
    @render.plot
    def confusion_matrix_plot():
        try:
            df = load_data(input.file())

            if "churn" not in df.columns:
                plt.figure()
                plt.text(0.2, 0.5, "No churn column found.")
                return plt.gcf()

            results = get_predictions(df)
            y_true = df["churn"]
            y_pred = results["predicted_class"]

            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax)
            return fig

        except Exception as e:
            plt.figure()
            plt.text(0.1, 0.5, str(e))
            return plt.gcf()

    # RISK DISTRIBUTION
    @output
    @render.plot
    def risk_distribution():
        try:
            df = load_data(input.file())
            results = get_predictions(df)

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

    # FEATURE IMPORTANCE / EXPLANATION
    @output
    @render.table
    def feature_importance():
        try:
            df = load_data(input.file())
            X = (
                df.drop(columns=["churn"])
                if "churn" in df.columns
                else df.copy()
            )
            row_num = int(input.customer_row())
            if row_num >= len(X):
                return pd.DataFrame({"Error": ["Customer row out of range."]})
            customer = X.iloc[[row_num]]
            return explain_customer(customer)
        except Exception as e:
            return pd.DataFrame({"Error": [str(e)]})


# ----------------------------
# RUN APP
# ----------------------------
app = App(app_ui, server)