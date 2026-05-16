from shiny import App, ui, render
import pandas as pd
import os
import matplotlib.pyplot as plt
import sys
import numpy as np

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ----------------------------
# HF SAFE IMPORT PATH
# ----------------------------
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from services.predict import predict_churn, explain_customer


# ----------------------------
# SAFE TYPE CONVERTER (FIXES YOUR ERROR)
# ----------------------------
def safe(v):
    if isinstance(v, (np.integer, np.floating)):
        return str(v.item())
    return str(v)


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
# BUSINESS INSIGHTS ENGINE
# ----------------------------
def generate_insight(row):
    if row["risk_tier"] == "High" and row["probability"] > 0.7:
        return "High churn risk driven by strong disengagement signals. Immediate retention action required."
    elif row["risk_tier"] == "High":
        return "High risk customer showing churn tendencies."
    elif row["risk_tier"] == "Medium":
        return "Moderate churn risk. Monitor and engage."
    else:
        return "Low churn risk. Stable customer."


# ----------------------------
# UI
# ----------------------------
app_ui = ui.page_fluid(

    ui.h2("ChurnGuard - Customer Intelligence Dashboard"),

    ui.input_file("file", "Upload CSV File"),
    ui.output_text("status"),

    # ---------------- KPI CARDS ----------------
    ui.row(
        ui.column(3, ui.output_ui("kpi_total")),
        ui.column(3, ui.output_ui("kpi_high_risk")),
        ui.column(3, ui.output_ui("kpi_avg_risk")),
        ui.column(3, ui.output_ui("kpi_churn_rate")),
    ),

    ui.h4("Data Preview"),
    ui.output_table("preview"),

    ui.h4("Top High-Risk Customers"),
    ui.output_table("predictions"),

    ui.download_button("download_preds", "Download Predictions"),

    ui.h4("Risk Distribution"),
    ui.output_plot("risk_distribution"),

    ui.h4("Confusion Matrix (Model Performance)"),
    ui.output_plot("conf_matrix"),

    ui.h4("Customer Explanation"),
    ui.input_numeric("customer_row", "Customer Row", value=0, min=0),
    ui.output_table("feature_importance")
)


# ----------------------------
# SERVER
# ----------------------------
def server(input, output, session):

    # ---------------- STATUS ----------------
    @output
    @render.text
    def status():
        return "Custom file loaded." if input.file() else "Using sample dataset."

    # ---------------- DATA HELPERS ----------------
    def get_data():
        return load_data(input.file())

    def get_X():
        df = get_data()
        return df.drop(columns=["churn"]) if "churn" in df.columns else df

    def get_preds():
        return predict_churn(get_X())

    # ---------------- KPI: TOTAL ----------------
    @output
    @render.ui
    def kpi_total():
        return ui.card(
            ui.h5("Total Customers"),
            ui.h2(safe(len(get_data())))
        )

    # ---------------- KPI: HIGH RISK ----------------
    @output
    @render.ui
    def kpi_high_risk():
        count = (get_preds()["risk_tier"] == "High").sum()
        return ui.card(
            ui.h5("High Risk Customers"),
            ui.h2(safe(count))
        )

    # ---------------- KPI: AVG RISK ----------------
    @output
    @render.ui
    def kpi_avg_risk():
        avg = get_preds()["probability"].mean() * 100
        return ui.card(
            ui.h5("Avg Risk Score"),
            ui.h2(safe(round(avg, 2)) + "%")
        )

    # ---------------- KPI: CHURN RATE ----------------
    @output
    @render.ui
    def kpi_churn_rate():
        churn = get_preds()["predicted_class"].mean() * 100
        return ui.card(
            ui.h5("Predicted Churn Rate"),
            ui.h2(safe(round(churn, 2)) + "%")
        )

    # ---------------- PREVIEW ----------------
    @output
    @render.table
    def preview():
        return get_data().head()

    # ---------------- PREDICTIONS ----------------
    @output
    @render.table
    def predictions():

        df = get_data().copy()
        preds = get_preds()

        df["risk_score"] = (preds["probability"] * 100).round(2)
        df["risk_tier"] = preds["risk_tier"]
        df["recommendation"] = preds["recommendation"]
        df["insight"] = preds.apply(generate_insight, axis=1)

        df["_risk"] = preds["probability"]
        df = df.sort_values("_risk", ascending=False)

        return df.head(15)

    # ---------------- DOWNLOAD ----------------
    @render.download(filename="churn_predictions.csv")
    def download_preds():

        df = get_data().copy()
        preds = get_preds()

        df["risk_score"] = (preds["probability"] * 100).round(2)
        df["risk_tier"] = preds["risk_tier"]
        df["recommendation"] = preds["recommendation"]

        yield df.to_csv(index=False)

    # ---------------- RISK DISTRIBUTION ----------------
    @output
    @render.plot
    def risk_distribution():

        preds = get_preds()

        plt.figure()
        plt.hist(preds["probability"], bins=10)
        plt.title("Risk Distribution")
        return plt.gcf()

    # ---------------- CONFUSION MATRIX ----------------
    @output
    @render.plot
    def conf_matrix():

        df = get_data()

        if "churn" not in df.columns:
            plt.figure()
            plt.text(0.2, 0.5, "No ground truth available")
            return plt.gcf()

        preds = get_preds()

        cm = confusion_matrix(df["churn"], preds["predicted_class"])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)

        fig, ax = plt.subplots()
        disp.plot(ax=ax)

        return fig

    # ---------------- EXPLANATION ----------------
    @output
    @render.table
    def feature_importance():

        df = get_data()
        X = df.drop(columns=["churn"]) if "churn" in df.columns else df

        customer = X.iloc[[input.customer_row()]]

        return explain_customer(customer)


# ----------------------------
# RUN APP
# ----------------------------
app = App(app_ui, server)