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

    ui.p(
        "Upload CSV or use sample dataset."
    ),

    ui.input_file(
        "file",
        "Upload CSV File"
    ),

    ui.output_text("status"),

    # ----------------------------
    # KPI CARDS
    # ----------------------------
    ui.row(

        ui.column(
            3,
            ui.output_ui("kpi_total")
        ),

        ui.column(
            3,
            ui.output_ui("kpi_high_risk")
        ),

        ui.column(
            3,
            ui.output_ui("kpi_avg_risk")
        ),

        ui.column(
            3,
            ui.output_ui("kpi_churn_rate")
        )
    ),

    # ----------------------------
    # PREVIEW
    # ----------------------------
    ui.h4("Data Preview"),

    ui.output_table("preview"),

    # ----------------------------
    # PREDICTIONS
    # ----------------------------
    ui.h4("Top High-Risk Customers"),

    ui.output_table("predictions"),

    ui.download_button(
        "download_preds",
        "Download Predictions"
    ),

    # ----------------------------
    # RISK DISTRIBUTION
    # ----------------------------
    ui.h4("Risk Distribution"),

    ui.output_plot("risk_distribution"),

    # ----------------------------
    # CUSTOMER EXPLANATION
    # ----------------------------
    ui.h4("Customer Explanation"),

    ui.input_numeric(
        "customer_row",
        "Customer Row",
        value=0,
        min=0
    ),

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
    # KPI: TOTAL CUSTOMERS
    # ----------------------------
    @output
    @render.ui
    def kpi_total():

        df = load_data(input.file())

        return ui.card(
            ui.h5("Total Customers"),
            ui.h2(str(len(df)))
        )

    # ----------------------------
    # KPI: HIGH RISK CUSTOMERS
    # ----------------------------
    @output
    @render.ui
    def kpi_high_risk():

        df = load_data(input.file())

        X = (
            df.drop(columns=["churn"])
            if "churn" in df.columns
            else df.copy()
        )

        results = predict_churn(X)

        high_risk_count = (
            results["risk_tier"] == "High"
        ).sum()

        return ui.card(
            ui.h5("High Risk Customers"),
            ui.h2(str(high_risk_count))
        )

    # ----------------------------
    # KPI: AVG RISK SCORE
    # ----------------------------
    @output
    @render.ui
    def kpi_avg_risk():

        df = load_data(input.file())

        X = (
            df.drop(columns=["churn"])
            if "churn" in df.columns
            else df.copy()
        )

        results = predict_churn(X)

        avg_risk = round(
            results["probability"].mean() * 100,
            2
        )

        return ui.card(
            ui.h5("Avg Risk Score"),
            ui.h2(f"{avg_risk}%")
        )

    # ----------------------------
    # KPI: PREDICTED CHURN RATE
    # ----------------------------
    @output
    @render.ui
    def kpi_churn_rate():

        df = load_data(input.file())

        X = (
            df.drop(columns=["churn"])
            if "churn" in df.columns
            else df.copy()
        )

        results = predict_churn(X)

        churn_rate = round(
            results["predicted_class"].mean() * 100,
            2
        )

        return ui.card(
            ui.h5("Predicted Churn Rate"),
            ui.h2(f"{churn_rate}%")
        )

    # ----------------------------
    # DATA PREVIEW
    # ----------------------------
    @output
    @render.table
    def preview():

        try:

            df = load_data(input.file())

            return df.head()

        except Exception as e:

            return pd.DataFrame({
                "Error": [str(e)]
            })

    # ----------------------------
    # PREDICTIONS
    # ----------------------------
    @output
    @render.table
    def predictions():

        try:

            df = load_data(input.file())

            X = (
                df.drop(columns=["churn"])
                if "churn" in df.columns
                else df.copy()
            )

            results = predict_churn(X)

            df["risk_score"] = (
                results["probability"] * 100
            ).round(2).astype(str) + "%"

            df["predicted_churn"] = (
                results["predicted_class"]
            )

            df["risk_tier"] = (
                results["risk_tier"]
            )

            df["recommendation"] = (
                results["recommendation"]
            )

            df["_risk"] = (
                results["probability"]
            )

            df = df.sort_values(
                "_risk",
                ascending=False
            )

            cols = [
                "customer_id",
                "risk_score",
                "risk_tier",
                "predicted_churn",
                "recommendation"
            ]

            cols = [
                c for c in cols
                if c in df.columns
            ]

            return df[cols].head(15)

        except Exception as e:

            return pd.DataFrame({
                "Error": [str(e)]
            })

    # ----------------------------
    # DOWNLOAD PREDICTIONS
    # ----------------------------
    @render.download(
        filename="churn_predictions.csv"
    )
    def download_preds():

        df = load_data(input.file())

        X = (
            df.drop(columns=["churn"])
            if "churn" in df.columns
            else df.copy()
        )

        results = predict_churn(X)

        export_df = df.copy()

        export_df["risk_score"] = (
            results["probability"] * 100
        ).round(2)

        export_df["risk_tier"] = (
            results["risk_tier"]
        )

        export_df["recommendation"] = (
            results["recommendation"]
        )

        yield export_df.to_csv(index=False)

    # ----------------------------
    # RISK DISTRIBUTION
    # ----------------------------
    @output
    @render.plot
    def risk_distribution():

        try:

            df = load_data(input.file())

            X = (
                df.drop(columns=["churn"])
                if "churn" in df.columns
                else df.copy()
            )

            results = predict_churn(X)

            plt.figure()

            plt.hist(
                results["probability"].dropna(),
                bins=10
            )

            plt.xlabel("Churn Probability")
            plt.ylabel("Customers")
            plt.title("Risk Distribution")

            return plt.gcf()

        except Exception as e:

            plt.figure()

            plt.text(
                0.1,
                0.5,
                str(e)
            )

            return plt.gcf()

    # ----------------------------
    # CUSTOMER EXPLANATION
    # ----------------------------
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

            customer = X.iloc[
                [input.customer_row()]
            ]

            return explain_customer(customer)

        except Exception as e:

            return pd.DataFrame({
                "Error": [str(e)]
            })


# ----------------------------
# RUN APP
# ----------------------------
app = App(app_ui, server)