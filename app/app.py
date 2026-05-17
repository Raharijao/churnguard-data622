import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
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

from services.predict import (
    predict_churn,
    explain_customer
)


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
# KPI CARD
# ----------------------------
def create_kpi_card(title, value):

    return ui.card(
        ui.card_header(title),
        ui.h2(str(value))
    )


# ----------------------------
# BUSINESS NARRATIVE
# ----------------------------
def generate_customer_narrative(customer, explanation):

    drivers = [
        explanation["top_driver_1"].iloc[0],
        explanation["top_driver_2"].iloc[0],
        explanation["top_driver_3"].iloc[0]
    ]

    risk = explanation["risk_tier"].iloc[0]

    narrative = (
        f"This customer is classified as {risk} churn risk. "
        f"The primary churn drivers are {drivers[0]}, "
        f"{drivers[1]}, and {drivers[2]}. "
        f"Recommended action: proactive retention engagement."
    )

    return narrative


# ----------------------------
# UI
# ----------------------------
app_ui = ui.page_fluid(

    ui.h2(
        "ChurnGuard - Customer Churn Prediction Dashboard"
    ),

    ui.p(
        "AI-powered customer churn prediction and retention analytics."
    ),

    ui.input_file(
        "file",
        "Upload CSV File"
    ),

    ui.output_text("status"),

    ui.hr(),

    # KPI CARDS
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

    ui.hr(),

    # DATA PREVIEW
    ui.h4("Data Preview"),

    ui.output_table("preview"),

    ui.hr(),

    # PREDICTIONS
    ui.h4("Top High-Risk Customers"),

    ui.output_table("predictions"),

    ui.download_button(
        "download_preds",
        "Download Predictions"
    ),

    ui.hr(),

    # METRICS
    ui.h4("Model Evaluation Metrics"),

    ui.output_table("metrics_table"),

    ui.hr(),

    # CONFUSION MATRIX
    ui.h4("Confusion Matrix"),

    ui.output_plot("confusion_matrix_plot"),

    ui.hr(),

    # RISK DISTRIBUTION
    ui.h4("Risk Distribution"),

    ui.output_plot("risk_distribution"),

    ui.hr(),

    # CUSTOMER EXPLANATION
    ui.h4("Customer Explanation"),

    ui.input_numeric(
        "customer_row",
        "Customer Row",
        value=0,
        min=0
    ),

    ui.output_table("feature_importance"),

    ui.hr(),

    # BUSINESS INSIGHTS
    ui.h4("AI Business Narrative"),

    ui.output_text_verbatim("customer_narrative")
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

        return create_kpi_card(
            "Total Customers",
            str(len(df))
        )


    # KPI HIGH RISK
    @output
    @render.ui
    def kpi_high_risk():

        df = load_data(input.file())

        X = (
            df.drop(columns=["churn"])
            if "churn" in df.columns
            else df.copy()
        )

        predictions = X.apply(lambda row: predict_churn(row.to_dict())["prediction"], axis=1)

        high_risk = int(
            (results["risk_tier"] == "High").sum()
        )

        return create_kpi_card(
            "High Risk Customers",
            str(high_risk)
        )


    # KPI AVG RISK
    @output
    @render.ui
    def kpi_avg_risk():

        df = load_data(input.file())

        X = (
            df.drop(columns=["churn"])
            if "churn" in df.columns
            else df.copy()
        )

        predictions = X.apply(lambda row: predict_churn(row.to_dict())["prediction"], axis=1)

        avg_risk = round(
            float(results["probability"].mean()) * 100,
            2
        )

        return create_kpi_card(
            "Average Risk Score",
            f"{avg_risk}%"
        )


    # KPI CHURN RATE
    @output
    @render.ui
    def kpi_churn_rate():

        df = load_data(input.file())

        X = (
            df.drop(columns=["churn"])
            if "churn" in df.columns
            else df.copy()
        )

        predictions = X.apply(lambda row: predict_churn(row.to_dict())["prediction"], axis=1)

        churn_rate = round(
            float(results["predicted_class"].mean()) * 100,
            2
        )

        return create_kpi_card(
            "Predicted Churn Rate",
            f"{churn_rate}%"
        )


    # DATA PREVIEW
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


    # PREDICTIONS
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

            predictions = X.apply(lambda row: predict_churn(row.to_dict())["prediction"], axis=1)

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
                "risk_score",
                "risk_tier",
                "predicted_churn",
                "recommendation"
            ]

            return df[cols].head(15)

        except Exception as e:

            return pd.DataFrame({
                "Error": [str(e)]
            })


    # DOWNLOAD
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

        predictions = X.apply(lambda row: predict_churn(row.to_dict())["prediction"], axis=1)

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


    # METRICS
    @output
    @render.table
    def metrics_table():

        df = load_data(input.file())

        if "churn" not in df.columns:

            return pd.DataFrame({
                "Message": [
                    "No actual churn column found."
                ]
            })

        X = df.drop(columns=["churn"])
        y_true = df["churn"]

        predictions = X.apply(lambda row: predict_churn(row.to_dict())["prediction"], axis=1)

        y_pred = results["predicted_class"]
        y_prob = results["probability"]

        metrics = pd.DataFrame({

            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "ROC-AUC"
            ],

            "Value": [

                round(
                    accuracy_score(
                        y_true,
                        y_pred
                    ),
                    3
                ),

                round(
                    precision_score(
                        y_true,
                        y_pred
                    ),
                    3
                ),

                round(
                    recall_score(
                        y_true,
                        y_pred
                    ),
                    3
                ),

                round(
                    roc_auc_score(
                        y_true,
                        y_prob
                    ),
                    3
                )
            ]
        })

        return metrics


    # CONFUSION MATRIX
    @output
    @render.plot
    def confusion_matrix_plot():

        df = load_data(input.file())

        if "churn" not in df.columns:

            plt.figure()
            plt.text(
                0.2,
                0.5,
                "No churn column found."
            )
            return plt.gcf()

        X = df.drop(columns=["churn"])
        y_true = df["churn"]

        predictions = X.apply(lambda row: predict_churn(row.to_dict())["prediction"], axis=1)

        y_pred = results["predicted_class"]

        cm = confusion_matrix(
            y_true,
            y_pred
        )

        fig, ax = plt.subplots()

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm
        )

        disp.plot(ax=ax)

        return fig


    # RISK DISTRIBUTION
    @output
    @render.plot
    def risk_distribution():

        df = load_data(input.file())

        X = (
            df.drop(columns=["churn"])
            if "churn" in df.columns
            else df.copy()
        )

        predictions = X.apply(lambda row: predict_churn(row.to_dict())["prediction"], axis=1)

        plt.figure()

        plt.hist(
            results["probability"],
            bins=10
        )

        plt.xlabel(
            "Churn Probability"
        )

        plt.ylabel(
            "Customers"
        )

        plt.title(
            "Risk Distribution"
        )

        return plt.gcf()


    # EXPLANATION
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

            row_num = int(
                input.customer_row()
            )

            if row_num >= len(X):

                return pd.DataFrame({
                    "Error": [
                        "Customer row out of range."
                    ]
                })

            customer = X.iloc[[row_num]]

            return explain_customer(customer)

        except Exception as e:

            return pd.DataFrame({
                "Error": [str(e)]
            })


    # BUSINESS NARRATIVE
    @output
    @render.text
    def customer_narrative():

        try:

            df = load_data(input.file())

            X = (
                df.drop(columns=["churn"])
                if "churn" in df.columns
                else df.copy()
            )

            row_num = int(
                input.customer_row()
            )

            customer = X.iloc[[row_num]]

            explanation = explain_customer(
                customer
            )

            return generate_customer_narrative(
                customer,
                explanation
            )

        except Exception as e:

            return str(e)


# ----------------------------
# RUN APP
# ----------------------------
app = App(app_ui, server)