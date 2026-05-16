from shiny import App, ui, render
import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix

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

    ui.p("Upload CSV or use sample dataset."),

    ui.input_file("file", "Upload CSV File"),

    ui.output_text("status"),

    # ----------------------------
    # KPI ROW
    # ----------------------------
    ui.row(
        ui.column(3, ui.output_ui("kpi_total")),
        ui.column(3, ui.output_ui("kpi_high_risk")),
        ui.column(3, ui.output_ui("kpi_avg_risk")),
        ui.column(3, ui.output_ui("kpi_churn_rate"))
    ),

    # ----------------------------
    # MODEL METRICS
    # ----------------------------
    ui.h4("Model Metrics"),
    ui.output_ui("model_metrics"),
    ui.output_plot("conf_matrix"),

    # ----------------------------
    # BUSINESS INSIGHTS
    # ----------------------------
    ui.h4("Business Insights Engine"),
    ui.output_ui("business_insights"),

    # ----------------------------
    # DATA PREVIEW
    # ----------------------------
    ui.h4("Data Preview"),
    ui.output_table("preview"),

    # ----------------------------
    # PREDICTIONS
    # ----------------------------
    ui.h4("Top High-Risk Customers"),
    ui.output_table("predictions"),

    ui.download_button("download_preds", "Download Predictions"),

    # ----------------------------
    # RISK DISTRIBUTION
    # ----------------------------
    ui.h4("Risk Distribution"),
    ui.output_plot("risk_distribution"),

    # ----------------------------
    # CUSTOMER ANALYSIS
    # ----------------------------
    ui.h4("Customer Explorer"),

    ui.input_numeric(
        "customer_row",
        "Customer Row",
        value=0,
        min=0
    ),

    ui.output_table("feature_importance"),

    # ----------------------------
    # AI CUSTOMER NARRATIVE (NEW)
    # ----------------------------
    ui.h4("AI Customer Narrative"),
    ui.output_ui("customer_narrative")
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
    # KPI TOTAL
    # ----------------------------
    @output
    @render.ui
    def kpi_total():
        df = load_data(input.file())
        return ui.card(ui.h5("Total Customers"), ui.h2(str(len(df))))

    # ----------------------------
    # KPI HIGH RISK
    # ----------------------------
    @output
    @render.ui
    def kpi_high_risk():
        df = load_data(input.file())
        X = df.drop(columns=["churn"]) if "churn" in df.columns else df.copy()
        results = predict_churn(X)

        return ui.card(
            ui.h5("High Risk Customers"),
            ui.h2(str((results["risk_tier"] == "High").sum()))
        )

    # ----------------------------
    # KPI AVG RISK
    # ----------------------------
    @output
    @render.ui
    def kpi_avg_risk():
        df = load_data(input.file())
        X = df.drop(columns=["churn"]) if "churn" in df.columns else df.copy()
        results = predict_churn(X)

        avg = round(results["probability"].mean() * 100, 2)

        return ui.card(
            ui.h5("Avg Risk Score"),
            ui.h2(f"{avg}%")
        )

    # ----------------------------
    # KPI CHURN RATE
    # ----------------------------
    @output
    @render.ui
    def kpi_churn_rate():
        df = load_data(input.file())
        X = df.drop(columns=["churn"]) if "churn" in df.columns else df.copy()
        results = predict_churn(X)

        churn = round(results["predicted_class"].mean() * 100, 2)

        return ui.card(
            ui.h5("Predicted Churn Rate"),
            ui.h2(f"{churn}%")
        )

    # ----------------------------
    # MODEL METRICS
    # ----------------------------
    @output
    @render.ui
    def model_metrics():

        df = load_data(input.file())

        if "churn" not in df.columns:
            return ui.p("No ground truth labels available.")

        X = df.drop(columns=["churn"])
        y_true = df["churn"]

        results = predict_churn(X)
        y_pred = results["predicted_class"]

        return ui.card(
            ui.h4("Performance"),
            ui.tags.ul(
                ui.tags.li(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}"),
                ui.tags.li(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.3f}"),
                ui.tags.li(f"Recall: {recall_score(y_true, y_pred, zero_division=0):.3f}"),
                ui.tags.li(f"ROC-AUC: {roc_auc_score(y_true, results['probability']):.3f}")
            )
        )

    # ----------------------------
    # CONFUSION MATRIX
    # ----------------------------
    @output
    @render.plot
    def conf_matrix():

        df = load_data(input.file())

        if "churn" not in df.columns:
            plt.figure()
            plt.text(0.3, 0.5, "No labels")
            return plt.gcf()

        X = df.drop(columns=["churn"])
        y_true = df["churn"]

        results = predict_churn(X)
        y_pred = results["predicted_class"]

        cm = confusion_matrix(y_true, y_pred)

        plt.figure()
        plt.imshow(cm)
        plt.title("Confusion Matrix")

        return plt.gcf()

    # ----------------------------
    # BUSINESS INSIGHTS
    # ----------------------------
    @output
    @render.ui
    def business_insights():

        df = load_data(input.file())
        X = df.drop(columns=["churn"]) if "churn" in df.columns else df.copy()
        results = predict_churn(X)

        churn_rate = results["predicted_class"].mean()

        return ui.card(
            ui.h4("Business Insights"),
            ui.p(f"Estimated churn rate: {churn_rate:.0%}")
        )

    # ----------------------------
    # PREVIEW
    # ----------------------------
    @output
    @render.table
    def preview():
        df = load_data(input.file())
        return df.head()

    # ----------------------------
    # PREDICTIONS
    # ----------------------------
    @output
    @render.table
    def predictions():

        df = load_data(input.file())
        X = df.drop(columns=["churn"]) if "churn" in df.columns else df.copy()
        results = predict_churn(X)

        df["risk_score"] = (results["probability"] * 100).round(2).astype(str) + "%"
        df["risk_tier"] = results["risk_tier"]
        df["recommendation"] = results["recommendation"]

        df["_risk"] = results["probability"]
        df = df.sort_values("_risk", ascending=False)

        return df.head(15)

    # ----------------------------
    # DOWNLOAD
    # ----------------------------
    @render.download(filename="churn_predictions.csv")
    def download_preds():

        df = load_data(input.file())
        X = df.drop(columns=["churn"]) if "churn" in df.columns else df.copy()
        results = predict_churn(X)

        export_df = df.copy()
        export_df["risk_score"] = (results["probability"] * 100).round(2)
        export_df["risk_tier"] = results["risk_tier"]
        export_df["recommendation"] = results["recommendation"]

        yield export_df.to_csv(index=False)

    # ----------------------------
    # RISK DISTRIBUTION
    # ----------------------------
    @output
    @render.plot
    def risk_distribution():

        df = load_data(input.file())
        X = df.drop(columns=["churn"]) if "churn" in df.columns else df.copy()

        results = predict_churn(X)

        plt.figure()
        plt.hist(results["probability"], bins=10)
        return plt.gcf()

    # ----------------------------
    # FEATURE EXPLANATION
    # ----------------------------
    @output
    @render.table
    def feature_importance():

        df = load_data(input.file())
        X = df.drop(columns=["churn"]) if "churn" in df.columns else df.copy()

        customer = X.iloc[[input.customer_row()]]

        return explain_customer(customer)

    # ----------------------------
    # AI CUSTOMER NARRATIVE
    # ----------------------------
    @output
    @render.ui
    def customer_narrative():

        df = load_data(input.file())
        X = df.drop(columns=["churn"]) if "churn" in df.columns else df.copy()

        i = input.customer_row()

        if i >= len(X):
            return ui.card(ui.h4("Invalid customer index"))

        customer = X.iloc[[i]]
        results = predict_churn(customer)

        prob = results["probability"].iloc[0]
        tier = results["risk_tier"].iloc[0]
        rec = results["recommendation"].iloc[0]

        narrative = f"""
        This customer has a {prob:.0%} churn probability ({tier} risk).

        Recommended action: {rec}.
        """

        return ui.card(
            ui.h4("Customer Narrative"),
            ui.p(narrative)
        )


# ----------------------------
# RUN
# ----------------------------
app = App(app_ui, server)