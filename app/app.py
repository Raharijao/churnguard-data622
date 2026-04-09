from shiny import App, ui, render
import pandas as pd
import joblib
import os
import shap
import matplotlib.pyplot as plt

# load model safely
model_path = "models/logistic_model.pkl"
model = joblib.load(model_path)

app_ui = ui.page_fluid(
    ui.h2("Churn Prediction App"),
    
    ui.input_file("file", "Upload CSV File"),
    
    ui.output_text("status"),
    
    ui.h4("Data Preview"),
    ui.output_table("preview"),
    
    ui.h4("Top High-Risk Customers"),
    ui.output_table("predictions"),

    ui.h4("Risk Distribution"),
    ui.output_plot("risk_distribution"),

    ui.h4("Feature Importance (Why customers churn)"),
    ui.output_plot("feature_importance")
)

def server(input, output, session):
    
    @output
    @render.text
    def status():
        if input.file() is None:
            return "Please upload a CSV file."
        return "File uploaded successfully."

    @output
    @render.table
    def preview():
        file = input.file()
        if file is None:
            return
        
        df = pd.read_csv(file[0]["datapath"])
        return df.head()

    @output
    @render.table
    def predictions():
        file = input.file()
        if file is None:
            return
        
        df = pd.read_csv(file[0]["datapath"])

        try:
            X = df.drop(columns=["churn"]) if "churn" in df.columns else df.copy()

            # probabilities
            proba = model.predict_proba(X)[:, 1]

            # formatted risk score
            df["risk_score"] = (proba * 100).round(2).astype(str) + "%"

            # numeric for sorting
            df["_risk_score_numeric"] = proba

            # predicted churn
            df["predicted_churn"] = (proba >= 0.5).astype(int)

            # risk tier
            def risk_tier(p):
                if p >= 0.7:
                    return "High"
                elif p >= 0.4:
                    return "Medium"
                else:
                    return "Low"

            df["risk_tier"] = df["_risk_score_numeric"].apply(risk_tier)

            # sorting highest risk first
            df = df.sort_values(by="_risk_score_numeric", ascending=False)

            # columns to display
            cols_to_show = [
                "customer_id",
                "risk_score",
                "risk_tier",
                "predicted_churn",
                "churn"
            ]
            cols_to_show = [c for c in cols_to_show if c in df.columns]

            return df[cols_to_show].head(15)

        except Exception as e:
            return pd.DataFrame({"Error": [str(e)]})

    # Risk Distribution
    @output
    @render.plot
    def risk_distribution():
        file = input.file()
        if file is None:
            return
        
        df = pd.read_csv(file[0]["datapath"])
        X = df.drop(columns=["churn"]) if "churn" in df.columns else df.copy()

        try:
            proba = model.predict_proba(X)[:, 1]

            plt.figure()
            plt.hist(proba, bins=10)
            plt.xlabel("Churn Probability")
            plt.ylabel("Number of Customers")
            plt.title("Risk Distribution")

            return plt.gcf()

        except Exception as e:
            plt.figure()
            plt.text(0.1, 0.5, str(e))
            return plt.gcf()

    # SHAP Feature 
    @output
    @render.plot
    def feature_importance():
        file = input.file()
        if file is None:
            return
        
        df = pd.read_csv(file[0]["datapath"])
        X = df.drop(columns=["churn"]) if "churn" in df.columns else df.copy()

        try:
            # sample for performance
            X_sample = X.sample(min(100, len(X)), random_state=42)

            # extract pipeline parts
            preprocessor = model.named_steps["preprocessor"]
            classifier = model.named_steps["classifier"]

            # transform data (make numeric)
            X_transformed = preprocessor.transform(X_sample)

            # feature names
            feature_names = preprocessor.get_feature_names_out()

            # SHAP for linear model
            explainer = shap.LinearExplainer(classifier, X_transformed)
            shap_values = explainer(X_transformed)

            # plot
            plt.figure()
            shap.summary_plot(
                shap_values,
                X_transformed,
                feature_names=feature_names,
                show=False
            )

            return plt.gcf()

        except Exception as e:
            plt.figure()
            plt.text(0.1, 0.5, "SHAP Error:\n" + str(e))
            return plt.gcf()

app = App(app_ui, server)
