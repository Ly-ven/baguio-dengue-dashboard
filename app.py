import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Baguio Dengue Forecast Dashboard",
    layout="wide"
)

st.title("Baguio City Dengue Forecast Dashboard")
st.caption("Interactive web-based dashboard for dengue prediction and visualization")

ARTIFACTS_DIR = Path("artifacts")

# =========================
# HELPER FUNCTIONS
# =========================
def safe_read_csv(path):
    return pd.read_csv(path) if path.exists() else None

def safe_read_json(path):
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None

def safe_load_model(path):
    if path.exists():
        return joblib.load(path)
    return None

@st.cache_data
def load_artifacts():
    monthly = safe_read_csv(ARTIFACTS_DIR / "monthly_modeling_dataset.csv")
    model_comparison = safe_read_csv(ARTIFACTS_DIR / "model_comparison.csv")
    feature_importance = safe_read_csv(ARTIFACTS_DIR / "feature_importance.csv")
    feature_sensitivity = safe_read_csv(ARTIFACTS_DIR / "feature_sensitivity.csv")
    forecast = safe_read_csv(ARTIFACTS_DIR / "forecast_5yr.csv")
    barangay_monthly = safe_read_csv(ARTIFACTS_DIR / "barangay_monthly.csv")
    top_barangay_monthly = safe_read_csv(ARTIFACTS_DIR / "top_barangay_monthly.csv")
    meta = safe_read_json(ARTIFACTS_DIR / "meta.json")
    return (
        monthly,
        model_comparison,
        feature_importance,
        feature_sensitivity,
        forecast,
        barangay_monthly,
        top_barangay_monthly,
        meta
    )

monthly, model_comparison, feature_importance, feature_sensitivity, forecast, barangay_monthly, top_barangay_monthly, meta = load_artifacts()
model = safe_load_model(ARTIFACTS_DIR / "best_model.joblib")

# =========================
# SIDEBAR
# =========================
st.sidebar.header("About")
st.sidebar.write(
    "This dashboard displays historical dengue cases, model results, "
    "feature contributions, and forecast outputs from your Google Colab workflow."
)

if meta:
    st.sidebar.success(f"Best Model: {meta.get('best_model', 'Unknown')}")
    st.sidebar.info(f"Outbreak Threshold: {meta.get('outbreak_threshold_cases', 'N/A')}")

st.sidebar.header("Upload files manually (optional)")
uploaded_monthly = st.sidebar.file_uploader("Upload monthly_modeling_dataset.csv", type=["csv"])
uploaded_model_comparison = st.sidebar.file_uploader("Upload model_comparison.csv", type=["csv"])
uploaded_importance = st.sidebar.file_uploader("Upload feature_importance.csv", type=["csv"])
uploaded_sensitivity = st.sidebar.file_uploader("Upload feature_sensitivity.csv", type=["csv"])
uploaded_forecast = st.sidebar.file_uploader("Upload forecast_5yr.csv", type=["csv"])
uploaded_barangay = st.sidebar.file_uploader("Upload barangay_monthly.csv", type=["csv"])
uploaded_top_barangay = st.sidebar.file_uploader("Upload top_barangay_monthly.csv", type=["csv"])
uploaded_meta = st.sidebar.file_uploader("Upload meta.json", type=["json"])
uploaded_model = st.sidebar.file_uploader("Upload best_model.joblib", type=["joblib", "pkl"])

# Use uploaded files if artifacts folder is missing
if uploaded_monthly is not None:
    monthly = pd.read_csv(uploaded_monthly)

if uploaded_model_comparison is not None:
    model_comparison = pd.read_csv(uploaded_model_comparison)

if uploaded_importance is not None:
    feature_importance = pd.read_csv(uploaded_importance)

if uploaded_sensitivity is not None:
    feature_sensitivity = pd.read_csv(uploaded_sensitivity)

if uploaded_forecast is not None:
    forecast = pd.read_csv(uploaded_forecast)

if uploaded_barangay is not None:
    barangay_monthly = pd.read_csv(uploaded_barangay)

if uploaded_top_barangay is not None:
    top_barangay_monthly = pd.read_csv(uploaded_top_barangay)

if uploaded_meta is not None:
    meta = json.load(uploaded_meta)

if uploaded_model is not None:
    model = joblib.load(uploaded_model)

# =========================
# BASIC CHECK
# =========================
if monthly is None:
    st.error("No monthly_modeling_dataset.csv found. Put it inside the artifacts folder or upload it in the sidebar.")
    st.stop()

# Clean date columns if present
for df in [monthly, forecast, top_barangay_monthly]:
    if df is not None and "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Barangay Analytics",
    "Model Results",
    "Feature Transparency",
    "Forecast & Prediction"
])

# =========================
# TAB 1: OVERVIEW
# =========================
with tab1:
    st.subheader("Historical Dengue Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Months", len(monthly))
    if "CHSO_cases" in monthly.columns:
        col2.metric("Total Cases", int(monthly["CHSO_cases"].sum()))
        col3.metric("Average Monthly Cases", round(monthly["CHSO_cases"].mean(), 2))

    if "Date" in monthly.columns and "CHSO_cases" in monthly.columns:
        fig_cases = px.line(
            monthly,
            x="Date",
            y="CHSO_cases",
            markers=True,
            title="Monthly Dengue Cases"
        )
        st.plotly_chart(fig_cases, use_container_width=True)

    if all(col in monthly.columns for col in ["Year", "Month", "CHSO_cases"]):
        heatmap_data = monthly.pivot_table(index="Year", columns="Month", values="CHSO_cases", aggfunc="sum")
        fig_heat = px.imshow(
            heatmap_data,
            aspect="auto",
            text_auto=True,
            title="Year-Month Heatmap of Dengue Cases"
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    if all(col in monthly.columns for col in ["rainfall", "relative_humidity", "CHSO_cases"]):
        fig_bubble = px.scatter(
            monthly,
            x="rainfall",
            y="relative_humidity",
            size="CHSO_cases",
            color="CHSO_cases",
            hover_data=monthly.columns,
            title="Rainfall vs Relative Humidity Sized by Dengue Cases"
        )
        st.plotly_chart(fig_bubble, use_container_width=True)

# =========================
# TAB 2: BARANGAY ANALYTICS
# =========================
with tab2:
    st.subheader("Barangay Analytics")

    if top_barangay_monthly is not None:
        st.write("Top barangay by month")
        st.dataframe(top_barangay_monthly, use_container_width=True)

        if all(col in top_barangay_monthly.columns for col in ["Year", "Top_Barangay", "Top_Barangay_Cases"]):
            fig_tree = px.treemap(
                top_barangay_monthly,
                path=["Year", "Top_Barangay"],
                values="Top_Barangay_Cases",
                color="Top_Barangay_Cases",
                title="Top Barangays by Dengue Cases"
            )
            st.plotly_chart(fig_tree, use_container_width=True)
    else:
        st.info("No barangay file found. Export barangay_monthly.csv and top_barangay_monthly.csv from Colab if you want barangay charts.")

    if barangay_monthly is not None:
        st.write("Barangay monthly records")
        st.dataframe(barangay_monthly.head(50), use_container_width=True)

# =========================
# TAB 3: MODEL RESULTS
# =========================
with tab3:
    st.subheader("Model Comparison")

    if meta:
        st.success(f"Selected Model: {meta.get('best_model', 'Unknown')}")

    if model_comparison is not None:
        st.dataframe(model_comparison, use_container_width=True)

        metric_cols = [c for c in ["accuracy", "precision", "recall", "f1_score"] if c in model_comparison.columns]
        if "model" in model_comparison.columns and metric_cols:
            plot_df = model_comparison.melt(
                id_vars="model",
                value_vars=metric_cols,
                var_name="Metric",
                value_name="Score"
            )
            fig_model = px.bar(
                plot_df,
                x="model",
                y="Score",
                color="Metric",
                barmode="group",
                title="Model Comparison by Metric"
            )
            fig_model.update_yaxes(range=[0, 1])
            st.plotly_chart(fig_model, use_container_width=True)

    st.markdown("""
### How to read the metrics
- **Accuracy**: overall percentage of correct predictions
- **Precision**: when the model says outbreak, how often it is correct
- **Recall / Sensitivity**: among real outbreak months, how many the model correctly catches
- **F1 Score**: balance between precision and recall

Higher values are better.

A model can have high accuracy but still miss outbreak months.  
That is why **precision, recall, and F1 score** must also be checked.
""")

# =========================
# TAB 4: FEATURE TRANSPARENCY
# =========================
with tab4:
    st.subheader("What contributed to the prediction?")

    if feature_importance is not None:
        st.write("Feature Importance")
        st.dataframe(feature_importance, use_container_width=True)

        if all(col in feature_importance.columns for col in ["feature", "importance_mean"]):
            fig_imp = px.bar(
                feature_importance.head(15),
                x="importance_mean",
                y="feature",
                orientation="h",
                title="Top Contributing Features"
            )
            fig_imp.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_imp, use_container_width=True)

    if feature_sensitivity is not None:
        st.write("Sensitivity Analysis")
        st.dataframe(feature_sensitivity, use_container_width=True)

        if all(col in feature_sensitivity.columns for col in ["feature", "delta_probability"]):
            fig_sens = px.bar(
                feature_sensitivity,
                x="feature",
                y="delta_probability",
                title="Effect of +10% Change in Climate Variable on Outbreak Probability"
            )
            st.plotly_chart(fig_sens, use_container_width=True)

# =========================
# TAB 5: FORECAST & LIVE PREDICTION
# =========================
with tab5:
    st.subheader("Forecast")

    if forecast is not None:
        st.dataframe(forecast.head(30), use_container_width=True)

        if all(col in forecast.columns for col in ["Date", "predicted_outbreak_probability"]):
            fig_forecast = px.line(
                forecast,
                x="Date",
                y="predicted_outbreak_probability",
                markers=True,
                title="5-Year Forecasted Outbreak Probability"
            )
            st.plotly_chart(fig_forecast, use_container_width=True)

        if all(col in forecast.columns for col in ["Year", "Month", "predicted_outbreak_probability"]):
            forecast_heat = forecast.pivot_table(
                index="Year",
                columns="Month",
                values="predicted_outbreak_probability"
            )
            fig_forecast_heat = px.imshow(
                forecast_heat,
                aspect="auto",
                text_auto=True,
                title="Forecast Heatmap of Outbreak Probability"
            )
            st.plotly_chart(fig_forecast_heat, use_container_width=True)

    st.subheader("Live Prediction")

    if model is None or meta is None:
        st.info("Live prediction needs best_model.joblib and meta.json.")
    else:
        feature_columns = meta.get("feature_columns", [])

        if not feature_columns:
            st.warning("No feature column list found in meta.json")
        else:
            st.write("Enter the values below to predict one month.")

            input_data = {}
            cols = st.columns(3)

            for i, feat in enumerate(feature_columns):
                default_value = 0.0

                if feat in monthly.columns and pd.api.types.is_numeric_dtype(monthly[feat]):
                    default_value = float(monthly[feat].dropna().mean()) if monthly[feat].dropna().shape[0] > 0 else 0.0

                input_data[feat] = cols[i % 3].number_input(
                    feat,
                    value=default_value,
                    format="%.4f"
                )

            if st.button("Predict"):
                X_new = pd.DataFrame([input_data])
                pred = model.predict(X_new)[0]

                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(X_new)[0][1]
                else:
                    prob = None

                if pred == 1:
                    st.error("Prediction: OUTBREAK")
                else:
                    st.success("Prediction: NON-OUTBREAK")

                if prob is not None:
                    st.write(f"Outbreak Probability: **{prob:.4f}**")