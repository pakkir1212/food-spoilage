

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Locate the project root relative to this app.py file
PROJECT = Path(__file__).resolve().parent.parent

RESULTS = PROJECT / "results"
MODEL_PATH = PROJECT / "models" / "best_model.joblib"
METADATA_PATH = PROJECT / "models" / "model_metadata.json"
DATA_PATH = PROJECT / "data" / "food_spoilage.csv"
```

st.set_page_config(
    page_title="Food Spoilage | ML Analytics",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------- Professional visual system ----------
st.markdown("""
<style>
    :root {
        --navy: #142D4E;
        --teal: #087F8C;
        --teal-light: #E6F4F3;
        --ink: #1D2939;
        --muted: #667085;
        --line: #E4E7EC;
        --surface: #FFFFFF;
        --canvas: #F5F7FA;
    }

    .stApp {
        background: var(--canvas);
        color: var(--ink);
    }

    [data-testid="stHeader"] {
        background: rgba(245,247,250,0.94);
    }

    .block-container {
        max-width: 1440px;
        padding-top: 1.6rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: var(--navy) !important;
        letter-spacing: -0.025em;
    }

    h1 {
        font-size: 2.05rem !important;
        font-weight: 750 !important;
    }

    h2 {
        font-size: 1.4rem !important;
        font-weight: 700 !important;
    }

    h3 {
        font-size: 1.05rem !important;
        font-weight: 650 !important;
    }

    p, label, li {
        color: var(--ink);
    }

    .hero {
        background: linear-gradient(115deg, #142D4E 0%, #175D73 68%, #087F8C 100%);
        border-radius: 18px;
        padding: 30px 34px;
        color: white;
        margin: 0 0 24px 0;
        box-shadow: 0 8px 24px rgba(20,45,78,0.12);
    }

    .hero-eyebrow {
        color: #BDE9E7;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-size: 0.72rem;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .hero-title {
        color: #FFFFFF;
        font-size: 2rem;
        line-height: 1.2;
        font-weight: 750;
        margin-bottom: 10px;
    }

    .hero-subtitle {
        color: #E1EDF4;
        font-size: 1rem;
        line-height: 1.65;
        max-width: 850px;
    }

    .section-intro {
        color: var(--muted);
        margin-top: -8px;
        margin-bottom: 20px;
        font-size: 0.94rem;
    }

    [data-testid="stMetric"] {
        background: white;
        border: 1px solid var(--line);
        border-radius: 13px;
        padding: 17px 19px;
        box-shadow: 0 2px 8px rgba(16,24,40,0.025);
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted);
        font-size: 0.83rem;
    }

    [data-testid="stMetricValue"] {
        color: var(--navy);
        font-weight: 750;
    }

    [data-testid="stTabs"] [role="tablist"] {
        gap: 8px;
        border-bottom: 1px solid var(--line);
    }

    [data-testid="stTabs"] button[role="tab"] {
        padding: 13px 18px;
        color: #526174;
        font-weight: 600;
    }

    [data-testid="stTabs"] button[aria-selected="true"] {
        color: var(--teal);
        border-bottom-color: var(--teal);
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: white;
        border-radius: 14px;
        border-color: var(--line);
    }

    [data-testid="stExpander"] {
        background: white;
        border: 1px solid var(--line);
        border-radius: 10px;
        overflow: hidden;
    }

    .notice {
        background: #EAF3FA;
        border-left: 4px solid #3978A8;
        border-radius: 7px;
        padding: 13px 16px;
        color: #244764;
        font-size: 0.88rem;
        line-height: 1.6;
        margin: 12px 0 18px 0;
    }

    .result-card {
        background: white;
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 18px;
        height: 100%;
    }

    .result-card-title {
        color: var(--navy);
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .result-card-subtitle {
        color: var(--muted);
        font-size: 0.82rem;
        margin-bottom: 12px;
    }

    .prediction-output {
        background: var(--teal-light);
        border: 1px solid #B8E1DD;
        border-radius: 12px;
        padding: 22px;
        margin-top: 18px;
    }

    .prediction-label {
        color: #23666A;
        font-size: 0.86rem;
        font-weight: 600;
    }

    .prediction-value {
        color: #075E62;
        font-size: 2rem;
        font-weight: 750;
        margin-top: 5px;
    }

    div.stButton > button,
    div.stFormSubmitButton > button {
        background: var(--teal);
        color: white;
        border: 1px solid var(--teal);
        border-radius: 9px;
        padding: 0.55rem 1.2rem;
        font-weight: 650;
        transition: background 0.15s ease;
    }

    div.stButton > button:hover,
    div.stFormSubmitButton > button:hover {
        background: #066B76;
        color: white;
        border-color: #066B76;
    }

    [data-testid="stDownloadButton"] button {
        border-radius: 8px;
    }

    hr {
        border-color: var(--line);
    }

    .footer {
        text-align: center;
        color: var(--muted);
        font-size: 0.8rem;
        padding-top: 22px;
    }

    @media (max-width: 700px) {
        .hero {
            padding: 23px 20px;
        }
        .hero-title {
            font-size: 1.55rem;
        }
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)


# ---------- Load project artifacts ----------
if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    st.error("The saved model or its metadata is missing. Run the training script first.")
    st.stop()

with open(METADATA_PATH, "r", encoding="utf-8") as file:
    metadata = json.load(file)

model = joblib.load(MODEL_PATH)


# ---------- Header ----------
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Machine Learning Research Dashboard</div>
    <div class="hero-title">Food Spoilage Prediction</div>
    <div class="hero-subtitle">
        An experimental analytics platform for exploring spoilage observations,
        comparing regression models, and testing model predictions using
        preservative and storage-condition inputs.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="notice"><strong>Research limitation:</strong> '
    'This prototype uses only 16 observations. Model estimates are preliminary '
    'and are not validated food-safety assessments, expiry guarantees, '
    'or consumption recommendations.</div>',
    unsafe_allow_html=True
)

overview_tab, prediction_tab, results_tab, dataset_tab = st.tabs([
    "Overview", "Prediction", "Model Results", "Dataset"
])


# ---------- Overview ----------
with overview_tab:
    st.subheader("Project Overview")
    st.markdown(
        '<div class="section-intro">A concise summary of the dataset, '
        'model selection, and experimental setup.</div>',
        unsafe_allow_html=True
    )

    metrics_file = RESULTS / "training_metrics.csv"
    metrics = pd.read_csv(metrics_file) if metrics_file.exists() else pd.DataFrame()

    data = pd.read_csv(DATA_PATH) if DATA_PATH.exists() else pd.DataFrame()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Observations", len(data) if not data.empty else metadata.get("observations", "N/A"))
    m2.metric("Input Features", len(metadata.get("features", [])))
    m3.metric("Selected Model", metadata.get("selected_model", "N/A"))
    m4.metric("Evaluation Method", "LOOCV")

    st.write("")
    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        with st.container(border=True):
            st.markdown("#### Study Design")
            st.markdown("""
            **Prediction task:** numerical regression

            **Target variable:** `Spoilage`

            **Input features:**
            - Preservative formulation
            - Vacuum pressure
            - Contamination level
            - Cooling temperature

            **Evaluation:** Leave-One-Out Cross-Validation (LOOCV)
            """)

    with right:
        with st.container(border=True):
            st.markdown("#### Current Model Performance")
            if not metrics.empty and "Model" in metrics.columns:
                preferred = ["Model", "MAE", "RMSE", "R2"]
                display_cols = [c for c in preferred if c in metrics.columns]
                st.dataframe(
                    metrics[display_cols].round(3),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("Run the training script to generate model metrics.")

    st.markdown("#### Interpretation")
    st.write(
        "Lower MAE and RMSE indicate smaller prediction errors on the "
        "cross-validation folds. R² describes performance relative to the "
        "variation in the target. These metrics should be interpreted cautiously "
        "because the dataset is very small."
    )


# ---------- Prediction ----------
with prediction_tab:
    st.subheader("Prediction Workspace")
    st.markdown(
        '<div class="section-intro">Enter experimental conditions to obtain '
        'a numerical output from the saved regression model.</div>',
        unsafe_allow_html=True
    )

    st.markdown("#### Input Conditions")

    with st.form("prediction_form"):
        c1, c2 = st.columns(2, gap="large")

        with c1:
            preservative = st.selectbox(
                "Preservative formulation",
                ["Formula1", "Formula2"]
            )
            vacuum = st.slider(
                "Vacuum pressure",
                min_value=5, max_value=25, value=15, step=1
            )

        with c2:
            contamination = st.slider(
                "Contamination level",
                min_value=5, max_value=50, value=27, step=1
            )
            temperature = st.slider(
                "Cooling temperature",
                min_value=10, max_value=20, value=15, step=1
            )

        submitted = st.form_submit_button(
            "Generate Prediction",
            use_container_width=True
        )

    if submitted:
        sample = pd.DataFrame([{
            "Preservative": preservative,
            "VacuumPress": float(vacuum),
            "ContaminationLevel": float(contamination),
            "CoolTemp": float(temperature)
        }])

        prediction = float(model.predict(sample)[0])

        st.markdown(
            f'<div class="prediction-output">'
            f'<div class="prediction-label">Predicted numerical Spoilage value</div>'
            f'<div class="prediction-value">{prediction:.3f}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown("#### Submitted Conditions")
        st.dataframe(sample, use_container_width=True, hide_index=True)

        st.caption(
            "This output is a model estimate for research demonstration only. "
            "It must not be used to determine whether food is safe to consume."
        )


# ---------- Model results ----------
with results_tab:
    st.subheader("Experimental Results")
    st.markdown(
        '<div class="section-intro">Explore model evaluation, prediction '
        'diagnostics, target distribution, and feature importance.</div>',
        unsafe_allow_html=True
    )

    if not metrics.empty:
        st.markdown("#### Model Comparison")
        metric_cols = [c for c in ["Model", "MAE", "RMSE", "R2"] if c in metrics.columns]
        st.dataframe(
            metrics[metric_cols].round(4),
            use_container_width=True,
            hide_index=True
        )

        if "MAE" in metrics.columns:
            st.caption(
                "Models are ordered by the evaluation MAE recorded in the latest training run."
            )

    figures = [
        (
            "Model performance",
            "MAE comparison",
            "figure_3_mae_comparison.png",
            "Compares mean absolute error across models."
        ),
        (
            "Model performance",
            "RMSE comparison",
            "figure_4_rmse_comparison.png",
            "Highlights differences in squared-error-sensitive performance."
        ),
        (
            "Prediction diagnostics",
            "Actual versus predicted",
            "figure_2_actual_vs_predicted.png",
            "Compares cross-validation predictions with observed target values."
        ),
        (
            "Prediction diagnostics",
            "Residual analysis",
            "figure_5_residuals.png",
            "Displays prediction errors for diagnostic inspection."
        ),
        (
            "Dataset analysis",
            "Spoilage distribution",
            "figure_1_spoilage_distribution.png",
            "Shows the distribution of the numerical target."
        ),
        (
            "Dataset analysis",
            "Feature importance",
            "figure_6_feature_importance.png",
            "Descriptive feature importance from the original experiment."
        ),
    ]

    for section, entries in [
        ("Model Performance", figures[:2]),
        ("Prediction Diagnostics", figures[2:4]),
        ("Dataset and Feature Analysis", figures[4:6]),
    ]:
        st.markdown(f"#### {section}")
        left, right = st.columns(2, gap="large")

        for col, entry in zip([left, right], entries):
            _, title, filename, description = entry
            with col:
                with st.container(border=True):
                    st.markdown(f"**{title}**")
                    st.caption(description)
                    image_path = RESULTS / filename
                    if image_path.exists():
                        st.image(str(image_path), use_container_width=True)
                    else:
                        st.warning(f"Figure not found: {filename}")

        st.write("")

    st.markdown("#### Detailed Experiment Tables")
    st.caption(
        "Expand a section to inspect its records or download the corresponding CSV."
    )

    tables = [
        ("Dataset Profile", "table_1_dataset_profile.csv"),
        ("Model Configurations", "table_2_model_configurations.csv"),
        ("Original Performance Metrics", "table_3_performance_metrics.csv"),
        ("Cross-Validation Predictions", "table_4_predictions.csv"),
        ("Observation-Level Errors", "table_5_errors.csv"),
        ("Runtime Metrics", "table_6_runtime_metrics.csv"),
    ]

    for title, filename in tables:
        path = RESULTS / filename
        with st.expander(title):
            if path.exists():
                table = pd.read_csv(path)
                st.dataframe(
                    table,
                    use_container_width=True,
                    hide_index=True
                )
                st.download_button(
                    "Download CSV",
                    data=table.to_csv(index=False).encode("utf-8"),
                    file_name=filename,
                    mime="text/csv",
                    key=f"download_{filename}"
                )
            else:
                st.info(f"This table is not available: {filename}")

    with st.expander("Feature Importance Data"):
        importance_path = RESULTS / "feature_importance.csv"
        if importance_path.exists():
            st.dataframe(
                pd.read_csv(importance_path),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Feature importance data is not available.")

    new_predictions = RESULTS / "training_predictions.csv"
    if new_predictions.exists():
        with st.expander("Latest Training Predictions"):
            prediction_table = pd.read_csv(new_predictions)
            st.dataframe(
                prediction_table,
                use_container_width=True,
                hide_index=True
            )
            st.download_button(
                "Download latest predictions",
                data=prediction_table.to_csv(index=False).encode("utf-8"),
                file_name="training_predictions.csv",
                mime="text/csv",
                key="download_latest_predictions"
            )


# ---------- Dataset ----------
with dataset_tab:
    st.subheader("Dataset Explorer")
    st.markdown(
        '<div class="section-intro">Inspect the observations used in the '
        'regression experiments.</div>',
        unsafe_allow_html=True
    )

    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)

        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing Cells", int(df.isna().sum().sum()))

        search = st.text_input(
            "Filter rows",
            placeholder="Search values across the dataset..."
        )

        shown = df.copy()
        if search.strip():
            mask = shown.astype(str).apply(
                lambda column: column.str.contains(
                    search.strip(), case=False, na=False
                )
            ).any(axis=1)
            shown = shown.loc[mask]

        st.dataframe(
            shown,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download dataset CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="food_spoilage.csv",
            mime="text/csv"
        )

        with st.expander("Column Summary"):
            st.dataframe(
                df.describe(include="all").transpose(),
                use_container_width=True
            )
    else:
        st.error("Dataset CSV was not found in the project data directory.")


st.markdown("---")
st.markdown(
    '<div class="footer">Food Spoilage Prediction Research Prototype '
    '| Machine Learning Analytics</div>',
    unsafe_allow_html=True
)
