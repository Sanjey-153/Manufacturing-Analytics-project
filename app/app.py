"""
Streamlit Predictive Maintenance Dashboard for CNC Machines in Automotive Manufacturing.
Interactive Decision-Support Prototype using Industrial Sensor Telemetry and LightGBM.
Premium Modern SaaS Design System (F8 Visual Redesign).
"""
import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import streamlit as st

# ------------------------------------------------------------------------------
# STREAMLIT PAGE CONFIGURATION (Must be first Streamlit command)
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Predictive Maintenance Intelligence | CNC Manufacturing Analytics",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

import plotly.graph_objects as go
import plotly.express as px
import joblib

# Setup absolute paths so dashboard runs seamlessly regardless of CWD
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from config import (
    MODELS_DIR,
    PREPROCESSOR_PATH,
    FIGURES_DIR,
    TABLES_DIR,
    METRICS_DIR,
    RAW_DATA_PATH,
    FEATURE_COLS
)
from src.prediction import (
    load_inference_pipeline,
    predict_single,
    predict_batch,
    SANITY_RANGES,
    VALID_MACHINE_TYPES
)
from src.recommendations import generate_maintenance_recommendation, determine_risk_level

# Import Modular UI Components
from components.header import render_header
from components.sidebar import render_sidebar
from components.cards import (
    render_overview_hero,
    render_overview_kpis,
    render_architecture_diagram,
    render_footer
)
from components.risk import (
    create_risk_gauge,
    render_status_banner,
    render_maintenance_guidance
)
from components.telemetry import (
    render_telemetry_inputs,
    render_telemetry_snapshot
)
from components.charts import (
    create_model_comparison_charts,
    create_feature_importance_chart,
    create_batch_risk_distribution_chart,
    create_global_shap_chart,
    create_shap_contribution_chart
)
from components.live_telemetry import render_live_telemetry_page
from components.drift_page import render_drift_monitoring_page
from components.threshold_page import render_threshold_page
from components.work_orders_page import render_work_orders_page
from components.time_series_page import render_time_series_page
from components.industrial_validation_page import render_industrial_validation_page
from src.explainability import (
    explain_prediction,
    compute_global_shap_summary,
    get_tree_explainer
)
from components.explainability_ui import render_shap_explanation_card

# ------------------------------------------------------------------------------
# LOAD CENTRALIZED STYLESHEET
# ------------------------------------------------------------------------------
css_path = APP_DIR / "styles.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# CACHED DATA & MODEL RESOURCE LOADERS
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing LightGBM inference pipeline...")
def get_cached_pipeline():
    """Loads serialized LightGBM model and ColumnTransformer preprocessor."""
    try:
        model, preprocessor = load_inference_pipeline()
        return model, preprocessor, None
    except Exception as e:
        return None, None, str(e)


@st.cache_data
def get_cached_metrics_data():
    """Loads model benchmarks, deployment summary, and EDA population statistics."""
    data = {}
    try:
        comp_path = METRICS_DIR / "final_model_comparison.csv"
        if comp_path.exists():
            data["comparison_df"] = pd.read_csv(comp_path)

        dep_path = METRICS_DIR / "deployment_summary.json"
        if dep_path.exists():
            with open(dep_path, "r", encoding="utf-8") as f:
                data["deployment_summary"] = json.load(f)

        eda_path = METRICS_DIR / "eda_summary.json"
        if eda_path.exists():
            with open(eda_path, "r", encoding="utf-8") as f:
                data["eda_summary"] = json.load(f)

    except Exception as e:
        data["error"] = str(e)
    return data


model, preprocessor, pipeline_error = get_cached_pipeline()
metrics_data = get_cached_metrics_data()


# ------------------------------------------------------------------------------
# TOP HEADER BAR & NAVIGATION SIDEBAR
# ------------------------------------------------------------------------------
render_header(model_name="LightGBM", system_status="APPLICATION READY")
selected_nav = render_sidebar()


# ==============================================================================
# 01 OVERVIEW PAGE
# ==============================================================================
if selected_nav == "01 Overview":
    # Hero Section with Visual Process Pipeline
    render_overview_hero()

    # Dynamic KPI Cards (Actual Project Benchmark Values)
    render_overview_kpis(metrics_data)

    st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

    # Architectural & Engineering Highlights (Asymmetric Layout)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="content-card">
            <div style="font-size: 1.05rem; font-weight: 700; margin-bottom: 12px; color: #1F1F1F; display: flex; align-items: center; gap: 8px;">
                <span>🎯</span> Operational Objectives & Rigor
            </div>
            <ul style="color: #6B6B63; font-size: 0.86rem; line-height: 1.65; padding-left: 20px;">
                <li><strong style="color: #1F1F1F;">Zero Data Leakage:</strong> Preprocessing ColumnTransformers strictly fitted on training partitions; holdout test set (N=2,000) strictly isolated.</li>
                <li><strong style="color: #1F1F1F;">Recall-Prioritized Optimization:</strong> Model selection tuned specifically to minimize costly False Negatives (unpredicted breakdowns).</li>
                <li><strong style="color: #1F1F1F;">4-Tier Risk Stratification:</strong> Translates raw probabilities into calibrated operational intervention thresholds [Low, Moderate, High, Critical].</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="content-card">
            <div style="font-size: 1.05rem; font-weight: 700; margin-bottom: 12px; color: #1F1F1F; display: flex; align-items: center; gap: 8px;">
                <span>⚡</span> Monitored Failure Mechanisms
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 0.8rem;">
                <div style="background: #F8F9F3; padding: 10px 12px; border-radius: 8px; border: 1px solid #E4E4DD;">
                    <strong style="color: #1F1F1F;">HDF</strong>: Heat Dissipation (33.9%)
                </div>
                <div style="background: #F8F9F3; padding: 10px 12px; border-radius: 8px; border: 1px solid #E4E4DD;">
                    <strong style="color: #1F1F1F;">OSF</strong>: Overstrain Failure (28.9%)
                </div>
                <div style="background: #F8F9F3; padding: 10px 12px; border-radius: 8px; border: 1px solid #E4E4DD;">
                    <strong style="color: #1F1F1F;">PWF</strong>: Power Bounds (28.0%)
                </div>
                <div style="background: #F8F9F3; padding: 10px 12px; border-radius: 8px; border: 1px solid #E4E4DD;">
                    <strong style="color: #1F1F1F;">TWF</strong>: Tool Wear-Out (13.6%)
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# 02 MACHINE PREDICTION PAGE
# ==============================================================================
elif selected_nav == "02 Machine Prediction":
    if pipeline_error:
        st.error(f"Inference pipeline initialization error: {pipeline_error}")
        st.stop()

    col_left, col_right = st.columns([1.1, 1.2], gap="large")

    # LEFT COLUMN: Engineering Input Control Panel
    with col_left:
        payload, threshold, run_predict = render_telemetry_inputs()

    # RIGHT COLUMN: AI Failure Assessment
    with col_right:
        st.markdown("""
        <div class="telemetry-panel">
            <div class="panel-header">
                <span>📊</span> LIVE MACHINE ASSESSMENT
            </div>
        </div>
        """, unsafe_allow_html=True)

        if run_predict:
            status_box = st.empty()
            anim_steps = [
                ("📡 Reading telemetry...", 0.12),
                ("⚙️ Processing parameters...", 0.12),
                ("🧠 Running LightGBM inference...", 0.15),
                ("📊 Calculating failure probability...", 0.12),
                ("🛠️ Generating assessment...", 0.12)
            ]
            for msg, dur in anim_steps:
                status_box.markdown(f"""
                <div style="padding: 10px 14px; background: #F8F9F3; border: 1px solid #E4E4DD; border-radius: 8px; color: #1F1F1F; font-size: 0.82rem; font-weight: 600;">
                    {msg}
                </div>
                """, unsafe_allow_html=True)
                time.sleep(dur)
            status_box.empty()

            try:
                res = predict_single(payload, model=model, preprocessor=preprocessor, threshold=threshold)
                pred_class = res["predicted_class"]
                prob = res["failure_probability"]
                risk = res["risk_level"]

                # 1. Result Status Banner
                render_status_banner(pred_class, risk, prob)

                # 2. Plotly Probability Gauge
                gauge_fig = create_risk_gauge(prob, threshold)
                st.plotly_chart(gauge_fig, use_container_width=True)

                # 3. Echoed Input Telemetry Snapshot
                render_telemetry_snapshot(payload)

                # 4. Maintenance Guidance Card
                render_maintenance_guidance(res)

                # 5. SHAP Per-Prediction Attribution (F2)
                try:
                    shap_res = explain_prediction(payload, model=model, preprocessor=preprocessor, threshold=threshold)
                    render_shap_explanation_card(shap_res, title="MODEL EXPLANATION (SHAP ATTRIBUTION)", chart_key="machine_pred_shap_chart")
                except Exception as ex:
                    st.warning(f"SHAP explanation unavailable: {ex}")

            except Exception as e:
                st.error(f"Inference calculation error: {e}")

        else:
            st.markdown("""
            <div class="empty-state-card">
                <div class="empty-state-icon">📡</div>
                <div class="empty-state-title">READY FOR ANALYSIS</div>
                <div class="empty-state-desc">
                    Enter machine telemetry parameters on the left panel and click <strong>RUN FAILURE ANALYSIS</strong> to compute real-time failure probability and diagnostic guidance.
                </div>
            </div>
            """, unsafe_allow_html=True)


# ==============================================================================
# 03 BATCH TELEMETRY PAGE
# ==============================================================================
elif selected_nav == "03 Batch Telemetry":
    st.markdown("""
    <div class="content-card">
        <div style="font-size: 1.25rem; font-weight: 700; color: #1F1F1F; margin-bottom: 4px;">
            📂 Batch Telemetry Analysis & Fleet Diagnostics
        </div>
        <div style="font-size: 0.85rem; color: #6B6B63;">
            Upload multi-record CSV telemetry files or evaluate benchmark test batches from the AI4I 2020 dataset.
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "DRAG & DROP CSV OR BROWSE FILE",
        type=["csv"],
        help="Required columns: Type, Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min]"
    )

    batch_df = None
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Uploaded CSV with {len(batch_df):,} records.")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
    else:
        st.markdown("<div style='font-size: 0.8rem; color: #6B6B63; margin-bottom: 8px;'>Or load benchmark test instances:</div>", unsafe_allow_html=True)
        if st.button("LOAD SAMPLE TEST BATCH (20 INSTANCES)"):
            raw_path = RAW_DATA_PATH
            if raw_path.exists():
                raw_df = pd.read_csv(raw_path)
                batch_df = raw_df[FEATURE_COLS].iloc[25:45].copy().reset_index(drop=True)

    if batch_df is not None:
        try:
            with st.spinner("Processing batch inference through LightGBM..."):
                results = predict_batch(batch_df, model=model, preprocessor=preprocessor)

            total_recs = len(results)
            failures = (results["Predicted_Class"] == 1).sum()
            healthy = total_recs - failures
            max_prob = results["Failure_Probability"].max()

            # Batch Summary Metrics
            b1, b2, b3, b4 = st.columns(4)
            with b1:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">TOTAL RECORDS</div>
                    <div class="kpi-value">{total_recs:,}</div>
                    <div class="kpi-subtitle">Processed Batch</div>
                </div>
                """, unsafe_allow_html=True)
            with b2:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">PREDICTED FAILURES</div>
                    <div class="kpi-value kpi-critical">{failures:,}</div>
                    <div class="kpi-subtitle">Class 1 Alerts</div>
                </div>
                """, unsafe_allow_html=True)
            with b3:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">HEALTHY UNITS</div>
                    <div class="kpi-value kpi-healthy">{healthy:,}</div>
                    <div class="kpi-subtitle">Nominal Operation</div>
                </div>
                """, unsafe_allow_html=True)
            with b4:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">HIGHEST RISK</div>
                    <div class="kpi-value kpi-warning">{max_prob*100:.1f}%</div>
                    <div class="kpi-subtitle">Peak Anomaly Score</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Batch Risk Distribution Visualization & Results Table
            t_col1, t_col2 = st.columns([1, 1.8], gap="medium")
            with t_col1:
                donut_fig = create_batch_risk_distribution_chart(results)
                st.plotly_chart(donut_fig, use_container_width=True)

            with t_col2:
                st.markdown("### Batch Results")
                display_cols = ["Type", "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]", "Status", "Failure_Probability", "Risk_Level"]
                st.dataframe(results[display_cols], use_container_width=True, height=280)

            csv_out = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 DOWNLOAD PREDICTIONS CSV",
                data=csv_out,
                file_name="predictive_maintenance_batch_predictions.csv",
                mime="text/csv",
                type="primary"
            )

            # On-Demand SHAP Attribution for Selected Batch Instance (F2)
            st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
            with st.expander("🔍 EXPLAIN HIGH-RISK BATCH INSTANCE (SHAP LOCAL ATTRIBUTION)", expanded=False):
                st.markdown("<div style='font-size: 0.8rem; color: #6B6B63; margin-bottom: 10px;'>Select any instance or evaluate the peak failure risk instance from this uploaded batch:</div>", unsafe_allow_html=True)
                
                highest_idx = int(results["Failure_Probability"].idxmax()) if not results.empty else 0
                selected_row_idx = st.number_input(
                    "Select Row Index to Explain",
                    min_value=0,
                    max_value=len(results) - 1,
                    value=highest_idx,
                    step=1
                )
                
                if st.button("CALCULATE SHAP EXPLANATION FOR SELECTED ROW", key="btn_explain_batch_row"):
                    row_data = results.iloc[selected_row_idx].to_dict()
                    try:
                        shap_batch_res = explain_prediction(row_data, model=model, preprocessor=preprocessor)
                        render_shap_explanation_card(
                            shap_batch_res,
                            title=f"BATCH ROW #{selected_row_idx} ATTRIBUTION (Risk: {shap_batch_res['risk_tier']})",
                            chart_key=f"batch_shap_chart_{selected_row_idx}"
                        )
                    except Exception as ex:
                        st.error(f"SHAP explanation failed: {ex}")

        except Exception as e:
            st.error(f"Batch inference failed: {e}")


# ==============================================================================
# 04 MANUFACTURING ANALYTICS (EDA)
# ==============================================================================
elif selected_nav == "04 Manufacturing Analytics":
    st.markdown("""
    <div class="content-card">
        <div style="font-size: 1.25rem; font-weight: 700; color: #1F1F1F; margin-bottom: 4px;">
            📈 Manufacturing Analytics & Sensor Dynamics
        </div>
        <div style="font-size: 0.85rem; color: #6B6B63;">
            Empirical population distributions, correlation analysis, failure modes, and operating regimes from the AI4I 2020 CNC dataset (N = 10,000 observations).
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Failure & Machine Types",
        "Sensor Distributions & Correlations",
        "Failure Modes Breakdown",
        "Operating Regimes & Regimes"
    ])

    with tab1:
        st.subheader("Target Imbalance & Product Quality Variants")
        c1, c2 = st.columns(2)
        with c1:
            fig1 = FIGURES_DIR / "01_machine_failure_distribution.png"
            if fig1.exists():
                st.image(str(fig1), caption="Target Imbalance: 96.61% Healthy vs 3.39% Failures (Severe Imbalance)", use_container_width=True)
        with c2:
            fig2 = FIGURES_DIR / "02_machine_type_distribution.png"
            if fig2.exists():
                st.image(str(fig2), caption="Product Quality Variants: L (50%), M (30%), H (20%)", use_container_width=True)

        fig7 = FIGURES_DIR / "07_machine_type_vs_failure.png"
        if fig7.exists():
            st.image(str(fig7), caption="Failure Rates by Quality Variant (L: 3.92%, M: 2.77%, H: 2.09%)", use_container_width=True)

    with tab2:
        st.subheader("Sensor Distributions & Correlation Matrix")
        fig5 = FIGURES_DIR / "05_sensor_distributions.png"
        if fig5.exists():
            st.image(str(fig5), caption="Univariate Sensor Parameter Distributions across 10,000 Observations", use_container_width=True)

        fig3 = FIGURES_DIR / "03_sensor_correlation_heatmap.png"
        if fig3.exists():
            st.image(str(fig3), caption="Pearson Correlation Heatmap (Note Speed-Torque Inverse Coupling r = -0.875)", use_container_width=True)

    with tab3:
        st.subheader("Specific Failure Modes Breakdown")
        fig4 = FIGURES_DIR / "04_failure_mode_distribution.png"
        if fig4.exists():
            st.image(str(fig4), caption="Failure Modes Distribution (HDF, OSF, PWF, TWF, RNF)", use_container_width=True)

        st.markdown("""
        <div style="background: #F8F9F3; padding: 16px 20px; border: 1px solid #E4E4DD; border-radius: 10px; font-size: 0.84rem; line-height: 1.65;">
            <strong style="color: #1F1F1F;">Physical Failure Mechanism Definitions:</strong>
            <ul style="color: #6B6B63; margin-top: 8px;">
                <li><strong style="color: #1F1F1F;">Heat Dissipation Failure (HDF):</strong> 115 incidents (33.92%) — Triggered when ΔT (Process Temp − Air Temp) < 8.6 K and Rotational Speed < 1380 rpm.</li>
                <li><strong style="color: #1F1F1F;">Overstrain Failure (OSF):</strong> 98 incidents (28.91%) — Triggered when tool wear × torque exceeds variant mechanical stress thresholds.</li>
                <li><strong style="color: #1F1F1F;">Power Failure (PWF):</strong> 95 incidents (28.02%) — Occurs when spindle cutting power (P = τ · ω) falls outside [3,500 W, 9,000 W].</li>
                <li><strong style="color: #1F1F1F;">Tool Wear Failure (TWF):</strong> 46 incidents (13.57%) — Continuous cutter wear exceeds wearout limit (200–240 min).</li>
                <li><strong style="color: #1F1F1F;">Random Failure (RNF):</strong> 19 incidents (5.60%) — Random process anomaly uncorrelated with regular telemetry.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with tab4:
        st.subheader("Operating Regimes & Multi-Variable Failure Boundaries")
        fig6 = FIGURES_DIR / "06_features_by_failure_boxplot.png"
        if fig6.exists():
            st.image(str(fig6), caption="Sensor Distributions Stratified by Machine Failure Status", use_container_width=True)

        fig8 = FIGURES_DIR / "08_feature_interactions_vs_failure.png"
        if fig8.exists():
            st.image(str(fig8), caption="Operating Regimes & Multi-Variable Failure Boundaries", use_container_width=True)


# ==============================================================================
# 05 MODEL PERFORMANCE & COMPARISON
# ==============================================================================
elif selected_nav == "05 Model Performance":
    st.markdown("""
    <div class="content-card">
        <div style="font-size: 1.25rem; font-weight: 700; color: #1F1F1F; margin-bottom: 4px;">
            🏆 Model Performance Benchmark & Selection
        </div>
        <div style="font-size: 0.85rem; color: #6B6B63;">
            Empirical holdout test evaluation (N = 2,000 observations, 68 failure cases) comparing Decision Tree, Random Forest, XGBoost, LightGBM, and Stacking Ensemble.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if "comparison_df" in metrics_data:
        comp_df = metrics_data["comparison_df"]

        # Primary Model Champion Banner
        st.markdown("""
        <div style="background: #EBF3ED; border: 1px solid #BCD5C3; border-radius: 10px; padding: 14px 20px; margin-bottom: 20px;">
            <span style="color: #477A55; font-weight: 700; font-size: 0.95rem;">PRIMARY CHAMPION: LightGBM</span>
            <span style="color: #6B6B63; font-size: 0.82rem; margin-left: 12px;">Selected for highest fault recall (80.88%), lowest missed breakdowns (13 FN), and optimal F1-score (0.6433).</span>
        </div>
        """, unsafe_allow_html=True)

        # Plotly Comparison Charts
        fig_rec, fig_fn, fig_f1 = create_model_comparison_charts(comp_df)

        ch1, ch2 = st.columns(2)
        with ch1:
            st.plotly_chart(fig_rec, use_container_width=True)
        with ch2:
            st.plotly_chart(fig_fn, use_container_width=True)

        # Comparison Table
        st.markdown("### Empirical Benchmark Comparison Table")
        st.dataframe(comp_df, use_container_width=True)

        # Stacking Evaluation Note
        st.markdown("""
        <div class="content-card" style="margin-top: 14px;">
            <div style="font-size: 0.92rem; font-weight: 700; color: #1F1F1F; margin-bottom: 8px;">
                ℹ️ Stacking Ensemble Architecture Evaluation
            </div>
            <div style="font-size: 0.84rem; color: #6B6B63; line-height: 1.55;">
                A 5-Fold Cross-Validation Out-of-Fold Stacking Classifier (combining Decision Tree, Random Forest, XGBoost, and LightGBM with a Logistic Regression meta-learner) was rigorously trained and evaluated. On the holdout test set, the Stacking Ensemble achieved <strong>76.47% recall (16 False Negatives)</strong>, while the single tuned <strong>LightGBM achieved 80.88% recall (13 False Negatives)</strong>. Because predictive maintenance strictly prioritizes minimizing unpredicted breakdowns (False Negatives), LightGBM was validated and deployed as the Primary Model.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Confusion Matrices Section
        st.markdown("### Confusion Matrices (Holdout Test Set N=2,000)")
        cm_cols = st.columns(4)
        cm_files = [
            ("Decision Tree", "confusion_matrix_decision_tree.png"),
            ("Random Forest", "confusion_matrix_random_forest.png"),
            ("XGBoost", "confusion_matrix_xgboost.png"),
            ("LightGBM (Champion)", "confusion_matrix_lightgbm.png")
        ]
        for col, (m_name, f_name) in zip(cm_cols, cm_files):
            with col:
                f_path = FIGURES_DIR / f_name
                if f_path.exists():
                    st.image(str(f_path), caption=m_name, use_container_width=True)

        fig_stack = FIGURES_DIR / "confusion_matrix_stacking_ensemble.png"
        if fig_stack.exists():
            st.image(str(fig_stack), caption="Stacking Ensemble Confusion Matrix", width=420)

    else:
        st.warning("Model comparison metrics file not found.")


# ==============================================================================
# 06 EXPLAINABILITY & FEATURE IMPORTANCE (SHAP XAI - F2)
# ==============================================================================
elif selected_nav == "06 Explainability":
    st.markdown("""
    <div class="content-card">
        <div style="font-size: 1.25rem; font-weight: 700; color: #1F1F1F; margin-bottom: 4px;">
            🔍 Explainable AI (XAI) & SHAP Feature Attribution
        </div>
        <div style="font-size: 0.85rem; color: #6B6B63;">
            Rigorous per-prediction and population-level explainability for the LightGBM classifier using SHapley Additive exPlanations (SHAP).
        </div>
        <div class="disclaimer-box" style="margin-top: 10px; margin-bottom: 0px; font-size: 0.78rem;">
            <strong>SHAP Principle:</strong> SHAP estimates how individual input features contribute to a model prediction. Positive contributions push the model output toward higher predicted failure risk, while negative contributions push it toward lower risk.
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_local, tab_global, tab_theory = st.tabs([
        "🔬 Interactive Per-Prediction Attribution (Local SHAP)",
        "🌐 Global Model Explainability (Population SHAP vs Split Gain)",
        "📚 XAI Methodology & Research Rigor"
    ])

    with tab_local:
        st.markdown("### Interactive Local Feature Attribution Explorer")
        st.markdown("<div style='font-size: 0.82rem; color: #6B6B63; margin-bottom: 12px;'>Select a benchmark operating scenario or customize parameters to inspect the exact directional feature attributions:</div>", unsafe_allow_html=True)

        scen_col, param_col = st.columns([1, 2], gap="medium")

        preset_scenarios = {
            "Nominal Healthy Machine (Type M)": {
                "Type": "M", "Air temperature [K]": 298.1, "Process temperature [K]": 308.6,
                "Rotational speed [rpm]": 1551.0, "Torque [Nm]": 42.8, "Tool wear [min]": 0.0
            },
            "Severe Overstrain Load (Type L, High Torque & Tool Wear)": {
                "Type": "L", "Air temperature [K]": 302.5, "Process temperature [K]": 310.8,
                "Rotational speed [rpm]": 1320.0, "Torque [Nm]": 68.4, "Tool wear [min]": 215.0
            },
            "Heat Dissipation Breakdown (Low ΔT, Low Spindle Speed)": {
                "Type": "L", "Air temperature [K]": 304.0, "Process temperature [K]": 309.5,
                "Rotational speed [rpm]": 1280.0, "Torque [Nm]": 55.0, "Tool wear [min]": 80.0
            },
            "Cutter Wear-Out Limit (Tool Wear > 230 min)": {
                "Type": "H", "Air temperature [K]": 299.0, "Process temperature [K]": 309.0,
                "Rotational speed [rpm]": 1500.0, "Torque [Nm]": 45.0, "Tool wear [min]": 240.0
            }
        }

        with scen_col:
            selected_preset_name = st.selectbox(
                "PRESET BENCHMARK SCENARIO",
                list(preset_scenarios.keys()),
                index=0
            )
            preset_vals = preset_scenarios[selected_preset_name]

            st.markdown("<div style='margin-top: 10px; font-size: 0.76rem; color: #6B6B63; font-weight: 600;'>FINE-TUNE SENSOR INPUTS:</div>", unsafe_allow_html=True)
            in_type = st.selectbox("Machine Type", ["L", "M", "H"], index=["L", "M", "H"].index(preset_vals["Type"]))
            in_air = st.slider("Air Temperature [K]", 295.0, 305.0, float(preset_vals["Air temperature [K]"]), step=0.1)
            in_proc = st.slider("Process Temperature [K]", 305.0, 315.0, float(preset_vals["Process temperature [K]"]), step=0.1)
            in_speed = st.slider("Rotational Speed [rpm]", 1100, 2900, int(preset_vals["Rotational speed [rpm]"]), step=10)
            in_torque = st.slider("Torque [Nm]", 3.0, 80.0, float(preset_vals["Torque [Nm]"]), step=0.5)
            in_wear = st.slider("Tool Wear [min]", 0, 260, int(preset_vals["Tool wear [min]"]), step=1)

            active_payload = {
                "Type": in_type,
                "Air temperature [K]": in_air,
                "Process temperature [K]": in_proc,
                "Rotational speed [rpm]": float(in_speed),
                "Torque [Nm]": in_torque,
                "Tool wear [min]": float(in_wear)
            }

        with param_col:
            try:
                shap_local = explain_prediction(active_payload, model=model, preprocessor=preprocessor, threshold=0.50)
                
                pred_badge = "badge-risk-crit" if shap_local["predicted_class"] == 1 else "badge-risk-low"
                st.markdown(f"""
                <div style="background: #FFFFFF; border: 1px solid #E4E4DD; border-radius: 10px; padding: 12px 16px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                    <div>
                        <span style="color: #6B6B63; font-size: 0.8rem;">PREDICTED STATUS:</span> 
                        <strong style="color: #1F1F1F; margin-left: 6px;">{shap_local['status_label']}</strong>
                    </div>
                    <div>
                        <span style="color: #6B6B63; font-size: 0.8rem;">FAILURE PROBABILITY:</span> 
                        <strong style="color: #1F1F1F; font-family: 'JetBrains Mono', monospace; margin-left: 6px;">{shap_local['failure_probability_percentage']}</strong>
                    </div>
                    <div>
                        <span class="{pred_badge}">{shap_local['risk_tier'].upper()}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                render_shap_explanation_card(
                    shap_local,
                    title=f"LOCAL ATTRIBUTION: {selected_preset_name.split(' (')[0]}",
                    show_table=True,
                    chart_key="scenario_shap_chart"
                )
            except Exception as e:
                st.error(f"Failed to generate SHAP attribution: {e}")

    with tab_global:
        st.markdown("### Population-Level Global Model Explainability")
        st.markdown("<div style='font-size: 0.82rem; color: #6B6B63; margin-bottom: 16px;'>Comparison between Global Mean Absolute SHAP Values (population attribution) and LightGBM Split Gain Feature Importance.</div>", unsafe_allow_html=True)

        g_col1, g_col2 = st.columns(2, gap="medium")

        with g_col1:
            st.markdown("<div style='font-size: 0.92rem; font-weight: 700; color: #1F1F1F; margin-bottom: 8px;'>1. Global SHAP Feature Importance (Mean |SHAP|)</div>", unsafe_allow_html=True)
            try:
                with st.spinner("Computing global population SHAP summary..."):
                    global_shap_df = compute_global_shap_summary(model=model, preprocessor=preprocessor, n_samples=100)
                    g_fig = create_global_shap_chart(global_shap_df)
                    st.plotly_chart(g_fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Global SHAP calculation unavailable: {e}")

            st.markdown("""
            <div style="font-size: 0.8rem; color: #6B6B63; background: #F8F9F3; padding: 10px 14px; border-radius: 8px; border: 1px solid #E4E4DD;">
                <strong>Interpretation:</strong> Mean |SHAP| averages the absolute impact of each feature across all sampled operating cycles.
            </div>
            """, unsafe_allow_html=True)

        with g_col2:
            st.markdown("<div style='font-size: 0.92rem; font-weight: 700; color: #1F1F1F; margin-bottom: 8px;'>2. LightGBM Tree Split Gain Importance</div>", unsafe_allow_html=True)
            fi_fig = create_feature_importance_chart()
            st.plotly_chart(fi_fig, use_container_width=True)

            st.markdown("""
            <div style="font-size: 0.8rem; color: #6B6B63; background: #F8F9F3; padding: 10px 14px; border-radius: 8px; border: 1px solid #E4E4DD;">
                <strong>Interpretation:</strong> Split Gain measures cumulative reduction in training loss contributed by each feature across decision tree splits.
            </div>
            """, unsafe_allow_html=True)

    with tab_theory:
        st.markdown("### Explainable AI Principles & Academic Rigor")
        
        t_c1, t_c2 = st.columns(2, gap="medium")
        with t_c1:
            st.markdown("""
            <div class="content-card">
                <div style="font-size: 0.95rem; font-weight: 700; color: #1F1F1F; margin-bottom: 10px;">
                    📐 Theoretical Foundation of SHAP
                </div>
                <div style="font-size: 0.84rem; color: #6B6B63; line-height: 1.65;">
                    SHAP (SHapley Additive exPlanations) is based on cooperative game theory (Shapley values). For any prediction, the total model output margin is decomposed additively:
                    <br><br>
                    <code style="color: #1F1F1F; background: #F8F9F3; padding: 4px 8px; border-radius: 4px; display: block; text-align: center; margin-bottom: 10px; border: 1px solid #E4E4DD;">
                        f(x) = φ₀ + ∑ φᵢ
                    </code>
                    where <strong>φ₀</strong> is the expected baseline model margin, and <strong>φᵢ</strong> represents the exact attribution of feature <em>i</em>.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with t_c2:
            st.markdown("""
            <div class="content-card">
                <div style="font-size: 0.95rem; font-weight: 700; color: #1F1F1F; margin-bottom: 10px;">
                    ⚖️ Model Attribution vs Physical Causality
                </div>
                <div style="font-size: 0.84rem; color: #6B6B63; line-height: 1.65;">
                    <strong style="color: #1F1F1F;">Crucial Academic Distinction:</strong>
                    <ul style="padding-left: 18px; margin-top: 6px;">
                        <li><strong style="color: #1F1F1F;">Model Attribution:</strong> SHAP values explain the internal mathematical reasoning of the trained LightGBM model for a given data point.</li>
                        <li><strong style="color: #1F1F1F;">Not Physical Causality:</strong> A positive SHAP value indicates that a feature pushed the statistical prediction toward failure; it does <em>not</em> prove physical root-cause failure mechanics.</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ==============================================================================
# 07 SYSTEM ARCHITECTURE & INTEGRITY
# ==============================================================================
elif selected_nav == "07 System Architecture":
    st.markdown("""
    <div class="content-card">
        <div style="font-size: 1.25rem; font-weight: 700; color: #1F1F1F; margin-bottom: 4px;">
            ℹ️ System Architecture & Verification
        </div>
        <div style="font-size: 0.85rem; color: #6B6B63;">
            Engineering specifications, data pipeline verification, and academic project provenance.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Full Architecture Flowchart
    render_architecture_diagram()

    st.markdown("""
    <div class="content-card">
        <div style="font-size: 1rem; font-weight: 700; color: #1F1F1F; margin-bottom: 12px;">
            📋 System Specifications & Operational Integrity
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; font-size: 0.82rem;">
            <div style="background: #F8F9F3; padding: 12px 16px; border: 1px solid #E4E4DD; border-radius: 8px;">
                <span style="color: #1F1F1F; font-weight: 700;">Data Ingestion:</span><br>
                <span style="color: #6B6B63;">AI4I 2020 Predictive Maintenance Dataset (10,000 CNC milling cycles).</span>
            </div>
            <div style="background: #F8F9F3; padding: 12px 16px; border: 1px solid #E4E4DD; border-radius: 8px;">
                <span style="color: #1F1F1F; font-weight: 700;">Leakage Prevention:</span><br>
                <span style="color: #6B6B63;">Identifiers and failure flags strictly excluded from all feature matrices.</span>
            </div>
            <div style="background: #F8F9F3; padding: 12px 16px; border: 1px solid #E4E4DD; border-radius: 8px;">
                <span style="color: #1F1F1F; font-weight: 700;">SMOTE Isolation:</span><br>
                <span style="color: #6B6B63;">Resampling applied exclusively to training split; holdout test split completely untouched.</span>
            </div>
            <div style="background: #F8F9F3; padding: 12px 16px; border: 1px solid #E4E4DD; border-radius: 8px;">
                <span style="color: #1F1F1F; font-weight: 700;">Inference Latency:</span><br>
                <span style="color: #6B6B63;">< 2 ms per instance via pure Python Scikit-Learn / LightGBM pipeline.</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### Project Metadata & Academic Provenance
    - **Project Title:** AI-Based Predictive Maintenance System for CNC Machines in Automotive Manufacturing Using Industrial Sensor Data
    - **Academic Department:** Department of Artificial Intelligence and Data Science, Rajalakshmi Engineering College
    - **Authors:** Pragadeshwaran (231801126), Sanjey Ram Kumar (231801153), Santhosh G (231801154)
    - **Domain:** Industrial Manufacturing Analytics & Machine Health Prognostics
    - **Application Scope:** Decision-support analytical dashboard. Does not actuate physical CNC machinery.
    """)


# ==============================================================================
# 08 LIVE TELEMETRY (SIMULATED NEAR-REAL-TIME PROGNOSTICS - F1)
# ==============================================================================
elif selected_nav == "08 Live Telemetry":
    render_live_telemetry_page(model=model, preprocessor=preprocessor)


# ==============================================================================
# 09 DRIFT MONITORING (DATA & MODEL DISTRIBUTION MONITOR - F3)
# ==============================================================================
elif selected_nav == "09 Drift Monitoring":
    render_drift_monitoring_page(model=model, preprocessor=preprocessor)


# ==============================================================================
# 10 THRESHOLD & COST ANALYSIS (DECISION SENSITIVITY FRAMEWORK - F4)
# ==============================================================================
elif selected_nav == "10 Threshold & Cost Analysis":
    render_threshold_page(model=model, preprocessor=preprocessor)


# ==============================================================================
# 11 MAINTENANCE WORK ORDERS (DECISION SUPPORT WORKFLOW - F5)
# ==============================================================================
elif selected_nav == "11 Maintenance Work Orders":
    render_work_orders_page()


# ==============================================================================
# 12 TIME-SERIES ARCHITECTURE (SEQUENTIAL TELEMETRY READINESS - F6)
# ==============================================================================
elif selected_nav == "12 Time-Series Architecture":
    render_time_series_page()


# ==============================================================================
# 13 INDUSTRIAL VALIDATION (REAL-WORLD VALIDATION READINESS - F7)
# ==============================================================================
elif selected_nav == "13 Industrial Validation":
    render_industrial_validation_page(model=model, preprocessor=preprocessor)


# ------------------------------------------------------------------------------
# GLOBAL FOOTER
# ------------------------------------------------------------------------------
render_footer()
