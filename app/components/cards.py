"""
Reusable Metric Cards, Hero Sections, Process Flows, and Architecture Visuals.
"""
import streamlit as st


def render_overview_hero():
    """Renders the executive landing page hero section with end-to-end technical process visual."""
    st.markdown("""
    <div class="hero-container">
        <div class="hero-badge">AI4I 2020 CNC Benchmark</div>
        <div class="hero-title">AI-Based Predictive Maintenance</div>
        <div class="hero-subtext">
            Industrial machine operating parameter intelligence. Transforming multi-sensor telemetry into early failure warnings, risk-stratified probabilities, and actionable maintenance protocols.
        </div>
        <div class="process-flow-box">
            <div class="process-flow-title">
                End-to-End Decision Support Pipeline
            </div>
            <div class="process-steps">
                <div class="process-node">
                    <div class="process-node-icon">📡</div>
                    <div class="process-node-title">Machine Telemetry</div>
                    <div class="process-node-desc">Temp, RPM, Torque, Wear</div>
                </div>
                <div class="process-arrow">➔</div>
                <div class="process-node">
                    <div class="process-node-icon">⚙️</div>
                    <div class="process-node-title">Data Processing</div>
                    <div class="process-node-desc">OHE & Leakage Isolation</div>
                </div>
                <div class="process-arrow">➔</div>
                <div class="process-node">
                    <div class="process-node-icon">🧠</div>
                    <div class="process-node-title">Machine Learning</div>
                    <div class="process-node-desc">LightGBM Classifier</div>
                </div>
                <div class="process-arrow">➔</div>
                <div class="process-node">
                    <div class="process-node-icon">📊</div>
                    <div class="process-node-title">Failure Probability</div>
                    <div class="process-node-desc">Posterior Risk Gauge</div>
                </div>
                <div class="process-arrow">➔</div>
                <div class="process-node">
                    <div class="process-node-icon">🎯</div>
                    <div class="process-node-title">Risk Assessment</div>
                    <div class="process-node-desc">4 Operational Bands</div>
                </div>
                <div class="process-arrow">➔</div>
                <div class="process-node">
                    <div class="process-node-icon">🛠️</div>
                    <div class="process-node-title">Maintenance Guidance</div>
                    <div class="process-node-desc">Prescriptive Actions</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_overview_kpis(metrics_data: dict):
    """Renders the validated project benchmark KPI cards with actual values."""
    dep_summary = metrics_data.get("deployment_summary", {})
    lgb_metrics = dep_summary.get("primary_model_evaluation", {})

    obs_val = "10,000"
    prev_val = "3.39%"
    recall_val = f"{lgb_metrics.get('recall', 0.8088) * 100:.2f}%"
    f1_val = f"{lgb_metrics.get('f1_score', 0.6433) * 100:.2f}%"
    fn_val = f"{lgb_metrics.get('confusion_matrix', {}).get('false_negatives', 13)}"
    acc_val = f"{lgb_metrics.get('accuracy', 0.9695) * 100:.2f}%"

    st.markdown("""
    <div style="font-size: 0.74rem; font-weight: 700; color: #6B6B63; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 12px;">
        BENCHMARK MODEL & DATASET METRICS (HOLDOUT EVALUATION N = 2,000)
    </div>
    """, unsafe_allow_html=True)

    k1, k2, k3, k4, k5, k6 = st.columns(6)

    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">DATASET RECORDS</div>
            <div class="kpi-value">{obs_val}</div>
            <div class="kpi-subtitle">CNC Milling Cycles</div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">FAILURE PREVALENCE</div>
            <div class="kpi-value kpi-watch">{prev_val}</div>
            <div class="kpi-subtitle">Severe Imbalance</div>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">LIGHTGBM RECALL</div>
            <div class="kpi-value kpi-healthy">{recall_val}</div>
            <div class="kpi-subtitle">Primary Champion</div>
        </div>
        """, unsafe_allow_html=True)

    with k4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">LIGHTGBM F1</div>
            <div class="kpi-value">{f1_val}</div>
            <div class="kpi-subtitle">Harmonic Mean</div>
        </div>
        """, unsafe_allow_html=True)

    with k5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">FALSE NEGATIVES</div>
            <div class="kpi-value kpi-healthy">{fn_val} <span style="font-size: 0.85rem; color: #6B6B63;">/ 68</span></div>
            <div class="kpi-subtitle">Lowest Missed Breakdowns</div>
        </div>
        """, unsafe_allow_html=True)

    with k6:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">ACCURACY</div>
            <div class="kpi-value">{acc_val}</div>
            <div class="kpi-subtitle">Test Verification</div>
        </div>
        """, unsafe_allow_html=True)


def render_architecture_diagram():
    """Renders the comprehensive system architecture flow diagram distinguishing training and inference."""
    st.markdown("""
<div class="content-card">
    <div style="font-size: 1.1rem; font-weight: 700; color: #1F1F1F; margin-bottom: 6px; display: flex; align-items: center; gap: 8px;">
        <span>📐</span> Complete End-to-End System Pipeline Architecture
    </div>
    <div style="font-size: 0.84rem; color: #6B6B63; margin-bottom: 18px;">
        Visualizing the end-to-end data transformation from raw telemetry ingestion to calibrated real-time inference.
    </div>
    <div style="background: #F8F9F3; border: 1px solid #E4E4DD; border-radius: 10px; padding: 20px;">
        <div style="margin-bottom: 18px;">
            <div style="font-size: 0.72rem; font-weight: 700; color: #1F1F1F; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 10px;">
                1. OFFLINE TRAINING &amp; VALIDATION PIPELINE (LEAKAGE-SAFE)
            </div>
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px; font-size: 0.76rem;">
                <div style="background: #FFFFFF; border: 1px solid #E4E4DD; border-radius: 8px; padding: 8px 12px; text-align: center;">
                    <span style="color: #1F1F1F; font-weight: 700;">AI4I Dataset</span><br><span style="color: #6B6B63;">10,000 records</span>
                </div>
                <div style="color: #1F1F1F; font-weight: bold;">➔</div>
                <div style="background: #FFFFFF; border: 1px solid #E4E4DD; border-radius: 8px; padding: 8px 12px; text-align: center;">
                    <span style="color: #1F1F1F; font-weight: 700;">Feature Selection</span><br><span style="color: #6B6B63;">Exclude IDs &amp; Targets</span>
                </div>
                <div style="color: #1F1F1F; font-weight: bold;">➔</div>
                <div style="background: #FFFFFF; border: 1px solid #E4E4DD; border-radius: 8px; padding: 8px 12px; text-align: center;">
                    <span style="color: #1F1F1F; font-weight: 700;">Train/Test Split</span><br><span style="color: #6B6B63;">80/20 Stratified</span>
                </div>
                <div style="color: #1F1F1F; font-weight: bold;">➔</div>
                <div style="background: #FFFFFF; border: 1px solid #E4E4DD; border-radius: 8px; padding: 8px 12px; text-align: center;">
                    <span style="color: #1F1F1F; font-weight: 700;">SMOTE Training Data</span><br><span style="color: #6B6B63;">Train partition only</span>
                </div>
                <div style="color: #1F1F1F; font-weight: bold;">➔</div>
                <div style="background: #FFFFFF; border: 1px solid #E4E4DD; border-radius: 8px; padding: 8px 12px; text-align: center;">
                    <span style="color: #1F1F1F; font-weight: 700;">Model Benchmarking</span><br><span style="color: #6B6B63;">DT, RF, XGB, LGBM</span>
                </div>
                <div style="color: #1F1F1F; font-weight: bold;">➔</div>
                <div style="background: #FFFFFF; border: 1px solid #E4E4DD; border-radius: 8px; padding: 8px 12px; text-align: center;">
                    <span style="color: #1F1F1F; font-weight: 700;">Stacking Ensemble</span><br><span style="color: #6B6B63;">5-Fold OOF Meta-Learner</span>
                </div>
                <div style="color: #1F1F1F; font-weight: bold;">➔</div>
                <div style="background: #EBF3ED; border: 1px solid #BCD5C3; border-radius: 8px; padding: 8px 12px; text-align: center;">
                    <span style="color: #477A55; font-weight: 700;">Primary Champion</span><br><span style="color: #477A55;">LightGBM Model</span>
                </div>
            </div>
        </div>
        <div>
            <div style="font-size: 0.72rem; font-weight: 700; color: #477A55; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 10px;">
                2. ONLINE INFERENCE &amp; DECISION SUPPORT PIPELINE (&lt; 2 ms LATENCY)
            </div>
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px; font-size: 0.76rem;">
                <div style="background: #FFFFFF; border: 1px solid #E4E4DD; border-radius: 8px; padding: 8px 12px; text-align: center;">
                    <span style="color: #1F1F1F; font-weight: 700;">Telemetry Input</span><br><span style="color: #6B6B63;">6 Raw Sensor Feeds</span>
                </div>
                <div style="color: #477A55; font-weight: bold;">➔</div>
                <div style="background: #FFFFFF; border: 1px solid #E4E4DD; border-radius: 8px; padding: 8px 12px; text-align: center;">
                    <span style="color: #1F1F1F; font-weight: 700;">Prediction Engine</span><br><span style="color: #6B6B63;">ColumnTransformer</span>
                </div>
                <div style="color: #477A55; font-weight: bold;">➔</div>
                <div style="background: #FFFFFF; border: 1px solid #E4E4DD; border-radius: 8px; padding: 8px 12px; text-align: center;">
                    <span style="color: #1F1F1F; font-weight: 700;">LightGBM Inference</span><br><span style="color: #6B6B63;">Posterior Probability</span>
                </div>
                <div style="color: #477A55; font-weight: bold;">➔</div>
                <div style="background: #FFFFFF; border: 1px solid #E4E4DD; border-radius: 8px; padding: 8px 12px; text-align: center;">
                    <span style="color: #1F1F1F; font-weight: 700;">Risk Assessment</span><br><span style="color: #6B6B63;">4 Risk Calibrations</span>
                </div>
                <div style="color: #477A55; font-weight: bold;">➔</div>
                <div style="background: #FFFFFF; border: 1px solid #E4E4DD; border-radius: 8px; padding: 8px 12px; text-align: center;">
                    <span style="color: #1F1F1F; font-weight: 700;">Maintenance Guidance</span><br><span style="color: #6B6B63;">Diagnostic Rules</span>
                </div>
                <div style="color: #477A55; font-weight: bold;">➔</div>
                <div style="background: #EFEFF5; border: 1px solid #CDCBDD; border-radius: 8px; padding: 8px 12px; text-align: center;">
                    <span style="color: #5D587A; font-weight: 700;">Operations UI</span><br><span style="color: #5D587A;">Executive Analytics</span>
                </div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


def render_footer():
    """Renders the global application footer with research citations and constraints."""
    st.markdown("""
    <div class="app-footer">
        <strong>Manufacturing Analytics Intelligence</strong> | AI-Assisted Predictive Maintenance System | 
        AI4I 2020 CNC Telemetry Benchmark | LightGBM Primary Model | <em>Decision-Support Research Prototype</em>
    </div>
    """, unsafe_allow_html=True)
