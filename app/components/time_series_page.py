"""
Page 12: Advanced Time-Series Architecture Component for Streamlit Predictive Maintenance Dashboard (F6).
Redesigned with clean light SaaS aesthetics, pipeline diagram, and future-ready indicators.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Setup paths
APP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.time_series_pipeline import (
    generate_synthetic_sequential_fixture,
    validate_temporal_telemetry,
    compute_derived_temporal_features,
    build_sequence_windows,
    chronological_split,
    get_temporal_model_status,
    SUPPORTED_SEQUENCE_LENGTHS,
    DEFAULT_SEQUENCE_LENGTH
)


def render_time_series_page():
    """Renders the dedicated 12 Time-Series Architecture dashboard page."""
    # 1. PAGE HEADER & RESEARCH NOTICE
    st.markdown("""
    <div class="content-card">
        <div style="font-size: 1.25rem; font-weight: 700; color: #1F1F1F; margin-bottom: 4px;">
            ⏳ Advanced Time-Series Architecture
        </div>
        <div style="font-size: 0.85rem; color: #6B6B63; line-height: 1.5;">
            Architectural preparation for high-frequency sequential CNC sensor streams.
            Establishes sliding window tensor transformations, temporal lag/rolling dynamics, and future deep-learning model interfaces.
        </div>
        <div class="disclaimer-box" style="margin-top: 12px; margin-bottom: 0px; font-size: 0.78rem;">
            <strong>⚠️ Scientific Limitation Notice:</strong> The AI4I 2020 dataset comprises independent milling cycle records. No temporal model (LSTM/Transformer) is currently deployed as the production model. The validated <strong>LightGBM</strong> classifier remains the authoritative operational baseline.
        </div>
    </div>
    """, unsafe_allow_html=True)

    status_info = get_temporal_model_status()

    # 2. STATUS & SPECIFICATION CARDS
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">PRODUCTION BASELINE</div>
            <div class="kpi-value kpi-healthy" style="font-size: 1.4rem;">LightGBM</div>
            <div class="kpi-subtitle">Validated ML Classifier</div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">ARCHITECTURE STATE</div>
            <div class="kpi-value" style="font-size: 1.4rem;">READY</div>
            <div class="kpi-subtitle">Sequence Pipelines Prepared</div>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">TEMPORAL MODEL STATUS</div>
            <div class="kpi-value kpi-watch" style="font-size: 1.15rem;">NOT DEPLOYED</div>
            <div class="kpi-subtitle">Awaiting SCADA Telemetry</div>
        </div>
        """, unsafe_allow_html=True)

    with k4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">DEFAULT SEQUENCE WINDOW</div>
            <div class="kpi-value kpi-analytical" style="font-size: 1.4rem;">{DEFAULT_SEQUENCE_LENGTH} Steps</div>
            <div class="kpi-subtitle">Configurable [5 to 50]</div>
        </div>
        """, unsafe_allow_html=True)

    # 3. ARCHITECTURAL FLOWCHART (CURRENT VS FUTURE READY)
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="content-card">
        <div style="font-size: 1.05rem; font-weight: 700; color: #1F1F1F; margin-bottom: 12px;">
            🏗️ End-to-End Sequential Architecture Pipeline
        </div>
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; font-size: 0.8rem; text-align: center;">
            <div style="background: #F8F9F3; padding: 14px; border: 1px solid #E4E4DD; border-radius: 10px;">
                <div style="color: #1F1F1F; font-weight: 700; margin-bottom: 4px;">1. INGESTION</div>
                <div style="color: #6B6B63;">Continuous Telemetry Stream (1–10 Hz)</div>
                <div style="margin-top: 8px;"><span class="badge-risk-low" style="font-size: 0.7rem;">VALIDATED</span></div>
            </div>
            <div style="background: #F8F9F3; padding: 14px; border: 1px solid #E4E4DD; border-radius: 10px;">
                <div style="color: #1F1F1F; font-weight: 700; margin-bottom: 4px;">2. QUALITY GATING</div>
                <div style="color: #6B6B63;">Timestamp Continuity & Machine Isolation</div>
                <div style="margin-top: 8px;"><span class="badge-risk-low" style="font-size: 0.7rem;">VALIDATED</span></div>
            </div>
            <div style="background: #F8F9F3; padding: 14px; border: 1px solid #E4E4DD; border-radius: 10px;">
                <div style="color: #1F1F1F; font-weight: 700; margin-bottom: 4px;">3. FEATURE TENSORS</div>
                <div style="color: #6B6B63;">Sliding Window 3D Tensors & Lags</div>
                <div style="margin-top: 8px;"><span class="badge-risk-low" style="font-size: 0.7rem;">VALIDATED</span></div>
            </div>
            <div style="background: #EFEFF5; padding: 14px; border: 1px solid #CDCBDD; border-radius: 10px;">
                <div style="color: #5D587A; font-weight: 700; margin-bottom: 4px;">4. TEMPORAL INFERENCE</div>
                <div style="color: #6B6B63;">Future LSTM / Transformer Sequence Modeling</div>
                <div style="margin-top: 8px;"><span class="badge-future" style="font-size: 0.7rem;">FUTURE READY</span></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 4. INTERACTIVE SEQUENCE WINDOW & FEATURE DEMO
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size: 1rem; font-weight: 700; color: #1F1F1F; margin-bottom: 8px;">
        🧪 Interactive Sequence Window & Temporal Feature Demonstration
    </div>
    """, unsafe_allow_html=True)

    c_w1, c_w2, c_w3 = st.columns([1.5, 1.5, 3])
    with c_w1:
        sel_len = st.selectbox("Configured Sequence Length (L)", SUPPORTED_SEQUENCE_LENGTHS, index=2)
    with c_w2:
        sel_steps = st.slider("Simulated Stream Length", min_value=50, max_value=200, value=100, step=25)
    with c_w3:
        st.markdown(f"""
        <div style="background: #F8F9F3; border: 1px solid #E4E4DD; padding: 10px 14px; border-radius: 8px; font-size: 0.8rem; margin-top: 18px;">
            Tensor Shape Output: <strong>(N - {sel_len} + 1, {sel_len}, 5)</strong> representing [Batches, TimeSteps, Sensors].
        </div>
        """, unsafe_allow_html=True)

    # Generate sequential data fixture
    df_seq = generate_synthetic_sequential_fixture(n_steps=sel_steps)
    val_report = validate_temporal_telemetry(df_seq)
    df_temporal_feats = compute_derived_temporal_features(df_seq)
    X_tensor, y_tensor = build_sequence_windows(df_seq, sequence_length=sel_len)

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        # Multi-sensor continuous stream Plotly chart
        fig_stream = go.Figure()
        fig_stream.add_trace(go.Scatter(x=df_seq["Timestamp"], y=df_seq["Torque [Nm]"], name="Torque [Nm]", line=dict(color="#1F1F1F", width=2)))
        fig_stream.add_trace(go.Scatter(x=df_seq["Timestamp"], y=df_seq["Tool wear [min]"], name="Tool Wear [min]", line=dict(color="#9A7A32", width=2)))
        fig_stream.add_trace(go.Scatter(x=df_seq["Timestamp"], y=df_seq["Rotational speed [rpm]"] / 30.0, name="RPM (Scaled /30)", line=dict(color="#477A55", width=1.5, dash="dot")))
        
        fig_stream.update_layout(
            title="<b>Continuous Multi-Sensor Telemetry Sequence</b>",
            xaxis_title="Simulation Timestamp",
            yaxis_title="Sensor Magnitude",
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=300,
            margin=dict(l=20, r=20, t=45, b=20),
            font={'family': 'Inter, sans-serif', 'color': '#1F1F1F'},
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig_stream.update_xaxes(linecolor="#E4E4DD")
        fig_stream.update_yaxes(gridcolor="#ECECE6", linecolor="#E4E4DD")
        st.plotly_chart(fig_stream, use_container_width=True)

    with col_g2:
        # Rate of change / First difference chart
        fig_diff = go.Figure()
        fig_diff.add_trace(go.Scatter(x=df_temporal_feats["Timestamp"], y=df_temporal_feats["Torque [Nm]_first_diff"], name="Torque First-Diff (dX/dt)", line=dict(color="#8B3F3F", width=1.5)))
        fig_diff.add_trace(go.Scatter(x=df_temporal_feats["Timestamp"], y=df_temporal_feats["Torque [Nm]_roll_mean"], name="Torque Rolling Mean (w=5)", line=dict(color="#1F1F1F", width=2, dash="dash")))

        fig_diff.update_layout(
            title="<b>Derived Dynamic Features (Rate-of-Change & Rolling Mean)</b>",
            xaxis_title="Simulation Timestamp",
            yaxis_title="Delta / Smoothed Value",
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=300,
            margin=dict(l=20, r=20, t=45, b=20),
            font={'family': 'Inter, sans-serif', 'color': '#1F1F1F'},
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig_diff.update_xaxes(linecolor="#E4E4DD")
        fig_diff.update_yaxes(gridcolor="#ECECE6", linecolor="#E4E4DD")
        st.plotly_chart(fig_diff, use_container_width=True)

    # 5. DATA QUALITY CHECKER & FEATURE REGISTRY
    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
    with st.expander("📋 Temporal Data Quality Inspection & Feature Schema", expanded=False):
        st.markdown(f"""
        <div style="font-size: 0.82rem; color: #6B6B63; margin-bottom: 12px;">
            Quality status: <strong style="color: {'#477A55' if val_report['status']=='VALID' else '#9A7A32'};">{val_report['status']}</strong> | Total records: <strong>{val_report['total_records']}</strong> | Machines: <strong>{val_report['unique_machines']}</strong>
        </div>
        """, unsafe_allow_html=True)

        st.dataframe(df_temporal_feats.head(10), use_container_width=True, height=250)

    # 6. FUTURE TEMPORAL MODEL INTERFACE & CHRONOLOGICAL SPLIT CODE
    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
    with st.expander("💻 TemporalPredictor Interface & Chronological Split Specification", expanded=False):
        st.code("""
# Future Temporal Model Interface Specification (src/time_series_pipeline.py)
class TemporalPredictor(ABC):
    @abstractmethod
    def fit(self, X_seq: np.ndarray, y_seq: np.ndarray) -> "TemporalPredictor":
        \"\"\"Fits on 3D sequence tensors (N_sequences, sequence_length, N_features).\"\"\"
        pass

    @abstractmethod
    def predict_proba(self, X_seq: np.ndarray) -> np.ndarray:
        \"\"\"Predicts failure probability for sequence windows.\"\"\"
        pass

# Chronological Split (No random leakage across time):
# Train: 0% -> 70% | Validation: 70% -> 85% | Test: 85% -> 100%
df_train, df_val, df_test = chronological_split(df_sequential, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
        """, language="python")
