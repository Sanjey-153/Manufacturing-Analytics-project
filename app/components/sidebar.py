"""
Sidebar Navigation Component for Manufacturing Analytics Dashboard.
"""
import streamlit as st


def render_sidebar():
    """Renders the custom navigation sidebar with brand header, nav menu, and specs panel."""
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-brand-card">
            <div style="font-size: 1.35rem; color: #1F1F1F;">⚡</div>
            <div class="sidebar-logo-text">Manufacturing Analytics</div>
            <div class="sidebar-sub-text">Predictive Maintenance</div>
        </div>
        """, unsafe_allow_html=True)

        nav_options = [
            "01 Overview",
            "02 Machine Prediction",
            "03 Batch Telemetry",
            "04 Manufacturing Analytics",
            "05 Model Performance",
            "06 Explainability",
            "07 System Architecture",
            "08 Live Telemetry",
            "09 Drift Monitoring",
            "10 Threshold & Cost Analysis",
            "11 Maintenance Work Orders",
            "12 Time-Series Architecture",
            "13 Industrial Validation"
        ]

        selected_nav = st.radio(
            "SYSTEM NAVIGATION",
            nav_options,
            index=0,
            label_visibility="collapsed"
        )

        st.markdown("""
        <div class="sidebar-specs-card">
            <div class="sidebar-specs-header">SYSTEM SPECIFICATIONS</div>
            <div class="sidebar-spec-row">
                <span class="sidebar-spec-key">Model:</span>
                <span class="sidebar-spec-val">LightGBM</span>
            </div>
            <div class="sidebar-spec-row">
                <span class="sidebar-spec-key">Preprocessor:</span>
                <span class="sidebar-spec-val">ColumnTransformer</span>
            </div>
            <div class="sidebar-spec-row">
                <span class="sidebar-spec-key">Operational Cutoff:</span>
                <span class="sidebar-spec-val">0.50</span>
            </div>
            <div class="sidebar-spec-row">
                <span class="sidebar-spec-key">Holdout Test N:</span>
                <span class="sidebar-spec-val">2,000</span>
            </div>
            <div class="sidebar-spec-row">
                <span class="sidebar-spec-key">Validation Checks:</span>
                <span class="sidebar-spec-val">166 / 166</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="sidebar-online-status">
            <span style="color: #6B6B63; font-weight: 600;">SYSTEM</span>
            <span style="color: #477A55; font-weight: 700; display: flex; align-items: center; gap: 6px;">
                <span class="status-dot-healthy"></span> OPERATIONAL
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="disclaimer-box" style="margin-top: 14px; font-size: 0.72rem;">
            <strong>Decision-Support Prototype:</strong> AI4I 2020 CNC Benchmark. Does not actuate physical hardware.
        </div>
        """, unsafe_allow_html=True)

        return selected_nav
