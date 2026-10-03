"""
Top Utility Header Bar Component for Manufacturing Analytics SaaS Platform.
"""
import streamlit as st


def render_header(model_name: str = "LightGBM", system_status: str = "APPLICATION READY"):
    """Renders the clean top SaaS utility bar with title, model badge, and system status."""
    st.markdown(f"""
    <div class="top-header-bar">
        <div class="top-header-left">
            <div class="top-header-icon">⚙️</div>
            <div>
                <div class="top-header-title">PREDICTIVE MAINTENANCE INTELLIGENCE</div>
                <div class="top-header-subtitle">CNC Machine Manufacturing Analytics</div>
            </div>
        </div>
        <div class="top-header-right">
            <div class="model-badge">Model: {model_name}</div>
            <div class="model-badge">Threshold: 0.50</div>
            <div class="status-indicator-badge">
                <span class="status-dot-healthy"></span>
                <span>{system_status}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
