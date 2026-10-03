"""
Risk Gauge, Status Banners, and Maintenance Guidance Components.
"""
import streamlit as st
import plotly.graph_objects as go


def create_risk_gauge(prob: float, threshold: float = 0.50):
    """
    Creates a high-precision industrial risk probability gauge with muted risk tiers and threshold mark.
    """
    prob_pct = prob * 100

    if prob < 0.20:
        bar_color = "#477A55"
        tier_label = "LOW"
    elif prob < 0.50:
        bar_color = "#9A7A32"
        tier_label = "MODERATE"
    elif prob < 0.80:
        bar_color = "#A05A3A"
        tier_label = "HIGH"
    else:
        bar_color = "#8B3F3F"
        tier_label = "CRITICAL"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob_pct,
        number={
            'suffix': "%",
            'font': {'size': 36, 'color': '#1F1F1F', 'family': 'Inter, sans-serif'},
            'valueformat': '.2f'
        },
        delta={
            'reference': threshold * 100,
            'increasing': {'color': '#8B3F3F'},
            'decreasing': {'color': '#477A55'},
            'font': {'size': 13, 'family': 'Inter, sans-serif'}
        },
        title={
            'text': f"<span style='font-size:0.9em;color:#6B6B63;font-weight:600;'>FAILURE PROBABILITY</span><br><span style='font-size:0.8em;color:{bar_color};font-weight:700;'>PROTOTYPE RISK: {tier_label}</span>",
            'font': {'size': 14, 'color': '#6B6B63', 'family': 'Inter, sans-serif'}
        },
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 1,
                'tickcolor': "#E4E4DD",
                'tickfont': {'color': '#6B6B63', 'size': 10, 'family': 'Inter, sans-serif'}
            },
            'bar': {'color': bar_color, 'thickness': 0.26},
            'bgcolor': "#F5F5F5",
            'borderwidth': 1,
            'bordercolor': "#E4E4DD",
            'steps': [
                {'range': [0, 20], 'color': "#EBF3ED"},
                {'range': [20, 50], 'color': "#FAF5E8"},
                {'range': [50, 80], 'color': "#F8EFEA"},
                {'range': [80, 100], 'color': "#F7EDED"}
            ],
            'threshold': {
                'line': {'color': "#1F1F1F", 'width': 2.5},
                'thickness': 0.75,
                'value': threshold * 100
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=25, r=25, t=50, b=25),
        height=230,
        font={'family': 'Inter, sans-serif', 'color': '#1F1F1F'}
    )
    return fig


def render_status_banner(pred_class: int, risk_level: str, prob: float):
    """Renders the top prediction status banner with light SaaS styling."""
    if pred_class == 0:
        st.markdown(f"""
        <div class="result-banner-healthy">
            <div class="result-title">✓ Machine Healthy</div>
            <div class="result-subtitle">Operating parameters within nominal baseline envelope. Failure probability: <strong>{prob*100:.2f}%</strong></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-banner-failure">
            <div class="result-title">⚠ Failure Risk Detected</div>
            <div class="result-subtitle">Operating parameters exceed operational threshold. Risk tier: <strong>{risk_level.upper()}</strong> ({prob*100:.2f}%)</div>
        </div>
        """, unsafe_allow_html=True)


def render_maintenance_guidance(res: dict):
    """Renders conservative, engineering-grounded maintenance guidance and diagnostic observations."""
    risk = res.get("risk_level", "Low")
    prob = res.get("failure_probability", 0.0)

    if risk == "Low":
        badge_cls = "badge-risk-low"
    elif risk == "Moderate":
        badge_cls = "badge-risk-mod"
    elif risk == "High":
        badge_cls = "badge-risk-high"
    else:
        badge_cls = "badge-risk-crit"

    st.markdown(f"""
    <div class="recommendation-card">
        <div class="recommendation-header">
            <span>🛠️</span> MAINTENANCE GUIDANCE & ASSESSMENT
            <span class="{badge_cls}" style="margin-left: auto;">{risk.upper()} RISK</span>
        </div>
        <div style="font-size: 0.88rem; color: #1F1F1F; margin-bottom: 12px; line-height: 1.55;">
            {res.get('summary', 'Telemetry evaluated successfully.')}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Diagnostic Flags
    diag_flags = res.get("diagnostic_flags", [])
    if diag_flags:
        for flag in diag_flags:
            st.markdown(f"""
            <div style="padding: 8px 12px; background: #FAF5E8; border-left: 3px solid #9A7A32; border-radius: 4px; color: #9A7A32; font-size: 0.8rem; margin-top: 6px;">
                ⚠️ <strong>Diagnostic Flag:</strong> {flag}
            </div>
            """, unsafe_allow_html=True)

    # Actionable Protocol Steps
    actions = res.get("recommended_actions", [])
    if actions:
        st.markdown("<div style='font-size: 0.76rem; color: #6B6B63; margin-top: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;'>RECOMMENDED ACTION PROTOCOLS:</div>", unsafe_allow_html=True)
        for act in actions:
            st.markdown(f"<div style='font-size: 0.82rem; color: #1F1F1F; margin-left: 10px; margin-top: 3px;'>• {act}</div>", unsafe_allow_html=True)

    # Conservative advisory disclaimer
    st.markdown("""
    <div style="font-size: 0.72rem; color: #8E8E85; margin-top: 12px; line-height: 1.4; border-top: 1px solid #ECECE6; padding-top: 8px;">
        * Guidance is derived from AI4I heuristic rules and decision tree splits. These are algorithmic decision-support aids and do not constitute physical machine overrides or industry-certified standards.
    </div>
    """, unsafe_allow_html=True)
