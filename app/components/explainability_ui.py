"""
SHAP Explainability UI Components for Manufacturing Analytics.
Redesigned with Light SaaS Aesthetics and subtle analytical accents.
"""
from typing import Dict, Any, Optional
import pandas as pd
import streamlit as st

from components.charts import create_shap_contribution_chart, create_global_shap_chart


def render_shap_explanation_card(
    explanation: Dict[str, Any],
    title: str = "🧠 MODEL EXPLANATION (SHAP ATTRIBUTION)",
    show_table: bool = True,
    chart_key: Optional[str] = None
):
    """
    Renders an explainability card showing why the model produced this specific prediction.
    Answers: 'Which input features contributed to this prediction, and in which direction?'
    """
    if not explanation:
        st.warning("No explanation data available.")
        return

    top_contribs = explanation.get("top_contributions", [])
    prob = explanation.get("failure_probability", 0.0)
    risk_level = explanation.get("risk_level", "Low")
    pred_class = explanation.get("predicted_class", 0)

    st.markdown(f"""
    <div class="content-card" style="margin-top: 16px; padding: 18px 20px;">
        <div style="font-size: 0.95rem; font-weight: 700; color: #1F1F1F; margin-bottom: 4px; display: flex; align-items: center; gap: 8px;">
            <span>🔍</span> {title}
        </div>
        <div style="font-size: 0.8rem; color: #6B6B63; margin-bottom: 10px;">
            Per-prediction feature attribution computed via <strong>SHAP TreeExplainer</strong> for the LightGBM classifier.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 1. Horizontal SHAP Contribution Bar Chart
    fig_shap = create_shap_contribution_chart(explanation, top_n=8)
    st.plotly_chart(fig_shap, use_container_width=True, key=chart_key)

    # 2. Feature Contribution Table
    if show_table and top_contribs:
        st.markdown("<div style='font-size: 0.8rem; font-weight: 700; color: #1F1F1F; margin: 12px 0 6px 0; text-transform: uppercase;'>TOP CONTRIBUTING SENSOR PARAMETERS</div>", unsafe_allow_html=True)
        
        table_rows = []
        for c in top_contribs:
            table_rows.append({
                "Feature": c["display_name"],
                "Current Value": c["current_value"],
                "SHAP Contribution": f"{c['shap_value']:+.4f}",
                "Direction": c["direction"]
            })
        
        df_table = pd.DataFrame(table_rows)
        st.dataframe(df_table, use_container_width=True, height=180)

    # 3. Scientific Notice
    st.markdown(f"""
    <div class="disclaimer-box" style="font-size: 0.76rem; margin-top: 10px; margin-bottom: 0px;">
        <strong>Explainability Notice:</strong> {explanation.get('disclaimer')}
    </div>
    """, unsafe_allow_html=True)
