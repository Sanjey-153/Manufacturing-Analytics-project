"""
Page 10: Threshold & Cost Analysis Component for Streamlit Predictive Maintenance Dashboard (F4).
Redesigned with clean light SaaS aesthetics, clear 0.50 operational baseline, and decision sensitivity charts.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import streamlit as st

# Setup paths
APP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.threshold_analysis import (
    get_holdout_predictions,
    calculate_threshold_metrics,
    generate_threshold_sweep,
    find_cost_minimizing_threshold,
    compare_thresholds,
    build_cost_sensitivity_matrix,
    ILLUSTRATIVE_COST_SCENARIOS,
    DISCLAIMER_TEXT
)
from components.charts import (
    create_expected_cost_curve_chart,
    create_threshold_tradeoff_curves_chart,
    create_confusion_matrix_comparison_chart,
    create_cost_ratio_sensitivity_heatmap
)


@st.cache_data
def get_cached_holdout_data():
    """Caches holdout ground truth and LightGBM failure probability predictions."""
    y_true, y_prob, _ = get_holdout_predictions()
    return y_true, y_prob


def render_threshold_page(model: Optional[Any] = None, preprocessor: Optional[Any] = None):
    """
    Renders the dedicated 10 Threshold & Cost Analysis dashboard page.
    """
    # 1. PAGE HEADER & DISCLAIMER
    st.markdown("""
    <div class="content-card">
        <div style="font-size: 1.25rem; font-weight: 700; color: #1F1F1F; margin-bottom: 4px;">
            ⚖️ Decision Threshold & Cost Analysis
        </div>
        <div style="font-size: 0.85rem; color: #6B6B63; line-height: 1.5;">
            Evaluate decision boundaries and trade-offs on the validated <strong>AI4I holdout test set (N=2,000)</strong>.
            Examine how varying decision costs alter optimal operating thresholds.
        </div>
        <div class="disclaimer-box" style="margin-top: 12px; margin-bottom: 0px; font-size: 0.78rem;">
            <strong>⚠️ Mandatory Operational Notice:</strong> Threshold analysis is based on the benchmark holdout set. Cost values are user-defined assumptions. The operational classification threshold for all deployment inference and live telemetry remains fixed at <strong>0.50</strong>.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Load holdout predictions
    try:
        y_true, y_prob = get_cached_holdout_data()
    except Exception as e:
        st.error(f"Failed to load holdout benchmark predictions: {e}")
        return

    n_total = len(y_true)
    n_pos = int(np.sum(y_true))
    n_neg = n_total - n_pos

    # 2. COST ASSUMPTIONS PANEL
    st.markdown("""
    <div style="font-size: 1rem; font-weight: 700; color: #1F1F1F; margin-bottom: 10px;">
        ⚙️ Cost Assumption Configuration
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        c_top1, c_top2 = st.columns([1.5, 2.5])
        
        with c_top1:
            scenario_choice = st.selectbox(
                "Select Cost Scenario Template",
                list(ILLUSTRATIVE_COST_SCENARIOS.keys()) + ["Custom User-Defined Scenario"],
                index=0,
                help="Choose an illustrative scenario or enter customized cost values."
            )

        with c_top2:
            if scenario_choice in ILLUSTRATIVE_COST_SCENARIOS:
                scen_data = ILLUSTRATIVE_COST_SCENARIOS[scenario_choice]
                c_fn_init = float(scen_data["C_FN"])
                c_fp_init = float(scen_data["C_FP"])
                c_tp_init = float(scen_data["C_TP"])
                c_tn_init = float(scen_data["C_TN"])
                scen_desc = scen_data["description"]
            else:
                c_fn_init = 50.0
                c_fp_init = 5.0
                c_tp_init = 2.0
                c_tn_init = 0.0
                scen_desc = "User-customized cost parameters."

            st.markdown(f"""
            <div style="background: #F8F9F3; border: 1px solid #E4E4DD; padding: 10px 14px; border-radius: 8px; font-size: 0.8rem; margin-top: 24px; color: #6B6B63;">
                <strong>Scenario Rationale:</strong> {scen_desc}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        with col_c1:
            c_fn = st.number_input("C_FN (Unpredicted Failure)", min_value=0.0, max_value=500.0, value=c_fn_init, step=5.0)
        with col_c2:
            c_fp = st.number_input("C_FP (Unnecessary Inspection)", min_value=0.0, max_value=100.0, value=c_fp_init, step=1.0)
        with col_c3:
            c_tp = st.number_input("C_TP (Planned Intervention)", min_value=0.0, max_value=100.0, value=c_tp_init, step=0.5)
        with col_c4:
            c_tn = st.number_input("C_TN (Nominal Operation)", min_value=0.0, max_value=50.0, value=c_tn_init, step=0.5)

    # 3. COMPUTE THRESHOLD SWEEP & OPTIMAL THRESHOLD
    df_sweep = generate_threshold_sweep(y_true, y_prob, c_tn=c_tn, c_fp=c_fp, c_fn=c_fn, c_tp=c_tp, step=0.01)
    opt_info = find_cost_minimizing_threshold(df_sweep)
    opt_thresh = float(opt_info["Threshold"])

    # 4. THRESHOLD COMPARISON KPIS (0.50 Baseline vs Analysis Threshold)
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size: 1rem; font-weight: 700; color: #1F1F1F; margin-bottom: 10px;">
        📊 Threshold Benchmark Comparison (Holdout N = 2,000)
    </div>
    """, unsafe_allow_html=True)

    metrics_050 = calculate_threshold_metrics(y_true, y_prob, threshold=0.50, c_tn=c_tn, c_fp=c_fp, c_fn=c_fn, c_tp=c_tp)
    metrics_opt = calculate_threshold_metrics(y_true, y_prob, threshold=opt_thresh, c_tn=c_tn, c_fp=c_fp, c_fn=c_fn, c_tp=c_tp)

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">OPERATIONAL BASELINE</div>
            <div class="kpi-value kpi-healthy" style="font-size: 1.5rem;">0.50</div>
            <div class="kpi-subtitle">Recall: {metrics_050['Recall']*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">ANALYSIS THRESHOLD</div>
            <div class="kpi-value kpi-watch" style="font-size: 1.5rem;">{opt_thresh:.2f}</div>
            <div class="kpi-subtitle">Recall: {metrics_opt['Recall']*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">BASELINE COST / OBS</div>
            <div class="kpi-value" style="font-size: 1.5rem;">{metrics_050['Expected_Cost']:.3f}</div>
            <div class="kpi-subtitle">Total: {metrics_050['Total_Cost']:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    with k4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">ANALYSIS COST / OBS</div>
            <div class="kpi-value kpi-healthy" style="font-size: 1.5rem;">{metrics_opt['Expected_Cost']:.3f}</div>
            <div class="kpi-subtitle">Total: {metrics_opt['Total_Cost']:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    with k5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">FALSE NEGATIVES</div>
            <div class="kpi-value" style="font-size: 1.5rem;">{metrics_050['FN']} <span style="font-size: 0.85rem; color: #6B6B63;">vs {metrics_opt['FN']}</span></div>
            <div class="kpi-subtitle">Missed Failures</div>
        </div>
        """, unsafe_allow_html=True)

    with k6:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">FALSE POSITIVES</div>
            <div class="kpi-value" style="font-size: 1.5rem;">{metrics_050['FP']} <span style="font-size: 0.85rem; color: #6B6B63;">vs {metrics_opt['FP']}</span></div>
            <div class="kpi-subtitle">Unnecessary Alarms</div>
        </div>
        """, unsafe_allow_html=True)

    # 5. CHARTS: EXPECTED COST CURVE & TRADEOFF CURVES
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    ch_c1, ch_c2 = st.columns(2)

    with ch_c1:
        fig_cost = create_expected_cost_curve_chart(
            df_sweep,
            operational_threshold=0.50,
            analysis_threshold=opt_thresh
        )
        st.plotly_chart(fig_cost, use_container_width=True)

    with ch_c2:
        fig_tradeoff = create_threshold_tradeoff_curves_chart(
            df_sweep,
            operational_threshold=0.50,
            analysis_threshold=opt_thresh
        )
        st.plotly_chart(fig_tradeoff, use_container_width=True)

    # 6. CHARTS: CONFUSION MATRIX COMPARISON & SENSITIVITY HEATMAP
    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
    ch_c3, ch_c4 = st.columns(2)

    with ch_c3:
        cm_050 = {"TP": metrics_050["TP"], "FN": metrics_050["FN"], "FP": metrics_050["FP"], "TN": metrics_050["TN"]}
        cm_opt = {"TP": metrics_opt["TP"], "FN": metrics_opt["FN"], "FP": metrics_opt["FP"], "TN": metrics_opt["TN"]}
        fig_cm = create_confusion_matrix_comparison_chart(
            cm_050, cm_opt,
            name_a="Operational Baseline (0.50)",
            name_b=f"Analysis Threshold ({opt_thresh:.2f})"
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with ch_c4:
        df_sens = build_cost_sensitivity_matrix(y_true, y_prob)
        fig_sens = create_cost_ratio_sensitivity_heatmap(df_sens)
        st.plotly_chart(fig_sens, use_container_width=True)

    # 7. RESEARCH INTEGRITY FOOTER
    st.markdown(f"""
    <div class="disclaimer-box" style="margin-top: 24px;">
        <strong>Research Integrity Notice:</strong> {DISCLAIMER_TEXT}
    </div>
    """, unsafe_allow_html=True)
