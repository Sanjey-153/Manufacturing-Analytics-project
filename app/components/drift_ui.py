"""
Drift Monitoring UI Components for Manufacturing Predictive Maintenance (Future Work F3).
Redesigned with Light SaaS Aesthetics and clean KPI cards.
"""
from typing import Dict, Any, List, Optional
import pandas as pd
import streamlit as st

from src.drift_monitor import (
    STATUS_INSUFFICIENT_DATA,
    STATUS_STABLE,
    STATUS_WATCH,
    STATUS_DRIFT_DETECTED
)
from components.charts import (
    create_feature_psi_chart,
    create_distribution_comparison_chart,
    create_probability_drift_chart,
    create_risk_tier_comparison_chart
)


def render_drift_kpis(drift_report: Dict[str, Any]):
    """Renders 6 top KPI cards for the Drift Monitoring overview."""
    overall_status = drift_report.get("overall_status", STATUS_INSUFFICIENT_DATA)
    window_size = drift_report.get("window_size", 0)
    most_affected = drift_report.get("most_affected_feature") or "None"
    max_psi = drift_report.get("max_psi", 0.0)
    
    quality_data = drift_report.get("data_quality", {})
    quality_status = quality_data.get("quality_status", "CLEAN")
    
    model_output = drift_report.get("model_output_drift", {})
    model_psi = model_output.get("psi", 0.0)
    model_status = model_output.get("status", "STABLE")

    if overall_status == STATUS_STABLE:
        status_cls = "kpi-healthy"
    elif overall_status == STATUS_WATCH:
        status_cls = "kpi-watch"
    elif overall_status == STATUS_DRIFT_DETECTED:
        status_cls = "kpi-critical"
    else:
        status_cls = ""

    qual_cls = "kpi-healthy" if quality_status == "CLEAN" else ("kpi-watch" if quality_status == "WARNING" else "kpi-critical")
    model_cls = "kpi-healthy" if model_status == STATUS_STABLE else ("kpi-watch" if model_status == STATUS_WATCH else "kpi-critical")

    k1, k2, k3, k4, k5, k6 = st.columns(6)

    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">OVERALL DRIFT STATUS</div>
            <div class="kpi-value {status_cls}" style="font-size: 1.15rem; font-weight: 700;">{overall_status}</div>
            <div class="kpi-subtitle">Prototype Thresholds</div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">MONITORING WINDOW</div>
            <div class="kpi-value">{window_size} <span style="font-size: 0.8rem; color: #6B6B63;">events</span></div>
            <div class="kpi-subtitle">Min Required: 50</div>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">FEATURES MONITORED</div>
            <div class="kpi-value">6</div>
            <div class="kpi-subtitle">1 Cat + 5 Num</div>
        </div>
        """, unsafe_allow_html=True)

    with k4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">MOST AFFECTED FEATURE</div>
            <div class="kpi-value kpi-analytical" style="font-size: 1.05rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{most_affected}</div>
            <div class="kpi-subtitle">Peak PSI: {max_psi:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    with k5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">MODEL OUTPUT DRIFT</div>
            <div class="kpi-value {model_cls}" style="font-size: 1.15rem;">{model_status}</div>
            <div class="kpi-subtitle">Prob PSI: {model_psi:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    with k6:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">DATA QUALITY</div>
            <div class="kpi-value {qual_cls}" style="font-size: 1.15rem;">{quality_status}</div>
            <div class="kpi-subtitle">{quality_data.get('total_issues', 0)} Anomalies</div>
        </div>
        """, unsafe_allow_html=True)


def render_feature_drift_table(drift_report: Dict[str, Any]):
    """Renders the detailed feature-level drift metrics table."""
    feat_drift = drift_report.get("feature_drift", {})
    if not feat_drift:
        st.info("No feature drift metrics available for current window.")
        return

    table_rows = []
    for col, data in feat_drift.items():
        ks_stat_str = f"{data['ks_statistic']:.4f}" if data.get('ks_statistic') is not None else "N/A"
        ks_pval_str = f"{data['ks_pvalue']:.4f}" if data.get('ks_pvalue') is not None else "N/A"
        
        if data.get("type") == "categorical":
            ref_summary = "H: 9.9%, M: 29.8%, L: 60.4%"
            curr_dist = data.get("curr_distribution", {})
            curr_summary = ", ".join([f"{k}: {v*100:.1f}%" for k, v in curr_dist.items()])
        else:
            unit = data.get("unit", "")
            ref_summary = f"{data.get('ref_median', 0.0):.1f} {unit}".strip()
            curr_summary = f"{data.get('curr_median', 0.0):.1f} {unit}".strip()

        table_rows.append({
            "Feature Name": data.get("display_name", col),
            "Type": data.get("type", "").capitalize(),
            "Reference Baseline (N=8,000)": ref_summary,
            "Current Window": curr_summary,
            "PSI Metric": f"{data.get('psi', 0.0):.4f}",
            "KS Statistic": ks_stat_str,
            "KS p-value": ks_pval_str,
            "Drift Status": data.get("status", "STABLE")
        })

    df_table = pd.DataFrame(table_rows)
    st.dataframe(df_table, use_container_width=True, height=260)


def render_data_quality_panel(quality_report: Dict[str, Any]):
    """Renders the data quality breakdown panel."""
    q_col1, q_col2, q_col3, q_col4 = st.columns(4)
    with q_col1:
        st.markdown(f"""
        <div style="background: #FFFFFF; border: 1px solid #E4E4DD; border-radius: 8px; padding: 12px 14px;">
            <div style="font-size: 0.72rem; color: #6B6B63;">MISSING / NULLS</div>
            <div style="font-size: 1.25rem; font-weight: 700; color: #1F1F1F;">{quality_report.get('missing_count', 0)}</div>
        </div>
        """, unsafe_allow_html=True)
    with q_col2:
        st.markdown(f"""
        <div style="background: #FFFFFF; border: 1px solid #E4E4DD; border-radius: 8px; padding: 12px 14px;">
            <div style="font-size: 0.72rem; color: #6B6B63;">NAN / INF VALUES</div>
            <div style="font-size: 1.25rem; font-weight: 700; color: #1F1F1F;">{quality_report.get('nan_count', 0) + quality_report.get('infinite_count', 0)}</div>
        </div>
        """, unsafe_allow_html=True)
    with q_col3:
        st.markdown(f"""
        <div style="background: #FFFFFF; border: 1px solid #E4E4DD; border-radius: 8px; padding: 12px 14px;">
            <div style="font-size: 0.72rem; color: #6B6B63;">UNEXPECTED CATEGORIES</div>
            <div style="font-size: 1.25rem; font-weight: 700; color: #1F1F1F;">{quality_report.get('unexpected_categories_count', 0)}</div>
        </div>
        """, unsafe_allow_html=True)
    with q_col4:
        st.markdown(f"""
        <div style="background: #FFFFFF; border: 1px solid #E4E4DD; border-radius: 8px; padding: 12px 14px;">
            <div style="font-size: 0.72rem; color: #6B6B63;">RANGE VIOLATIONS</div>
            <div style="font-size: 1.25rem; font-weight: 700; color: #1F1F1F;">{quality_report.get('range_violations_count', 0)}</div>
        </div>
        """, unsafe_allow_html=True)

    issues = quality_report.get("issue_details", [])
    if issues:
        with st.expander(f"Data Quality Issue Log ({len(issues)} items)", expanded=True):
            for iss in issues[:10]:
                st.markdown(f"<div style='font-size: 0.76rem; color: #8B3F3F;'>⚠ {iss}</div>", unsafe_allow_html=True)
