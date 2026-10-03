"""
Plotly Chart Generators for Manufacturing Analytics Dashboard.
Redesigned with Light SaaS Aesthetics: plotly_white, muted semantic palette, and clean typography.
"""
from typing import Dict, Any, List, Optional, Tuple, Union
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Centralized Chart Color Palette
COLOR_BASELINE = "#1F1F1F"
COLOR_HEALTHY = "#477A55"
COLOR_WATCH = "#9A7A32"
COLOR_WARNING = "#A05A3A"
COLOR_CRITICAL = "#8B3F3F"
COLOR_ANALYTICAL = "#5D587A"
COLOR_INFO = "#4D6675"
COLOR_BORDER = "#E4E4DD"
COLOR_GRID = "#ECECE6"


def create_model_comparison_charts(comp_df: pd.DataFrame):
    """Creates interactive Plotly comparison bar charts for Recall, False Negatives, and F1."""
    # Chart 1: Recall Comparison
    colors_recall = [COLOR_INFO if m != 'LightGBM' else COLOR_HEALTHY for m in comp_df['Model']]
    fig_rec = px.bar(
        comp_df,
        x='Model',
        y='Recall',
        title='Fault Recall Comparison (Higher is Better)',
        text=comp_df['Recall'].apply(lambda x: f"{x*100:.2f}%"),
        color='Model',
        color_discrete_map={m: c for m, c in zip(comp_df['Model'], colors_recall)},
        template='plotly_white'
    )
    fig_rec.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        font={'family': 'Inter, sans-serif', 'color': '#1F1F1F'}
    )
    fig_rec.update_traces(textposition='outside')
    fig_rec.update_yaxes(gridcolor=COLOR_GRID, linecolor=COLOR_BORDER)
    fig_rec.update_xaxes(linecolor=COLOR_BORDER)

    # Chart 2: False Negatives Comparison
    colors_fn = [COLOR_WARNING if m != 'LightGBM' else COLOR_HEALTHY for m in comp_df['Model']]
    fig_fn = px.bar(
        comp_df,
        x='Model',
        y='False Negative',
        title='Missed Failures / False Negatives (Lower is Better)',
        text='False Negative',
        color='Model',
        color_discrete_map={m: c for m, c in zip(comp_df['Model'], colors_fn)},
        template='plotly_white'
    )
    fig_fn.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        font={'family': 'Inter, sans-serif', 'color': '#1F1F1F'}
    )
    fig_fn.update_traces(textposition='outside')
    fig_fn.update_yaxes(gridcolor=COLOR_GRID, linecolor=COLOR_BORDER)
    fig_fn.update_xaxes(linecolor=COLOR_BORDER)

    # Chart 3: F1-Score Comparison
    colors_f1 = [COLOR_INFO if m != 'LightGBM' else COLOR_HEALTHY for m in comp_df['Model']]
    fig_f1 = px.bar(
        comp_df,
        x='Model',
        y='F1-Score',
        title='F1-Score Comparison (Balanced Metric)',
        text=comp_df['F1-Score'].apply(lambda x: f"{x*100:.2f}%"),
        color='Model',
        color_discrete_map={m: c for m, c in zip(comp_df['Model'], colors_f1)},
        template='plotly_white'
    )
    fig_f1.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        font={'family': 'Inter, sans-serif', 'color': '#1F1F1F'}
    )
    fig_f1.update_traces(textposition='outside')
    fig_f1.update_yaxes(gridcolor=COLOR_GRID, linecolor=COLOR_BORDER)
    fig_f1.update_xaxes(linecolor=COLOR_BORDER)

    return fig_rec, fig_fn, fig_f1


def create_feature_importance_chart():
    """
    Creates an interactive horizontal bar chart of the validated LightGBM feature importance values.
    """
    fi_data = [
        {"Feature": "Torque [Nm]", "Importance": 605, "Category": "Spindle Dynamics", "Physical Meaning": "Cutting load resistance"},
        {"Feature": "Tool wear [min]", "Importance": 580, "Category": "Degradation", "Physical Meaning": "Cumulative tool cutting wear"},
        {"Feature": "Rotational speed [rpm]", "Importance": 533, "Category": "Kinematics", "Physical Meaning": "Spindle rotational speed"},
        {"Feature": "Air temperature [K]", "Importance": 498, "Category": "Thermal", "Physical Meaning": "Ambient environment temperature"},
        {"Feature": "Process temperature [K]", "Importance": 457, "Category": "Thermal", "Physical Meaning": "Cutting zone temperature"},
        {"Feature": "Type_L (Low Quality)", "Importance": 173, "Category": "Machine Variant", "Physical Meaning": "Low structural tolerance variant"},
        {"Feature": "Type_M (Medium)", "Importance": 90, "Category": "Machine Variant", "Physical Meaning": "Medium standard variant"},
        {"Feature": "Type_H (High Quality)", "Importance": 57, "Category": "Machine Variant", "Physical Meaning": "High structural tolerance variant"}
    ]
    df_fi = pd.DataFrame(fi_data).sort_values("Importance", ascending=True)

    fig = px.bar(
        df_fi,
        x="Importance",
        y="Feature",
        orientation="h",
        title="LightGBM Feature Importance (Split Count & Information Gain)",
        text="Importance",
        color="Importance",
        color_continuous_scale=[[0, "#E4E4DD"], [1, "#1F1F1F"]],
        template="plotly_white",
        hover_data=["Category", "Physical Meaning"]
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=360,
        margin=dict(l=20, r=20, t=40, b=20),
        coloraxis_showscale=False,
        font={'family': 'Inter, sans-serif', 'color': '#1F1F1F'}
    )
    fig.update_traces(textposition='outside')
    fig.update_xaxes(gridcolor=COLOR_GRID, linecolor=COLOR_BORDER)
    fig.update_yaxes(linecolor=COLOR_BORDER)
    return fig


def create_batch_risk_distribution_chart(results_df: pd.DataFrame):
    """Creates a clean donut chart of risk level distribution for batch telemetry uploads."""
    risk_counts = results_df["Risk_Level"].value_counts().reset_index()
    risk_counts.columns = ["Risk_Level", "Count"]

    color_map = {
        "Low": COLOR_HEALTHY,
        "Moderate": COLOR_WATCH,
        "High": COLOR_WARNING,
        "Critical": COLOR_CRITICAL
    }

    fig = px.pie(
        risk_counts,
        names="Risk_Level",
        values="Count",
        title="Batch Risk Stratification Breakdown",
        hole=0.55,
        color="Risk_Level",
        color_discrete_map=color_map,
        template="plotly_white"
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=280,
        margin=dict(l=10, r=10, t=35, b=10),
        font={'family': 'Inter, sans-serif', 'color': '#1F1F1F'},
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
    )
    return fig


def create_live_risk_trend_chart(history_df: pd.DataFrame, threshold: float = 0.50):
    """
    Creates an interactive real-time probability trend line chart with decision boundary and risk regions.
    """
    if history_df.empty or "failure_probability" not in history_df.columns:
        fig = go.Figure()
        fig.update_layout(
            title="Awaiting Live Telemetry Stream...",
            template="plotly_white",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=300
        )
        return fig

    steps = history_df["step"]
    probs = history_df["failure_probability"] * 100

    fig = go.Figure()

    # Probability Trend Line
    fig.add_trace(go.Scatter(
        x=steps,
        y=probs,
        mode='lines+markers',
        name='Failure Probability',
        line=dict(color=COLOR_BASELINE, width=2.5),
        marker=dict(size=5, color=COLOR_BASELINE),
        hovertemplate="Step %{x}<br>Failure Prob: %{y:.2f}%<extra></extra>"
    ))

    # Decision Threshold Line (50%)
    fig.add_hline(
        y=threshold * 100,
        line_dash="dash",
        line_color=COLOR_CRITICAL,
        line_width=2,
        annotation_text=f"Decision Threshold ({threshold*100:.0f}%)",
        annotation_position="top right",
        annotation_font_color=COLOR_CRITICAL,
        annotation_font_size=10
    )

    # Risk Region Shading
    fig.add_hrect(y0=0, y1=20, fillcolor="#EBF3ED", opacity=0.6, line_width=0, layer="below")
    fig.add_hrect(y0=20, y1=50, fillcolor="#FAF5E8", opacity=0.6, line_width=0, layer="below")
    fig.add_hrect(y0=50, y1=80, fillcolor="#F8EFEA", opacity=0.6, line_width=0, layer="below")
    fig.add_hrect(y0=80, y1=100, fillcolor="#F7EDED", opacity=0.6, line_width=0, layer="below")

    # Tier annotations
    fig.add_annotation(x=0.01, y=10, xref="paper", yref="y", text="LOW RISK (<20%)", showarrow=False, font=dict(color=COLOR_HEALTHY, size=9))
    fig.add_annotation(x=0.01, y=35, xref="paper", yref="y", text="MODERATE (20-50%)", showarrow=False, font=dict(color=COLOR_WATCH, size=9))
    fig.add_annotation(x=0.01, y=65, xref="paper", yref="y", text="HIGH RISK (50-80%)", showarrow=False, font=dict(color=COLOR_WARNING, size=9))
    fig.add_annotation(x=0.01, y=90, xref="paper", yref="y", text="CRITICAL (≥80%)", showarrow=False, font=dict(color=COLOR_CRITICAL, size=9))

    fig.update_layout(
        title="Failure Probability Trend & Prototype Risk Tiers",
        xaxis_title="Telemetry Event Step",
        yaxis_title="Failure Probability (%)",
        yaxis=dict(range=[0, 105], ticksuffix="%", gridcolor=COLOR_GRID, linecolor=COLOR_BORDER),
        xaxis=dict(linecolor=COLOR_BORDER),
        template='plotly_white',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=320,
        margin=dict(l=20, r=20, t=45, b=20),
        font={'family': 'Inter, sans-serif', 'color': '#1F1F1F'}
    )
    return fig


def create_live_sensor_trend_charts(history_df: pd.DataFrame):
    """
    Creates individual sensor dynamic trend charts for the live telemetry stream.
    """
    if history_df.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            template="plotly_white",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=200
        )
        return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig

    steps = history_df["step"]

    def _make_sensor_chart(col_name: str, title: str, y_title: str, color: str, fmt: str = ".2f"):
        f = go.Figure()
        f.add_trace(go.Scatter(
            x=steps, y=history_df[col_name],
            mode='lines+markers', name=col_name,
            line=dict(color=color, width=2),
            marker=dict(size=4, color=color),
            hovertemplate=f"Step %{{x}}<br>{col_name}: %{{y:{fmt}}}<extra></extra>"
        ))
        f.update_layout(
            title=title,
            xaxis_title="Step", yaxis_title=y_title,
            template='plotly_white', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=200, margin=dict(l=15, r=15, t=35, b=15),
            font={'family': 'Inter, sans-serif', 'color': '#1F1F1F', 'size': 11}
        )
        f.update_xaxes(linecolor=COLOR_BORDER)
        f.update_yaxes(gridcolor=COLOR_GRID, linecolor=COLOR_BORDER)
        return f

    fig_air = _make_sensor_chart("Air temperature [K]", "1. Air Temperature [K]", "Kelvin (K)", COLOR_INFO)
    fig_proc = _make_sensor_chart("Process temperature [K]", "2. Process Temperature [K]", "Kelvin (K)", COLOR_WATCH)
    fig_speed = _make_sensor_chart("Rotational speed [rpm]", "3. Rotational Speed [RPM]", "RPM", COLOR_BASELINE, fmt=".1f")
    fig_torque = _make_sensor_chart("Torque [Nm]", "4. Torque [Nm]", "Nm", COLOR_CRITICAL)
    fig_wear = _make_sensor_chart("Tool wear [min]", "5. Tool Wear [min]", "Minutes", COLOR_ANALYTICAL, fmt=".1f")

    return fig_air, fig_proc, fig_speed, fig_torque, fig_wear


def create_shap_contribution_chart(explanation_dict: Dict[str, Any], top_n: int = 8):
    """
    Creates an interactive horizontal bar chart of local SHAP feature contributions
    for a single prediction, visually separating risk-increasing and risk-reducing factors.
    """
    contributions = explanation_dict.get("contributions", [])
    if not contributions:
        fig = go.Figure()
        fig.update_layout(title="No SHAP contributions available", template="plotly_white")
        return fig

    sub_contribs = contributions[:top_n]
    sub_contribs = list(reversed(sub_contribs))

    features = [f"{c['display_name']} ({c['current_value']})" for c in sub_contribs]
    shap_vals = [c["shap_value"] for c in sub_contribs]
    colors = [COLOR_CRITICAL if v >= 0 else COLOR_HEALTHY for v in shap_vals]
    hover_texts = [
        f"<b>{c['display_name']}</b><br>"
        f"Current Value: {c['current_value']}<br>"
        f"SHAP Contribution: {c['shap_value']:+.4f}<br>"
        f"Direction: {c['direction']}"
        for c in sub_contribs
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=shap_vals,
        y=features,
        orientation='h',
        marker=dict(color=colors, line=dict(color=COLOR_BORDER, width=1)),
        text=[f"{v:+.3f}" for v in shap_vals],
        textposition='outside',
        hovertext=hover_texts,
        hoverinfo='text'
    ))

    fig.add_vline(x=0, line_width=1.5, line_color=COLOR_BASELINE, line_dash="solid")

    fig.update_layout(
        title="<b>Local SHAP Feature Contributions</b> (Per-Prediction)",
        xaxis_title="SHAP Value (Contribution to Log-Odds)",
        yaxis_title="",
        template='plotly_white',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=320,
        margin=dict(l=20, r=30, t=45, b=20),
        font={'family': 'Inter, sans-serif', 'color': '#1F1F1F'}
    )
    fig.update_xaxes(gridcolor=COLOR_GRID, linecolor=COLOR_BORDER)
    fig.update_yaxes(linecolor=COLOR_BORDER)
    return fig


def create_global_shap_chart(global_summary: Dict[str, Any]):
    """
    Creates a horizontal bar chart of global SHAP feature importance (mean |SHAP|).
    """
    importance = global_summary.get("global_importance", [])
    if not importance:
        fig = go.Figure()
        fig.update_layout(title="No Global SHAP data available", template="plotly_white")
        return fig

    rev_imp = list(reversed(importance))
    features = [item["display_name"] for item in rev_imp]
    scores = [item["mean_abs_shap"] for item in rev_imp]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=scores,
        y=features,
        orientation='h',
        marker=dict(
            color=COLOR_ANALYTICAL,
            line=dict(color=COLOR_BORDER, width=1)
        ),
        text=[f"{s:.4f}" for s in scores],
        textposition='outside',
        hovertemplate="<b>%{y}</b><br>Mean |SHAP Value|: %{x:.4f}<extra></extra>"
    ))

    fig.update_layout(
        title="<b>Global SHAP Feature Importance</b> (Mean |SHAP Value| across population)",
        xaxis_title="Mean |SHAP Value|",
        yaxis_title="",
        template='plotly_white',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=340,
        margin=dict(l=20, r=30, t=45, b=20),
        font={'family': 'Inter, sans-serif', 'color': '#1F1F1F'}
    )
    fig.update_xaxes(gridcolor=COLOR_GRID, linecolor=COLOR_BORDER)
    fig.update_yaxes(linecolor=COLOR_BORDER)
    return fig


def create_feature_psi_chart(drift_report: Dict[str, Any]) -> go.Figure:
    """
    Creates a horizontal bar chart of Population Stability Index (PSI) values across all monitored features,
    with explicit prototype threshold lines at 0.10 (Watch) and 0.20 (Drift Detected).
    """
    feat_drift = drift_report.get("feature_drift", {})
    if not feat_drift:
        fig = go.Figure()
        fig.update_layout(title="No Feature Drift Data Available", template="plotly_white")
        return fig

    features = []
    psi_values = []
    colors = []
    statuses = []

    for f_name, f_data in feat_drift.items():
        disp_name = f_data.get("display_name", f_name)
        psi_val = f_data.get("psi", 0.0)
        status = f_data.get("status", "STABLE")

        features.append(disp_name)
        psi_values.append(psi_val)
        statuses.append(status)

        if psi_val < 0.10:
            colors.append(COLOR_HEALTHY)
        elif psi_val < 0.20:
            colors.append(COLOR_WATCH)
        else:
            colors.append(COLOR_CRITICAL)

    features = list(reversed(features))
    psi_values = list(reversed(psi_values))
    colors = list(reversed(colors))
    statuses = list(reversed(statuses))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=psi_values,
        y=features,
        orientation='h',
        marker=dict(color=colors, line=dict(color=COLOR_BORDER, width=1)),
        text=[f"{v:.4f}" for v in psi_values],
        textposition='outside',
        hovertext=[
            f"<b>{feat}</b><br>PSI: {psi:.4f}<br>Status: {st}<extra></extra>"
            for feat, psi, st in zip(features, psi_values, statuses)
        ],
        hoverinfo='text'
    ))

    max_x = max(max(psi_values, default=0.25) * 1.25, 0.28)
    fig.add_vline(
        x=0.10, line_width=1.5, line_dash="dash", line_color=COLOR_WATCH,
        annotation_text="Watch (0.10)", annotation_position="top right",
        annotation_font=dict(color=COLOR_WATCH, size=10)
    )
    fig.add_vline(
        x=0.20, line_width=1.5, line_dash="dash", line_color=COLOR_CRITICAL,
        annotation_text="Drift (0.20)", annotation_position="top right",
        annotation_font=dict(color=COLOR_CRITICAL, size=10)
    )

    fig.update_layout(
        title="<b>Feature Population Stability Index (PSI)</b> (Prototype Thresholds)",
        xaxis_title="PSI Metric Value",
        yaxis_title="",
        xaxis_range=[0, max_x],
        template='plotly_white',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=320,
        margin=dict(l=20, r=40, t=45, b=20),
        font={'family': 'Inter, sans-serif', 'color': '#1F1F1F'}
    )
    fig.update_xaxes(gridcolor=COLOR_GRID, linecolor=COLOR_BORDER)
    fig.update_yaxes(linecolor=COLOR_BORDER)
    return fig


def create_distribution_comparison_chart(
    feature_name: str,
    ref_data: Union[np.ndarray, List[Any], pd.Series],
    curr_data: Union[np.ndarray, List[Any], pd.Series],
    is_categorical: bool = False
) -> go.Figure:
    """
    Compares baseline reference distribution vs current streaming telemetry distribution.
    """
    fig = go.Figure()

    if is_categorical:
        ref_s = pd.Series(ref_data).value_counts(normalize=True)
        curr_s = pd.Series(curr_data).value_counts(normalize=True)
        all_cats = ["H", "M", "L"]

        ref_props = [ref_s.get(c, 0.0) * 100 for c in all_cats]
        curr_props = [curr_s.get(c, 0.0) * 100 for c in all_cats]

        fig.add_trace(go.Bar(
            name="Reference Baseline (N=8,000)",
            x=all_cats,
            y=ref_props,
            marker_color=COLOR_BASELINE,
            text=[f"{p:.1f}%" for p in ref_props],
            textposition="outside"
        ))
        fig.add_trace(go.Bar(
            name="Current Window",
            x=all_cats,
            y=curr_props,
            marker_color=COLOR_ANALYTICAL,
            text=[f"{p:.1f}%" for p in curr_props],
            textposition="outside"
        ))
        fig.update_layout(
            barmode="group",
            title=f"<b>Machine Type Distribution Comparison</b>",
            xaxis_title="Machine Quality Type",
            yaxis_title="Proportion (%)",
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=300,
            margin=dict(l=20, r=20, t=45, b=20),
            font={'family': 'Inter, sans-serif', 'color': '#1F1F1F'},
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
    else:
        ref_clean = np.asarray(ref_data, dtype=float)
        ref_clean = ref_clean[np.isfinite(ref_clean)]
        curr_clean = np.asarray(curr_data, dtype=float)
        curr_clean = curr_clean[np.isfinite(curr_clean)]

        fig.add_trace(go.Histogram(
            x=ref_clean,
            name="Reference Baseline",
            histnorm='probability density',
            marker_color='rgba(31, 31, 31, 0.4)',
            opacity=0.6,
            nbinsx=25
        ))
        fig.add_trace(go.Histogram(
            x=curr_clean,
            name="Current Window",
            histnorm='probability density',
            marker_color='rgba(93, 88, 122, 0.65)',
            opacity=0.65,
            nbinsx=25
        ))
        fig.update_layout(
            barmode="overlay",
            title=f"<b>{feature_name} Distribution Overlay</b>",
            xaxis_title=f"{feature_name}",
            yaxis_title="Probability Density",
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=300,
            margin=dict(l=20, r=20, t=45, b=20),
            font={'family': 'Inter, sans-serif', 'color': '#1F1F1F'},
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

    fig.update_xaxes(linecolor=COLOR_BORDER)
    fig.update_yaxes(gridcolor=COLOR_GRID, linecolor=COLOR_BORDER)
    return fig


def create_probability_drift_chart(
    ref_probs: Union[List[float], np.ndarray],
    curr_probs: Union[List[float], np.ndarray]
) -> go.Figure:
    """
    Creates density histogram comparing Reference Model Failure Probabilities vs Current Window.
    """
    ref_arr = np.asarray(ref_probs, dtype=float)
    curr_arr = np.asarray(curr_probs, dtype=float)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=ref_arr,
        name="Reference Baseline (N=8,000)",
        histnorm='probability density',
        marker_color='rgba(31, 31, 31, 0.4)',
        opacity=0.6,
        nbinsx=30
    ))
    fig.add_trace(go.Histogram(
        x=curr_arr,
        name="Current Window",
        histnorm='probability density',
        marker_color='rgba(139, 63, 63, 0.65)',
        opacity=0.65,
        nbinsx=30
    ))
    fig.add_vline(
        x=0.50, line_width=1.5, line_dash="dash", line_color=COLOR_CRITICAL,
        annotation_text="Decision Threshold (0.50)", annotation_position="top right",
        annotation_font=dict(color=COLOR_CRITICAL, size=10)
    )

    fig.update_layout(
        title="<b>Model Output Failure Probability Distribution</b>",
        xaxis_title="Predicted Failure Probability",
        yaxis_title="Density",
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=300,
        margin=dict(l=20, r=20, t=45, b=20),
        font={'family': 'Inter, sans-serif', 'color': '#1F1F1F'},
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(gridcolor=COLOR_GRID, linecolor=COLOR_BORDER)
    fig.update_yaxes(gridcolor=COLOR_GRID, linecolor=COLOR_BORDER)
    return fig


def create_risk_tier_comparison_chart(risk_tier_comparison: Dict[str, Any]) -> go.Figure:
    """
    Creates grouped bar chart comparing baseline reference vs current window risk tier percentages.
    """
    tiers = ["Low", "Moderate", "High", "Critical"]
    ref_pcts = []
    curr_pcts = []

    for t in tiers:
        t_data = risk_tier_comparison.get(t, {})
        ref_pcts.append(t_data.get("ref_proportion", 0.0) * 100)
        curr_pcts.append(t_data.get("curr_proportion", 0.0) * 100)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Reference Baseline",
        x=tiers,
        y=ref_pcts,
        marker_color=COLOR_BASELINE,
        text=[f"{p:.1f}%" for p in ref_pcts],
        textposition="outside"
    ))
    fig.add_trace(go.Bar(
        name="Current Window",
        x=tiers,
        y=curr_pcts,
        marker_color=COLOR_WARNING,
        text=[f"{p:.1f}%" for p in curr_pcts],
        textposition="outside"
    ))

    fig.update_layout(
        barmode="group",
        title="<b>Risk Tier Distribution Comparison</b>",
        xaxis_title="Prototype Risk Tier",
        yaxis_title="Proportion of Population (%)",
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=300,
        margin=dict(l=20, r=20, t=45, b=20),
        font={'family': 'Inter, sans-serif', 'color': '#1F1F1F'},
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(linecolor=COLOR_BORDER)
    fig.update_yaxes(gridcolor=COLOR_GRID, linecolor=COLOR_BORDER)
    return fig


def create_expected_cost_curve_chart(
    df_sweep: pd.DataFrame,
    operational_threshold: float = 0.50,
    analysis_threshold: Optional[float] = None,
    cost_unit: str = "Relative Cost Units"
) -> go.Figure:
    """
    Creates interactive Expected Cost vs Classification Threshold curve with operational
    and cost-minimizing markers.
    """
    fig = go.Figure()

    # Cost curve
    fig.add_trace(go.Scatter(
        x=df_sweep["Threshold"],
        y=df_sweep["Expected_Cost"],
        mode="lines",
        name="Expected Cost / Obs",
        line=dict(color=COLOR_BASELINE, width=2.5),
        hovertemplate="Threshold: %{x:.2f}<br>Exp Cost: %{y:.4f}<extra></extra>"
    ))

    # Operational Threshold Line (0.50)
    fig.add_vline(
        x=operational_threshold,
        line_width=2,
        line_dash="dash",
        line_color=COLOR_HEALTHY,
        annotation_text=f"Operational Baseline ({operational_threshold:.2f})",
        annotation_position="top left",
        annotation_font=dict(color=COLOR_HEALTHY, size=10)
    )

    # Cost-Minimizing Analysis Threshold Line
    if analysis_threshold is not None and abs(analysis_threshold - operational_threshold) >= 0.005:
        fig.add_vline(
            x=analysis_threshold,
            line_width=2,
            line_dash="dot",
            line_color=COLOR_WATCH,
            annotation_text=f"Analysis Threshold ({analysis_threshold:.2f})",
            annotation_position="top right",
            annotation_font=dict(color=COLOR_WATCH, size=10)
        )

    fig.update_layout(
        title="<b>Expected Cost per Observation vs Classification Threshold</b>",
        xaxis_title="Classification Probability Threshold",
        yaxis_title=f"Expected Cost ({cost_unit})",
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=340,
        margin=dict(l=20, r=20, t=50, b=20),
        font={'family': 'Inter, sans-serif', 'color': '#1F1F1F'},
        hovermode="x unified"
    )
    fig.update_xaxes(gridcolor=COLOR_GRID, linecolor=COLOR_BORDER)
    fig.update_yaxes(gridcolor=COLOR_GRID, linecolor=COLOR_BORDER)
    return fig


def create_threshold_tradeoff_curves_chart(
    df_sweep: pd.DataFrame,
    operational_threshold: float = 0.50,
    analysis_threshold: Optional[float] = None
) -> go.Figure:
    """
    Creates multi-line Plotly chart showing Precision, Recall, F1, and Specificity trade-offs across thresholds.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_sweep["Threshold"],
        y=df_sweep["Recall"],
        mode="lines",
        name="Recall (Sensitivity)",
        line=dict(color=COLOR_HEALTHY, width=2.5),
        hovertemplate="Recall: %{y:.2%}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=df_sweep["Threshold"],
        y=df_sweep["Precision"],
        mode="lines",
        name="Precision",
        line=dict(color=COLOR_INFO, width=2.5),
        hovertemplate="Precision: %{y:.2%}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=df_sweep["Threshold"],
        y=df_sweep["F1"],
        mode="lines",
        name="F1-Score",
        line=dict(color=COLOR_WATCH, width=2.5, dash="dash"),
        hovertemplate="F1: %{y:.2%}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=df_sweep["Threshold"],
        y=df_sweep["Specificity"],
        mode="lines",
        name="Specificity",
        line=dict(color=COLOR_ANALYTICAL, width=1.5, dash="dot"),
        hovertemplate="Specificity: %{y:.2%}<extra></extra>"
    ))

    # Operational Threshold Line
    fig.add_vline(
        x=operational_threshold,
        line_width=1.5,
        line_dash="dash",
        line_color=COLOR_BASELINE
    )

    if analysis_threshold is not None:
        fig.add_vline(
            x=analysis_threshold,
            line_width=1.5,
            line_dash="dot",
            line_color=COLOR_WATCH
        )

    fig.update_layout(
        title="<b>Classification Performance Trade-Off Curves</b>",
        xaxis_title="Classification Probability Threshold",
        yaxis_title="Metric Value (0.0 to 1.0)",
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=340,
        margin=dict(l=20, r=20, t=50, b=20),
        font={'family': 'Inter, sans-serif', 'color': '#1F1F1F'},
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    fig.update_xaxes(gridcolor=COLOR_GRID, linecolor=COLOR_BORDER)
    fig.update_yaxes(gridcolor=COLOR_GRID, linecolor=COLOR_BORDER)
    return fig


def create_confusion_matrix_comparison_chart(
    cm_a: Dict[str, int],
    cm_b: Dict[str, int],
    name_a: str = "Operational Baseline (0.50)",
    name_b: str = "Analysis Threshold"
) -> go.Figure:
    """
    Creates side-by-side grouped bar chart comparing confusion matrix outcomes (TP, TN, FP, FN).
    """
    categories = ["True Positives (TP)", "False Negatives (FN)", "False Positives (FP)", "True Negatives (TN)"]
    vals_a = [cm_a["TP"], cm_a["FN"], cm_a["FP"], cm_a["TN"]]
    vals_b = [cm_b["TP"], cm_b["FN"], cm_b["FP"], cm_b["TN"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=name_a,
        x=categories,
        y=vals_a,
        marker_color=COLOR_HEALTHY,
        text=vals_a,
        textposition="outside"
    ))
    fig.add_trace(go.Bar(
        name=name_b,
        x=categories,
        y=vals_b,
        marker_color=COLOR_WATCH,
        text=vals_b,
        textposition="outside"
    ))

    fig.update_layout(
        barmode="group",
        title="<b>Confusion Matrix Outcome Comparison (Holdout N=2,000)</b>",
        xaxis_title="Classification Outcome Category",
        yaxis_title="Count of Observations",
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=320,
        margin=dict(l=20, r=20, t=45, b=20),
        font={'family': 'Inter, sans-serif', 'color': '#1F1F1F'},
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(linecolor=COLOR_BORDER)
    fig.update_yaxes(gridcolor=COLOR_GRID, linecolor=COLOR_BORDER)
    return fig


def create_cost_ratio_sensitivity_heatmap(df_sensitivity: pd.DataFrame) -> go.Figure:
    """
    Creates heatmap showing Expected Cost across Cost Ratios (C_FN / C_FP) and Probability Thresholds.
    """
    ratio_labels = df_sensitivity["Cost_Ratio (C_FN/C_FP)"].tolist()
    threshold_cols = [c for c in df_sensitivity.columns if c != "Cost_Ratio (C_FN/C_FP)"]
    z_values = df_sensitivity[threshold_cols].values

    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=threshold_cols,
        y=ratio_labels,
        colorscale=[[0, "#EBF3ED"], [0.5, "#FAF5E8"], [1, "#8B3F3F"]],
        hovertemplate="Cost Ratio: %{y}<br>Threshold: %{x}<br>Exp Cost: %{z:.3f}<extra></extra>"
    ))

    fig.update_layout(
        title="<b>Cost Sensitivity Surface: C_FN / C_FP Ratio vs Decision Threshold</b>",
        xaxis_title="Probability Threshold",
        yaxis_title="Cost Asymmetry Ratio (C_FN / C_FP)",
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=320,
        margin=dict(l=20, r=20, t=45, b=20),
        font={'family': 'Inter, sans-serif', 'color': '#1F1F1F'}
    )
    fig.update_xaxes(gridcolor=COLOR_GRID, linecolor=COLOR_BORDER)
    fig.update_yaxes(linecolor=COLOR_BORDER)
    return fig
