"""
09 Drift Monitoring Page Component for Manufacturing Predictive Maintenance (Future Work F3).
Redesigned with clean light SaaS aesthetics and calm statistical visuals.
"""
from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from config import RAW_DATA_PATH, FEATURE_COLS, RANDOM_STATE
from src.data_loader import load_raw_data
from src.preprocessing import separate_features_and_target, split_data
from src.drift_monitor import (
    load_reference_profile,
    get_reference_predictions,
    evaluate_drift,
    DriftAlertTracker,
    STATUS_INSUFFICIENT_DATA,
    STATUS_STABLE,
    STATUS_WATCH,
    STATUS_DRIFT_DETECTED,
    FEATURE_METADATA
)
from components.drift_ui import (
    render_drift_kpis,
    render_feature_drift_table,
    render_data_quality_panel
)
from components.charts import (
    create_feature_psi_chart,
    create_distribution_comparison_chart,
    create_probability_drift_chart,
    create_risk_tier_comparison_chart
)


def render_drift_monitoring_page(model: Any, preprocessor: Any):
    """
    Renders the dedicated 09 Drift Monitoring dashboard page.
    """
    # 1. Initialize Drift Alert Tracker in Session State
    if "drift_tracker" not in st.session_state:
        st.session_state["drift_tracker"] = DriftAlertTracker()
    drift_tracker: DriftAlertTracker = st.session_state["drift_tracker"]

    # 2. Page Header & Research Integrity Notice
    st.markdown("""
    <div class="content-card">
        <div style="font-size: 1.25rem; font-weight: 700; color: #1F1F1F; margin-bottom: 4px;">
            📊 Telemetry & Model Drift Monitoring
        </div>
        <div style="font-size: 0.85rem; color: #6B6B63; line-height: 1.5;">
            Evaluates statistical divergence between the current streaming window and the fixed reference baseline (N=8,000 training partition).
        </div>
        <div class="disclaimer-box" style="margin-top: 12px; margin-bottom: 0px; font-size: 0.78rem;">
            <strong>MONITORING PRINCIPLE:</strong> Drift monitoring evaluates statistical divergence (PSI, KS-test) against the fixed reference baseline. It does not measure ground-truth concept drift without verified production outcome labels. Prototype thresholds (Watch: 0.10, Drift: 0.20) are heuristic indicators.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 3. Reference Profile Loading
    try:
        ref_profile = load_reference_profile()
    except Exception as e:
        st.error(f"Failed to load reference baseline profile: {e}")
        return

    # 4. Stream Source & Monitoring Window Selection
    st.markdown("<div style='font-size: 0.74rem; font-weight: 700; color: #6B6B63; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 6px;'>DRIFT MONITORING CONFIGURATION</div>", unsafe_allow_html=True)
    
    cfg_c1, cfg_c2, cfg_c3 = st.columns([1.5, 1.2, 1], gap="medium")

    with cfg_c1:
        source_options = [
            "Active Live Telemetry Stream (F1 Session History)",
            "Preset Benchmark Scenario Batches (Exploratory Evaluation)",
            "Upload Custom Batch Telemetry (CSV)"
        ]
        selected_source = st.selectbox(
            "TELEMETRY EVALUATION SOURCE",
            source_options,
            index=0,
            help="Select incoming telemetry dataset or rolling window to evaluate against the fixed baseline."
        )

    with cfg_c2:
        window_size_opt = st.selectbox(
            "EVALUATION WINDOW SIZE",
            [25, 50, 100, 200],
            index=2,
            help="Number of most recent events to include in statistical drift assessment (Min required: 50)."
        )

    with cfg_c3:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 RE-EVALUATE DRIFT", use_container_width=True):
            st.rerun()

    # 5. Extract Current Data Window based on selected source
    current_df = pd.DataFrame()

    if selected_source == "Active Live Telemetry Stream (F1 Session History)":
        if "telemetry_sim" in st.session_state and st.session_state["telemetry_sim"] is not None:
            sim = st.session_state["telemetry_sim"]
            current_df = sim.get_history_df(limit=window_size_opt)
        else:
            current_df = pd.DataFrame()

    elif selected_source == "Preset Benchmark Scenario Batches (Exploratory Evaluation)":
        preset_names = [
            "1. Nominal Baseline Fleet (Low Drift Expected)",
            "2. Spindle Overload Stress Fleet (High Torque Drift)",
            "3. Thermal Dissipation Shift Fleet (High Temperature Drift)",
            "4. Progressive Tool Wear Fleet (High Wear Drift)",
            "5. Quality Type Imbalance Fleet (100% Type L Variant)"
        ]
        chosen_preset = st.selectbox("SELECT BENCHMARK DRIFT SCENARIO", preset_names, index=0)
        
        raw_df = load_raw_data()
        X_raw, y_raw = separate_features_and_target(raw_df)
        _, X_test_ref, _, _ = split_data(X_raw, y_raw, test_size=0.2, random_state=RANDOM_STATE)
        
        sample_size = min(window_size_opt, len(X_test_ref))
        base_sample = X_test_ref.iloc[:sample_size].copy().reset_index(drop=True)

        if "Overload" in chosen_preset:
            base_sample["Torque [Nm]"] = np.clip(base_sample["Torque [Nm]"] * 1.45 + 10.0, 3.8, 76.6)
            base_sample["Rotational speed [rpm]"] = np.clip(base_sample["Rotational speed [rpm]"] * 0.85, 1168, 2886)
        elif "Thermal" in chosen_preset:
            base_sample["Air temperature [K]"] = np.clip(base_sample["Air temperature [K]"] + 4.5, 295.3, 304.5)
            base_sample["Process temperature [K]"] = np.clip(base_sample["Process temperature [K]"] + 3.0, 305.7, 313.8)
        elif "Tool Wear" in chosen_preset:
            base_sample["Tool wear [min]"] = np.clip(base_sample["Tool wear [min]"] + 120.0, 0, 253)
        elif "Type L" in chosen_preset:
            base_sample["Type"] = "L"

        current_df = base_sample

    elif selected_source == "Upload Custom Batch Telemetry (CSV)":
        up_file = st.file_uploader("Upload Telemetry CSV for Exploratory Drift Analysis", type=["csv"], key="drift_upload")
        if up_file is not None:
            try:
                current_df = pd.read_csv(up_file)
                st.success(f"Uploaded CSV with {len(current_df):,} records.")
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")

    # 6. Evaluate Drift Assessment
    drift_report = evaluate_drift(
        current_data=current_df,
        reference_profile=ref_profile,
        min_sample_size=50,
        model=model,
        preprocessor=preprocessor
    )

    # 7. Update Transition Alert Tracker
    overall_status = drift_report.get("overall_status", STATUS_INSUFFICIENT_DATA)
    new_alert = drift_tracker.update(overall_status, drift_report)

    # 8. Render Top KPI Metrics Cards
    render_drift_kpis(drift_report)

    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    # 9. Insufficient Data Warning or Full Analytics State
    if overall_status == STATUS_INSUFFICIENT_DATA:
        st.markdown(f"""
        <div class="empty-state-card">
            <div class="empty-state-icon">⏳</div>
            <div class="empty-state-title">INSUFFICIENT MONITORING WINDOW ACCUMULATED</div>
            <div class="empty-state-desc">
                Current window has <strong>{len(current_df)} events</strong>. A minimum of <strong>50 events</strong> is strictly required for statistically robust Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) drift estimation.<br><br>
                <em>Switch to <strong>08 Live Telemetry</strong> to start/step simulated cycles, or select a <strong>Preset Benchmark Scenario Batch</strong> above to evaluate exploratory fleets.</em>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Alert banner if elevated transition occurred
        if overall_status in [STATUS_WATCH, STATUS_DRIFT_DETECTED]:
            banner_cls = "result-banner-failure" if overall_status == STATUS_DRIFT_DETECTED else "result-banner-healthy"
            st.markdown(f"""
            <div class="{banner_cls}" style="margin-bottom: 16px;">
                <div style="font-size: 1.05rem; font-weight: 700;">⚠ DRIFT ALERT: {overall_status}</div>
                <div style="font-size: 0.84rem; opacity: 0.95; margin-top: 4px;">
                    Statistically significant divergence detected against reference training baseline. Peak driver: <strong>{drift_report.get('most_affected_feature')}</strong> (PSI: {drift_report.get('max_psi', 0.0):.4f}).
                </div>
            </div>
            """, unsafe_allow_html=True)

        # 10. Visual Analytics Section: Left (PSI Bar Chart) & Right (Feature Drift Table)
        v_col1, v_col2 = st.columns([1.1, 1.3], gap="medium")
        with v_col1:
            psi_fig = create_feature_psi_chart(drift_report)
            st.plotly_chart(psi_fig, use_container_width=True)

        with v_col2:
            st.markdown("<div style='font-size: 0.84rem; font-weight: 700; color: #1F1F1F; margin-bottom: 8px;'>FEATURE DRIFT METRICS SUMMARY</div>", unsafe_allow_html=True)
            render_feature_drift_table(drift_report)

        st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

        # 11. Feature Distribution Comparison Deep-Dive
        st.markdown("### Feature Distribution Deep-Dive (Baseline vs Current Window)")
        
        feat_keys = ["Type"] + [col for col in FEATURE_COLS if col != "Type"]
        sel_col1, sel_col2 = st.columns([1, 2.5], gap="medium")
        
        with sel_col1:
            selected_feat_col = st.selectbox(
                "SELECT SENSOR FEATURE TO INSPECT",
                feat_keys,
                index=0,
                format_func=lambda x: FEATURE_METADATA.get(x, {}).get("name", x)
            )
            feat_info = drift_report.get("feature_drift", {}).get(selected_feat_col, {})
            st.markdown(f"""
            <div style="background: #FFFFFF; border: 1px solid #E4E4DD; border-radius: 10px; padding: 14px 16px; font-size: 0.82rem; margin-top: 10px;">
                <strong style="color: #1F1F1F;">{FEATURE_METADATA.get(selected_feat_col, {}).get('name', selected_feat_col)}</strong><br>
                <span style="color: #6B6B63;">Type:</span> {feat_info.get('type', '').capitalize()}<br>
                <span style="color: #6B6B63;">PSI Score:</span> <strong style="color: #1F1F1F;">{feat_info.get('psi', 0.0):.4f}</strong><br>
                <span style="color: #6B6B63;">Drift Status:</span> <span class="badge-neutral">{feat_info.get('status', 'STABLE')}</span><br>
                <span style="color: #6B6B63;">KS Statistic:</span> {feat_info.get('ks_statistic', 'N/A')}<br>
                <span style="color: #6B6B63;">KS p-value:</span> {feat_info.get('ks_pvalue', 'N/A')}
            </div>
            """, unsafe_allow_html=True)

        with sel_col2:
            raw_df = load_raw_data()
            X_raw, y_raw = separate_features_and_target(raw_df)
            X_train_ref, _, _, _ = split_data(X_raw, y_raw, test_size=0.2, random_state=RANDOM_STATE)
            
            is_cat = (selected_feat_col == "Type")
            ref_data_vals = X_train_ref[selected_feat_col]
            curr_data_vals = current_df[selected_feat_col] if selected_feat_col in current_df.columns else pd.Series()

            comp_fig = create_distribution_comparison_chart(
                feature_name=FEATURE_METADATA.get(selected_feat_col, {}).get("name", selected_feat_col),
                ref_data=ref_data_vals,
                curr_data=curr_data_vals,
                is_categorical=is_cat
            )
            st.plotly_chart(comp_fig, use_container_width=True)

        st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

        # 12. Model Output Drift & Risk-Tier Distribution Shifts
        st.markdown("### Model-Output Probability Drift & Risk-Tier Shifts")
        st.markdown("<div style='font-size: 0.82rem; color: #6B6B63; margin-bottom: 12px;'>Evaluates whether the distribution of predicted failure probabilities or operational risk tiers has shifted from baseline.</div>", unsafe_allow_html=True)

        m_col1, m_col2 = st.columns(2, gap="medium")
        with m_col1:
            ref_preds = get_reference_predictions()
            if "failure_probability" in current_df.columns:
                curr_preds = current_df["failure_probability"].dropna().tolist()
            else:
                curr_trans = preprocessor.transform(current_df[FEATURE_COLS])
                curr_preds = model.predict_proba(curr_trans)[:, 1].tolist()

            prob_fig = create_probability_drift_chart(ref_preds, curr_preds)
            st.plotly_chart(prob_fig, use_container_width=True)

        with m_col2:
            risk_tier_data = drift_report.get("risk_tier_comparison", {})
            risk_fig = create_risk_tier_comparison_chart(risk_tier_data)
            st.plotly_chart(risk_fig, use_container_width=True)

        st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

        # 13. Data Quality & Drift Alert Transitions
        q_col, a_col = st.columns([1.2, 1], gap="medium")
        
        with q_col:
            st.markdown("<div style='font-size: 0.84rem; font-weight: 700; color: #1F1F1F; margin-bottom: 8px;'>TELEMETRY DATA QUALITY VALIDATION</div>", unsafe_allow_html=True)
            render_data_quality_panel(drift_report.get("data_quality", {}))

        with a_col:
            st.markdown("<div style='font-size: 0.84rem; font-weight: 700; color: #1F1F1F; margin-bottom: 8px;'>DRIFT ALERT TRANSITIONS LOG</div>", unsafe_allow_html=True)
            alerts_list = drift_tracker.get_alerts(limit=5)
            if alerts_list:
                for alt in alerts_list:
                    badge_cls = "badge-risk-crit" if alt.get("severity") == "CRITICAL" else ("badge-risk-mod" if alt.get("severity") == "WARNING" else "badge-risk-low")
                    st.markdown(f"""
                    <div style="background: #FFFFFF; border: 1px solid #E4E4DD; border-radius: 8px; padding: 10px 14px; margin-bottom: 6px; font-size: 0.78rem;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 2px;">
                            <span style="color: #6B6B63;">{alt.get('timestamp')}</span>
                            <span class="{badge_cls}">{alt.get('current_status')}</span>
                        </div>
                        <div style="color: #1F1F1F; margin-top: 4px;">{alt.get('message')}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background: #FFFFFF; border: 1px dashed #E4E4DD; border-radius: 8px; padding: 14px; text-align: center; color: #8E8E85; font-size: 0.78rem;">
                    No drift transition alerts recorded in session.
                </div>
                """, unsafe_allow_html=True)

    # 14. Academic & Research Integrity Disclaimers
    st.markdown("""
    <div class="content-card" style="margin-top: 24px;">
        <div style="font-size: 0.95rem; font-weight: 700; color: #1F1F1F; margin-bottom: 6px;">
            📚 Academic Research Integrity & Scope Boundaries
        </div>
        <div style="font-size: 0.82rem; color: #6B6B63; line-height: 1.6;">
            1. <strong>Data Drift vs Model Performance Drift:</strong> True model performance deterioration (e.g. drop in recall or precision) and concept drift require reliable, verified ground-truth maintenance labels over time.<br>
            2. <strong>Fixed Reference Baseline:</strong> The reference profile is permanently derived from the 8,000-observation training partition of the AI4I 2020 dataset.<br>
            3. <strong>Prototype Monitoring:</strong> Population Stability Index (PSI) threshold bands (&lt;0.10 Stable, 0.10–0.20 Watch, &ge;0.20 Drift) are prototype heuristic bands.
        </div>
    </div>
    """, unsafe_allow_html=True)
