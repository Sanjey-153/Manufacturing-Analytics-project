"""
Live Simulated Telemetry and Near-Real-Time Predictive Maintenance Dashboard Component (F1).
Redesigned with clean light SaaS aesthetics and white chart integration.
"""
import time
from datetime import datetime
from typing import Any, Optional
from pathlib import Path
import pandas as pd
import streamlit as st

from src.telemetry_simulator import (
    TelemetrySimulator,
    MODE_REPLAY,
    MODE_SCENARIO,
    AVAILABLE_MODES,
    AVAILABLE_SCENARIOS,
    SCENARIO_NOMINAL,
    SCENARIO_OVERLOAD,
    SCENARIO_THERMAL,
    SCENARIO_TOOL_WEAR,
    SCENARIO_RECOVERY
)
from components.risk import create_risk_gauge, render_maintenance_guidance
from components.charts import (
    create_live_risk_trend_chart,
    create_live_sensor_trend_charts
)
from src.explainability import explain_prediction
from components.explainability_ui import render_shap_explanation_card


def render_live_telemetry_page(model: Any, preprocessor: Any):
    """
    Renders the 08 Live Telemetry dashboard page with real-time streaming controls,
    telemetry dynamics, live probability gauges, risk alerts, and maintenance protocols.
    """
    # 1. Initialize Simulator in Session State
    if "telemetry_sim" not in st.session_state or st.session_state["telemetry_sim"] is None:
        st.session_state["telemetry_sim"] = TelemetrySimulator(
            mode=MODE_REPLAY,
            scenario=SCENARIO_NOMINAL,
            max_history=200,
            model=model,
            preprocessor=preprocessor
        )

    if "sim_running" not in st.session_state:
        st.session_state["sim_running"] = False
    if "sim_interval" not in st.session_state:
        st.session_state["sim_interval"] = 1.0
    if "sim_window" not in st.session_state:
        st.session_state["sim_window"] = 50
    if "last_export_msg" not in st.session_state:
        st.session_state["last_export_msg"] = None

    sim: TelemetrySimulator = st.session_state["telemetry_sim"]

    # 2. Page Header & Research Integrity Disclaimers
    st.markdown("""
    <div class="content-card">
        <div style="font-size: 1.25rem; font-weight: 700; color: #1F1F1F; margin-bottom: 4px;">
            📡 Live Machine Telemetry Stream
        </div>
        <div style="font-size: 0.85rem; color: #6B6B63; line-height: 1.5;">
            Continuous near-real-time sensor stream monitoring, posterior probability estimation, and threshold-based risk tier categorization.
        </div>
        <div class="disclaimer-box" style="margin-top: 12px; margin-bottom: 0px; font-size: 0.78rem;">
            <strong>SIMULATED TELEMETRY:</strong> Development prototype — not connected to physical SCADA/MES equipment. Do not interpret as real-time factory data.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 3. Control Center Toolbar
    st.markdown("""
    <div style="font-size: 0.74rem; font-weight: 700; color: #6B6B63; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 8px;">
        STREAM CONTROLS &amp; SIMULATION CONFIGURATION
    </div>
    """, unsafe_allow_html=True)

    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1.3, 1.3, 1.1, 1], gap="small")

    with ctrl_col1:
        mode_idx = 0 if sim.mode == MODE_REPLAY else 1
        selected_mode = st.selectbox(
            "SIMULATION MODE",
            [MODE_REPLAY, MODE_SCENARIO],
            index=mode_idx,
            help="Benchmark Dataset Replay streams rows from ai4i2020.csv; Controlled Scenario generates trajectory models."
        )
        if selected_mode != sim.mode:
            sim.set_mode(selected_mode)

    with ctrl_col2:
        if sim.mode in [MODE_SCENARIO, "Controlled Simulation"]:
            scen_idx = AVAILABLE_SCENARIOS.index(sim.scenario) if sim.scenario in AVAILABLE_SCENARIOS else 0
            selected_scen = st.selectbox(
                "CONTROLLED SCENARIO",
                AVAILABLE_SCENARIOS,
                index=scen_idx,
                help="Select simulated machine trajectory envelope within dataset-derived sanity ranges."
            )
            if selected_scen != sim.scenario:
                sim.set_scenario(selected_scen)
        else:
            st.selectbox(
                "DATASET REPLAY SOURCE",
                ["Benchmark Dataset: ai4i2020.csv (10k Rows)"],
                index=0,
                disabled=True
            )

    with ctrl_col3:
        interval_options = {"1 second": 1.0, "2 seconds": 2.0, "5 seconds": 5.0}
        curr_int_label = "1 second" if st.session_state["sim_interval"] == 1.0 else ("2 seconds" if st.session_state["sim_interval"] == 2.0 else "5 seconds")
        selected_int_label = st.selectbox(
            "UPDATE INTERVAL",
            list(interval_options.keys()),
            index=list(interval_options.keys()).index(curr_int_label)
        )
        st.session_state["sim_interval"] = interval_options[selected_int_label]

    with ctrl_col4:
        selected_window = st.selectbox(
            "HISTORY WINDOW",
            [25, 50, 100, 200],
            index=[25, 50, 100, 200].index(st.session_state["sim_window"]) if st.session_state["sim_window"] in [25, 50, 100, 200] else 1,
            help="Number of recent telemetry events to retain in rolling history and display on live charts."
        )
        st.session_state["sim_window"] = selected_window
        sim.set_max_history(selected_window)

    # Action Buttons: START, PAUSE, RESET, STEP, EXPORT CSV
    b_col1, b_col2, b_col3, b_col4, b_col5 = st.columns([1, 1, 1, 1, 1.4], gap="small")

    with b_col1:
        if st.button("▶ START", use_container_width=True, disabled=st.session_state["sim_running"], type="primary"):
            st.session_state["sim_running"] = True
            st.rerun()

    with b_col2:
        if st.button("⏸ PAUSE", use_container_width=True, disabled=not st.session_state["sim_running"]):
            st.session_state["sim_running"] = False
            st.rerun()

    with b_col3:
        if st.button("⏭ STEP", use_container_width=True, disabled=st.session_state["sim_running"]):
            sim.step(model=model, preprocessor=preprocessor, threshold=0.50)
            st.rerun()

    with b_col4:
        if st.button("🔄 RESET", use_container_width=True):
            st.session_state["sim_running"] = False
            st.session_state["last_export_msg"] = None
            sim.reset()
            st.rerun()

    with b_col5:
        if st.button("💾 EXPORT CSV", use_container_width=True, disabled=(sim.step_count == 0)):
            try:
                saved_path = sim.export_session_log()
                st.session_state["last_export_msg"] = f"outputs/telemetry/{saved_path.name}"
            except Exception as e:
                st.error(f"Export failed: {e}")

    # Display Export Confirmation if generated
    if st.session_state.get("last_export_msg"):
        exp_col1, exp_col2 = st.columns([2.2, 1])
        with exp_col1:
            st.success(f"✓ Session saved to disk: `{st.session_state['last_export_msg']}`")
        with exp_col2:
            export_df = sim.get_export_df()
            if not export_df.empty:
                csv_bytes = export_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ DOWNLOAD",
                    data=csv_bytes,
                    file_name=Path(st.session_state["last_export_msg"]).name,
                    mime="text/csv",
                    use_container_width=True
                )

    # 4. Telemetry Health Indicator
    stream_status = "RUNNING" if st.session_state["sim_running"] else "PAUSED"
    status_dot_cls = "status-dot-healthy" if st.session_state["sim_running"] else ""
    status_color = "#477A55" if st.session_state["sim_running"] else "#9A7A32"

    curr_event = sim.get_current_event()
    last_update_str = curr_event["timestamp"] if curr_event else "Not Started"

    history_len = len(sim.history)
    if history_len < 50:
        drift_badge_text = "INSUFFICIENT (<50)"
        drift_badge_col = "#6B6B63"
    else:
        if "drift_tracker" in st.session_state:
            drift_badge_text = st.session_state["drift_tracker"].current_status
        else:
            drift_badge_text = "STABLE"
        drift_badge_col = "#477A55" if drift_badge_text == "STABLE" else ("#9A7A32" if drift_badge_text == "WATCH" else "#8B3F3F")

    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; background: #FFFFFF; border: 1px solid #E4E4DD; border-radius: 10px; padding: 10px 18px; margin: 12px 0 16px 0; font-size: 0.8rem; flex-wrap: wrap; gap: 8px;">
        <div style="display: flex; align-items: center; gap: 10px;">
            <span style="color: #6B6B63; font-weight: 600;">STREAM:</span>
            <span style="color: {status_color}; font-weight: 700; display: flex; align-items: center; gap: 6px;">
                <span class="{status_dot_cls}"></span> {stream_status}
            </span>
        </div>
        <div>
            <span style="color: #6B6B63;">Cycles:</span> <strong style="color: #1F1F1F; font-family: 'JetBrains Mono', monospace;">{sim.step_count}</strong>
        </div>
        <div>
            <span style="color: #6B6B63;">Drift:</span> <strong style="color: {drift_badge_col}; font-family: 'JetBrains Mono', monospace;">{drift_badge_text}</strong>
        </div>
        <div>
            <span style="color: #6B6B63;">Last Timestamp:</span> <strong style="color: #1F1F1F; font-family: 'JetBrains Mono', monospace;">{last_update_str}</strong>
        </div>
        <div>
            <span class="badge-neutral" style="font-size: 0.72rem;">{sim.mode}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 5. Scenario Visualization Note
    if sim.mode in [MODE_SCENARIO, "Controlled Simulation"]:
        extra_note = ""
        if sim.scenario == SCENARIO_THERMAL:
            extra_note = " Thermal stress enforces ΔT = Process Temp − Air Temp < 8.6 K."
        st.markdown(f"""
        <div style="padding: 8px 14px; background: #F8F9F3; border-left: 3px solid #1F1F1F; border-radius: 6px; font-size: 0.78rem; color: #1F1F1F; margin-bottom: 16px;">
            <strong>ACTIVE SCENARIO:</strong> <span style="font-weight: 700; text-transform: uppercase;">{sim.scenario}</span>.{extra_note}<br>
            <span style="color: #6B6B63; font-size: 0.74rem;">Scenario behavior is simulated for demonstration and does not represent validated physical machine dynamics.</span>
        </div>
        """, unsafe_allow_html=True)

    # 6. Live Telemetry Dashboard State
    if curr_event is None:
        st.markdown("""
        <div class="empty-state-card">
            <div class="empty-state-icon">📡</div>
            <div class="empty-state-title">AWAITING TELEMETRY STREAM</div>
            <div class="empty-state-desc">
                Click <strong>▶ START</strong> to initiate continuous near-real-time streaming or <strong>⏭ STEP</strong> to advance one cycle.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        pred_class = curr_event["predicted_class"]
        risk_level = curr_event["risk_level"]
        fail_prob = curr_event["failure_probability"]

        # Central Status Panel
        banner_class = "result-banner-healthy" if pred_class == 0 else "result-banner-failure"
        status_text = "✓ Machine Healthy" if pred_class == 0 else "⚠ Failure Risk Detected"
        machine_status_label = "HEALTHY" if pred_class == 0 else "AT RISK"
        banner_sub = (
            "All operating parameters within baseline envelope." if pred_class == 0
            else "Sensor trajectory indicates heightened breakdown risk. Review diagnostic flags."
        )

        st.markdown(f"""
        <div class="{banner_class}" style="margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px;">
                <div>
                    <div class="result-title">{status_text}</div>
                    <div class="result-subtitle">{banner_sub}</div>
                </div>
                <div style="display: flex; gap: 24px; text-align: right; flex-wrap: wrap;">
                    <div>
                        <div style="font-size: 0.72rem; color: #6B6B63; font-weight: 600; text-transform: uppercase;">Probability</div>
                        <div style="font-size: 1.45rem; font-weight: 700;">{fail_prob*100:.2f}%</div>
                    </div>
                    <div>
                        <div style="font-size: 0.72rem; color: #6B6B63; font-weight: 600; text-transform: uppercase;">Risk Tier</div>
                        <div style="font-size: 1.45rem; font-weight: 700;">{risk_level.upper()}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.72rem; color: #6B6B63; font-weight: 600; text-transform: uppercase;">Timestamp</div>
                        <div style="font-size: 1.15rem; font-weight: 700; margin-top: 4px;">{curr_event['timestamp']}</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 7. Live KPI Cards
        st.markdown("<div style='font-size: 0.72rem; font-weight: 700; color: #6B6B63; text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 6px;'>CURRENT MACHINE OPERATING PARAMETERS</div>", unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Air Temp</div>
                <div class="kpi-value">{curr_event['Air temperature [K]']:.1f} <span style="font-size:0.8rem; color:#6B6B63;">K</span></div>
                <div class="kpi-subtitle">Ambient Intake</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            delta_t_disp = curr_event['Process temperature [K]'] - curr_event['Air temperature [K]']
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Process Temp</div>
                <div class="kpi-value kpi-watch">{curr_event['Process temperature [K]']:.1f} <span style="font-size:0.8rem; color:#6B6B63;">K</span></div>
                <div class="kpi-subtitle">ΔT: {delta_t_disp:.1f} K</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Spindle RPM</div>
                <div class="kpi-value">{curr_event['Rotational speed [rpm]']:.0f}</div>
                <div class="kpi-subtitle">Speed</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Torque</div>
                <div class="kpi-value kpi-critical">{curr_event['Torque [Nm]']:.1f} <span style="font-size:0.8rem; color:#6B6B63;">Nm</span></div>
                <div class="kpi-subtitle">Load</div>
            </div>
            """, unsafe_allow_html=True)
        with c5:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Tool Wear</div>
                <div class="kpi-value kpi-analytical">{curr_event['Tool wear [min]']:.0f} <span style="font-size:0.8rem; color:#6B6B63;">min</span></div>
                <div class="kpi-subtitle">Cutter Runtime</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

        # 8. Visual Analytics: Left (Charts) & Right (Gauge, Alerts, Guidance)
        chart_col, side_col = st.columns([1.6, 1.05], gap="medium")

        history_df = sim.get_history_df(limit=st.session_state["sim_window"])

        with chart_col:
            st.markdown("<div style='font-size: 0.84rem; font-weight: 700; color: #1F1F1F; margin-bottom: 8px;'>PROBABILITY & RISK TREND</div>", unsafe_allow_html=True)
            risk_trend_fig = create_live_risk_trend_chart(history_df, threshold=0.50)
            st.plotly_chart(risk_trend_fig, use_container_width=True)

            st.markdown("<div style='font-size: 0.84rem; font-weight: 700; color: #1F1F1F; margin-top: 14px; margin-bottom: 8px;'>LIVE PARAMETER TRENDS (ROLLING HISTORY)</div>", unsafe_allow_html=True)
            f_air, f_proc, f_spd, f_trq, f_wear = create_live_sensor_trend_charts(history_df)

            t1, t2 = st.columns(2)
            with t1:
                st.plotly_chart(f_air, use_container_width=True)
            with t2:
                st.plotly_chart(f_proc, use_container_width=True)

            s1, s2 = st.columns(2)
            with s1:
                st.plotly_chart(f_spd, use_container_width=True)
            with s2:
                st.plotly_chart(f_trq, use_container_width=True)

            st.plotly_chart(f_wear, use_container_width=True)

        with side_col:
            # Real-Time Risk Gauge
            gauge_fig = create_risk_gauge(fail_prob, threshold=0.50)
            st.plotly_chart(gauge_fig, use_container_width=True)

            # Prescriptive Maintenance Guidance
            render_maintenance_guidance(curr_event)

            # On-Demand SHAP Attribution for Current Live Event (F2)
            with st.expander("🔍 EXPLAIN CURRENT EVENT (SHAP ATTRIBUTION)", expanded=False):
                st.markdown("<div style='font-size: 0.78rem; color: #6B6B63; margin-bottom: 8px;'>Generate per-prediction SHAP feature attributions for this active telemetry state:</div>", unsafe_allow_html=True)
                if st.button("CALCULATE SHAP EXPLANATION", key="btn_explain_current_event", use_container_width=True):
                    try:
                        shap_live = explain_prediction(curr_event, model=model, preprocessor=preprocessor, threshold=0.50)
                        render_shap_explanation_card(shap_live, title="CURRENT TELEMETRY ATTRIBUTION", show_table=True, chart_key="live_telemetry_shap_chart")
                    except Exception as e:
                        st.error(f"Explanation failed: {e}")

            # Elevated Risk Alerts & Transitions Panel
            elevated_alerts = sim.get_alerts()
            st.markdown("""
            <div style="font-size: 0.84rem; font-weight: 700; color: #1F1F1F; margin-top: 16px; margin-bottom: 8px; display: flex; align-items: center; gap: 6px;">
                <span>🚨</span> ELEVATED-RISK ALERTS (HIGH / CRITICAL)
            </div>
            """, unsafe_allow_html=True)

            if elevated_alerts:
                for alt in elevated_alerts[:5]:
                    risk_t = alt['current_risk']
                    badge_cls = "badge-risk-crit" if risk_t == "Critical" else ("badge-risk-high" if risk_t == "High" else "badge-risk-mod")
                    st.markdown(f"""
                    <div style="background: #FFFFFF; border: 1px solid #E4E4DD; border-radius: 8px; padding: 10px 12px; margin-bottom: 8px; font-size: 0.78rem;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                            <span style="color: #6B6B63;">Step {alt['step']} ({alt['timestamp']})</span>
                            <span class="{badge_cls}" style="font-size: 0.68rem; padding: 2px 8px;">{alt['previous_risk']} ➔ {alt['current_risk']}</span>
                        </div>
                        <div style="color: #1F1F1F; font-weight: 600;">Failure Probability: {alt['probability_pct']}</div>
                        <div style="color: #6B6B63; font-size: 0.72rem; margin-top: 2px;">{alt['telemetry_summary']}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background: #FFFFFF; border: 1px dashed #E4E4DD; border-radius: 8px; padding: 14px; text-align: center; color: #8E8E85; font-size: 0.78rem;">
                    No elevated risk alerts triggered yet.
                </div>
                """, unsafe_allow_html=True)

        # 9. Recent Telemetry Events Table
        st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)
        st.markdown(f"### Recent Telemetry Event Log (Last {len(history_df)} Cycles)")

        table_df = history_df.sort_values("step", ascending=False).copy()
        table_df = table_df.rename(columns={
            "timestamp": "Timestamp",
            "Type": "Type",
            "Air temperature [K]": "Air Temperature",
            "Process temperature [K]": "Process Temperature",
            "Rotational speed [rpm]": "RPM",
            "Torque [Nm]": "Torque",
            "Tool wear [min]": "Tool Wear",
            "failure_probability": "Failure Probability",
            "predicted_class": "Predicted Class",
            "risk_level": "Risk Tier"
        })

        table_cols = [
            "Timestamp", "Type", "Air Temperature", "Process Temperature",
            "RPM", "Torque", "Tool Wear", "Failure Probability",
            "Predicted Class", "Risk Tier"
        ]
        st.dataframe(
            table_df[table_cols],
            use_container_width=True,
            height=260
        )

    # 10. Real-Time Streaming Rerun Loop
    if st.session_state["sim_running"]:
        time.sleep(st.session_state["sim_interval"])
        sim.step(model=model, preprocessor=preprocessor, threshold=0.50)
        st.rerun()
