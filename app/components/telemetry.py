"""
Telemetry Controls, Engineering Input Panels, and Parameter Snapshot Cards.
"""
import streamlit as st


def render_telemetry_inputs():
    """
    Renders the compact engineering input panel with unit labels, quick presets, and calculations.
    Returns the dictionary payload, decision threshold, and whether the user triggered prediction.
    """
    st.markdown("""
    <div class="telemetry-panel">
        <div class="panel-header">
            <span>🎛️</span> MACHINE TELEMETRY CONTROL PANEL
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Manage Preset State
    if "pred_type" not in st.session_state:
        st.session_state["pred_type"] = "M"
        st.session_state["pred_air"] = 298.2
        st.session_state["pred_proc"] = 308.7
        st.session_state["pred_speed"] = 1500.0
        st.session_state["pred_torque"] = 40.0
        st.session_state["pred_wear"] = 15.0

    st.markdown("<div style='font-size: 0.72rem; font-weight: 700; color: #6B6B63; text-transform: uppercase; margin-bottom: 8px; letter-spacing: 0.6px;'>ENGINEERING PRESET SCENARIOS:</div>", unsafe_allow_html=True)
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        if st.button("🟢 Nominal", use_container_width=True, help="Baseline normal milling operation"):
            st.session_state["pred_type"] = "M"
            st.session_state["pred_air"] = 298.2
            st.session_state["pred_proc"] = 308.7
            st.session_state["pred_speed"] = 1500.0
            st.session_state["pred_torque"] = 40.0
            st.session_state["pred_wear"] = 15.0
            st.rerun()
    with p2:
        if st.button("🟠 Thermal", use_container_width=True, help="Heat dissipation strain (ΔT < 8.6 K)"):
            st.session_state["pred_type"] = "L"
            st.session_state["pred_air"] = 302.5
            st.session_state["pred_proc"] = 310.2
            st.session_state["pred_speed"] = 1350.0
            st.session_state["pred_torque"] = 48.0
            st.session_state["pred_wear"] = 120.0
            st.rerun()
    with p3:
        if st.button("🔴 Overload", use_container_width=True, help="High torque & overstrain failure"):
            st.session_state["pred_type"] = "L"
            st.session_state["pred_air"] = 300.0
            st.session_state["pred_proc"] = 310.0
            st.session_state["pred_speed"] = 1250.0
            st.session_state["pred_torque"] = 68.0
            st.session_state["pred_wear"] = 215.0
            st.rerun()
    with p4:
        if st.button("⚠️ Tool Wear", use_container_width=True, help="Severe cutter wearout (TWF)"):
            st.session_state["pred_type"] = "H"
            st.session_state["pred_air"] = 299.0
            st.session_state["pred_proc"] = 309.5
            st.session_state["pred_speed"] = 1480.0
            st.session_state["pred_torque"] = 44.0
            st.session_state["pred_wear"] = 245.0
            st.rerun()

    st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)

    # Machine Type
    type_idx = ["L", "M", "H"].index(st.session_state["pred_type"])
    machine_type = st.selectbox(
        "MACHINE TYPE / VARIANT",
        options=["L", "M", "H"],
        index=type_idx,
        help="Product quality variant: L = Low (50%), M = Medium (30%), H = High (20%). Variant L has lower structural tolerance."
    )

    # Thermal Parameters Group
    st.markdown("""
    <div class="engineering-field-group">
        <div class="engineering-field-header">🌡️ THERMAL DYNAMICS (KELVIN)</div>
    </div>
    """, unsafe_allow_html=True)
    t1, t2 = st.columns(2)
    with t1:
        air_temp = st.number_input(
            "Air Temperature [K]",
            min_value=280.0,
            max_value=320.0,
            value=float(st.session_state["pred_air"]),
            step=0.1,
            help="Ambient environment temperature in Kelvin [Nominal: 295 - 304 K]"
        )
    with t2:
        process_temp = st.number_input(
            "Process Temperature [K]",
            min_value=290.0,
            max_value=330.0,
            value=float(st.session_state["pred_proc"]),
            step=0.1,
            help="Spindle cutting zone process temperature in Kelvin [Nominal: 305 - 314 K]"
        )

    temp_diff = process_temp - air_temp
    diff_color = "#477A55" if temp_diff >= 8.6 else "#8B3F3F"
    diff_status = "NORMAL DISSIPATION" if temp_diff >= 8.6 else "HEAT DISSIPATION RISK (ΔT < 8.6 K)"
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; font-size: 0.76rem; padding: 6px 12px; background: #F8F9F3; border-radius: 6px; border: 1px solid #E4E4DD; margin-bottom: 14px;">
        <span style="color: #6B6B63;">Thermal Gradient (ΔT): <b>{temp_diff:.2f} K</b></span>
        <span style="color: {diff_color}; font-weight: 700;">{diff_status}</span>
    </div>
    """, unsafe_allow_html=True)

    # Spindle Dynamics Group
    st.markdown("""
    <div class="engineering-field-group">
        <div class="engineering-field-header">⚙️ KINEMATICS & SPINDLE LOAD</div>
    </div>
    """, unsafe_allow_html=True)
    s1, s2 = st.columns(2)
    with s1:
        rot_speed = st.number_input(
            "Rotational Speed [RPM]",
            min_value=500.0,
            max_value=3500.0,
            value=float(st.session_state["pred_speed"]),
            step=10.0,
            help="Main spindle rotational speed in RPM [Nominal: 1200 - 2800 RPM]"
        )
    with s2:
        torque = st.number_input(
            "Torque [Nm]",
            min_value=0.0,
            max_value=120.0,
            value=float(st.session_state["pred_torque"]),
            step=0.5,
            help="Cutting resistance torque in Newton-meters [Nominal: 10 - 75 Nm]"
        )

    # Power Estimate (Watts)
    power_watts = (2 * 3.14159 * rot_speed * torque) / 60.0
    pwr_color = "#477A55" if (3500 <= power_watts <= 9000) else "#9A7A32"
    pwr_status = "WITHIN POWER ENVELOPE" if (3500 <= power_watts <= 9000) else "OUTSIDE POWER BOUNDS [3.5-9 kW]"
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; font-size: 0.76rem; padding: 6px 12px; background: #F8F9F3; border-radius: 6px; border: 1px solid #E4E4DD; margin-bottom: 14px;">
        <span style="color: #6B6B63;">Spindle Power (P = τ·ω): <b>{power_watts:.0f} W</b></span>
        <span style="color: {pwr_color}; font-weight: 700;">{pwr_status}</span>
    </div>
    """, unsafe_allow_html=True)

    # Cutter Degradation Group
    st.markdown("""
    <div class="engineering-field-group">
        <div class="engineering-field-header">⏱️ TOOL DEGRADATION & THRESHOLD</div>
    </div>
    """, unsafe_allow_html=True)
    tool_wear = st.number_input(
        "Tool Wear [min]",
        min_value=0.0,
        max_value=350.0,
        value=float(st.session_state["pred_wear"]),
        step=1.0,
        help="Cumulative milling cutter wear time in minutes [Failure boundary: 200 - 250 min]"
    )

    threshold = st.slider(
        "Decision Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Classification probability cutoff (Default: 0.50)"
    )

    st.markdown("<div style='margin-bottom: 14px;'></div>", unsafe_allow_html=True)
    run_predict = st.button("⚡ RUN FAILURE ANALYSIS", use_container_width=True, type="primary")

    payload = {
        "Type": machine_type,
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rot_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear
    }

    return payload, threshold, run_predict


def render_telemetry_snapshot(payload: dict):
    """Renders the compact post-prediction telemetry snapshot cards."""
    st.markdown("""
    <div style="font-size: 0.72rem; font-weight: 700; color: #6B6B63; text-transform: uppercase; letter-spacing: 0.6px; margin-top: 14px;">
        EVALUATED TELEMETRY PARAMETERS:
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="telemetry-chip-container">
        <div class="telemetry-chip">
            <div class="chip-label">TYPE</div>
            <div class="chip-value">{payload['Type']} <span class="chip-unit">Variant</span></div>
        </div>
        <div class="telemetry-chip">
            <div class="chip-label">AIR TEMP</div>
            <div class="chip-value">{payload['Air temperature [K]']:.1f} <span class="chip-unit">K</span></div>
        </div>
        <div class="telemetry-chip">
            <div class="chip-label">PROCESS TEMP</div>
            <div class="chip-value">{payload['Process temperature [K]']:.1f} <span class="chip-unit">K</span></div>
        </div>
        <div class="telemetry-chip">
            <div class="chip-label">RPM</div>
            <div class="chip-value">{payload['Rotational speed [rpm]']:.0f} <span class="chip-unit">RPM</span></div>
        </div>
        <div class="telemetry-chip">
            <div class="chip-label">TORQUE</div>
            <div class="chip-value">{payload['Torque [Nm]']:.1f} <span class="chip-unit">Nm</span></div>
        </div>
        <div class="telemetry-chip">
            <div class="chip-label">TOOL WEAR</div>
            <div class="chip-value">{payload['Tool wear [min]']:.0f} <span class="chip-unit">min</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
