"""
Simulated Near-Real-Time Telemetry Engine for Predictive Maintenance Intelligence (F1).
Provides Benchmark Dataset Replay and Controlled Scenario Simulation modes.
All simulated telemetry streams are processed through the validated inference pipeline.
"""
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    RAW_DATA_PATH,
    FEATURE_COLS,
    OUTPUTS_DIR
)
from src.prediction import (
    predict_single,
    validate_single_input,
    load_inference_pipeline,
    SANITY_RANGES
)
from src.recommendations import determine_risk_level

# Simulation Modes
MODE_REPLAY = "Benchmark Dataset Replay"
MODE_SCENARIO = "Controlled Scenario"
AVAILABLE_MODES = [MODE_REPLAY, MODE_SCENARIO, "Controlled Simulation"]

# Controlled Scenarios
SCENARIO_NOMINAL = "Nominal Operation"
SCENARIO_THERMAL = "Thermal Stress"
SCENARIO_OVERLOAD = "Spindle Overload"
SCENARIO_TOOL_WEAR = "Progressive Tool Wear"
SCENARIO_RECOVERY = "Recovery / Normalization"
AVAILABLE_SCENARIOS = [
    SCENARIO_NOMINAL,
    SCENARIO_THERMAL,
    SCENARIO_OVERLOAD,
    SCENARIO_TOOL_WEAR,
    SCENARIO_RECOVERY
]


class TelemetrySimulator:
    """
    Simulates industrial machine sensor telemetry in near-real-time.
    Supports benchmark dataset replay and trajectory-based controlled scenarios.
    """

    def __init__(
        self,
        mode: str = MODE_REPLAY,
        scenario: str = SCENARIO_NOMINAL,
        replay_start_idx: int = 0,
        max_history: int = 200,
        model: Any = None,
        preprocessor: Any = None
    ):
        self.mode = mode if mode in AVAILABLE_MODES else MODE_REPLAY
        if self.mode == "Controlled Simulation":
            self.mode = MODE_SCENARIO
        self.scenario = scenario if scenario in AVAILABLE_SCENARIOS else SCENARIO_NOMINAL
        self.replay_idx = replay_start_idx
        self.max_history = max_history
        self.step_count = 0
        self.is_running = False
        
        # Load or store model artifacts
        if model is None or preprocessor is None:
            try:
                self.model, self.preprocessor = load_inference_pipeline()
            except Exception:
                self.model, self.preprocessor = None, None
        else:
            self.model = model
            self.preprocessor = preprocessor

        # Load raw benchmark dataset for replay
        self.replay_data: Optional[pd.DataFrame] = None
        if RAW_DATA_PATH.exists():
            raw_df = pd.read_csv(RAW_DATA_PATH)
            self.replay_data = raw_df[FEATURE_COLS].copy()
        
        # Scenario state variables for controlled progression
        self._scenario_step = 0
        self._curr_type = "M"
        self._curr_air_temp = 298.2
        self._curr_proc_temp = 308.7
        self._curr_speed = 1500.0
        self._curr_torque = 40.0
        self._curr_wear = 20.0
        self._curr_thermal_delta = 7.8

        # State tracking and event history
        self.history: List[Dict[str, Any]] = []
        self.alerts: List[Dict[str, Any]] = []          # Elevated Risk Alerts (entry into High/Critical)
        self.risk_transitions: List[Dict[str, Any]] = [] # General State Transitions
        self._last_risk_level: Optional[str] = None
        self._last_status: Optional[str] = None

    def set_mode(self, mode: str):
        """Switches simulation mode between Replay and Controlled Scenario."""
        if mode in [MODE_SCENARIO, "Controlled Simulation"]:
            self.mode = MODE_SCENARIO
        elif mode in AVAILABLE_MODES:
            self.mode = mode

    def set_scenario(self, scenario: str):
        """Sets the active controlled simulation trajectory."""
        if scenario in AVAILABLE_SCENARIOS:
            self.scenario = scenario
            self._scenario_step = 0
            self._init_scenario_state(scenario)

    def set_max_history(self, max_history: int):
        """Updates the maximum stored rolling history limit and trims existing history if needed."""
        self.max_history = max_history
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def _init_scenario_state(self, scenario: str):
        """Initializes internal trajectory coordinates for the selected scenario."""
        if scenario == SCENARIO_NOMINAL:
            self._curr_type = "M"
            self._curr_air_temp = 298.2
            self._curr_proc_temp = 308.7
            self._curr_speed = 1500.0
            self._curr_torque = 40.0
            self._curr_wear = 20.0
        elif scenario == SCENARIO_THERMAL:
            self._curr_type = "L"
            self._curr_air_temp = 301.0
            self._curr_thermal_delta = 8.0
            self._curr_proc_temp = 309.0
            self._curr_speed = 1370.0
            self._curr_torque = 46.0
            self._curr_wear = 90.0
        elif scenario == SCENARIO_OVERLOAD:
            self._curr_type = "L"
            self._curr_air_temp = 300.0
            self._curr_proc_temp = 309.5
            self._curr_speed = 1350.0
            self._curr_torque = 52.0
            self._curr_wear = 160.0
        elif scenario == SCENARIO_TOOL_WEAR:
            self._curr_type = "H"
            self._curr_air_temp = 298.5
            self._curr_proc_temp = 308.9
            self._curr_speed = 1450.0
            self._curr_torque = 42.0
            self._curr_wear = 180.0
        elif scenario == SCENARIO_RECOVERY:
            self._curr_type = "M"
            self._curr_air_temp = 301.0
            self._curr_proc_temp = 310.5
            self._curr_speed = 1300.0
            self._curr_torque = 60.0
            self._curr_wear = 150.0

    def reset(self, start_idx: int = 0):
        """Resets the simulator step count, trajectory state, history, and alerts."""
        self.replay_idx = start_idx
        self.step_count = 0
        self._scenario_step = 0
        self._init_scenario_state(self.scenario)
        self.history = []
        self.alerts = []
        self.risk_transitions = []
        self._last_risk_level = None
        self._last_status = None
        self.is_running = False

    def _generate_next_raw_telemetry(self) -> Dict[str, Any]:
        """
        Produces the next raw telemetry dictionary using the active mode.
        Ensures all values conform strictly to the 6 validated feature schema.
        """
        if self.mode == MODE_REPLAY and self.replay_data is not None:
            if len(self.replay_data) == 0:
                raise ValueError("Replay dataset is empty.")
            
            row = self.replay_data.iloc[self.replay_idx % len(self.replay_data)].to_dict()
            self.replay_idx += 1
            return {
                "Type": str(row["Type"]),
                "Air temperature [K]": float(row["Air temperature [K]"]),
                "Process temperature [K]": float(row["Process temperature [K]"]),
                "Rotational speed [rpm]": float(row["Rotational speed [rpm]"]),
                "Torque [Nm]": float(row["Torque [Nm]"]),
                "Tool wear [min]": float(row["Tool wear [min]"])
            }

        # MODE_SCENARIO: Controlled Trajectory Progression
        self._scenario_step += 1
        rng = np.random.default_rng()

        if self.scenario == SCENARIO_NOMINAL:
            # Steady nominal operation with mild random jitter
            air_temp = float(np.clip(298.2 + rng.normal(0, 0.15), 295.0, 302.0))
            proc_temp = float(np.clip(air_temp + 10.5 + rng.normal(0, 0.1), 305.0, 313.0))
            speed = float(np.clip(1500.0 + rng.normal(0, 20.0), 1400.0, 1650.0))
            torque = float(np.clip(40.0 + rng.normal(0, 1.5), 32.0, 48.0))
            wear = float(np.clip(self._curr_wear + 0.5, 0.0, 100.0))
            self._curr_wear = wear
            machine_type = "M"

        elif self.scenario == SCENARIO_THERMAL:
            # Rising ambient temperature with controlled thermal gradient strictly ΔT < 8.6 K
            self._curr_air_temp = min(304.2, self._curr_air_temp + 0.3)
            # Thermal delta is constructed directly to guarantee ΔT < 8.6 K across all steps
            # Base delta decays from 8.2 down to 7.0 K with tiny bounded noise
            target_delta = max(7.0, 8.2 - 0.06 * self._scenario_step)
            delta_noise = float(rng.uniform(-0.15, 0.15))
            thermal_delta = float(np.clip(target_delta + delta_noise, 6.5, 8.35))
            
            air_temp = float(self._curr_air_temp + rng.uniform(-0.05, 0.05))
            proc_temp = float(air_temp + thermal_delta)  # Guarantees proc_temp - air_temp = thermal_delta <= 8.35 < 8.6 K

            self._curr_speed = max(1240.0, self._curr_speed - 12.0)
            self._curr_torque = min(58.0, self._curr_torque + 1.1)
            self._curr_wear = min(200.0, self._curr_wear + 1.5)

            speed = float(self._curr_speed + rng.normal(0, 10.0))
            torque = float(self._curr_torque + rng.normal(0, 0.8))
            wear = float(self._curr_wear)
            machine_type = "L"

        elif self.scenario == SCENARIO_OVERLOAD:
            # Surging mechanical torque and overstrain failure
            self._curr_torque = min(74.5, self._curr_torque + 2.2)
            self._curr_speed = max(1180.0, self._curr_speed - 20.0)
            self._curr_wear = min(240.0, self._curr_wear + 2.5)
            self._curr_air_temp = min(302.0, self._curr_air_temp + 0.15)
            self._curr_proc_temp = min(311.5, self._curr_proc_temp + 0.2)

            air_temp = float(self._curr_air_temp + rng.normal(0, 0.1))
            proc_temp = float(self._curr_proc_temp + rng.normal(0, 0.1))
            speed = float(self._curr_speed + rng.normal(0, 12.0))
            torque = float(self._curr_torque + rng.normal(0, 1.0))
            wear = float(self._curr_wear)
            machine_type = "L"

        elif self.scenario == SCENARIO_TOOL_WEAR:
            # Steady cutter wear accumulation towards threshold (200-250 min)
            self._curr_wear = min(252.0, self._curr_wear + 4.5)
            self._curr_torque = min(62.0, self._curr_torque + 1.0)
            self._curr_speed = max(1350.0, self._curr_speed - 8.0)

            air_temp = float(self._curr_air_temp + rng.normal(0, 0.1))
            proc_temp = float(self._curr_proc_temp + rng.normal(0, 0.1))
            speed = float(self._curr_speed + rng.normal(0, 15.0))
            torque = float(self._curr_torque + rng.normal(0, 0.8))
            wear = float(self._curr_wear)
            machine_type = "H"

        elif self.scenario == SCENARIO_RECOVERY:
            # Post-maintenance recovery back to safe envelope
            self._curr_torque = max(38.0, self._curr_torque - 3.5)
            self._curr_speed = min(1550.0, self._curr_speed + 35.0)
            self._curr_air_temp = max(298.0, self._curr_air_temp - 0.4)
            self._curr_proc_temp = max(308.5, self._curr_proc_temp - 0.3)
            self._curr_wear = max(10.0, self._curr_wear - 20.0)

            air_temp = float(self._curr_air_temp + rng.normal(0, 0.1))
            proc_temp = float(self._curr_proc_temp + rng.normal(0, 0.1))
            speed = float(self._curr_speed + rng.normal(0, 15.0))
            torque = float(self._curr_torque + rng.normal(0, 0.8))
            wear = float(self._curr_wear)
            machine_type = "M"

        else:
            air_temp = 298.2
            proc_temp = 308.7
            speed = 1500.0
            torque = 40.0
            wear = 15.0
            machine_type = "M"

        return {
            "Type": machine_type,
            "Air temperature [K]": round(air_temp, 2),
            "Process temperature [K]": round(proc_temp, 2),
            "Rotational speed [rpm]": round(speed, 1),
            "Torque [Nm]": round(torque, 2),
            "Tool wear [min]": round(wear, 1)
        }

    def step(
        self,
        model: Any = None,
        preprocessor: Any = None,
        threshold: float = 0.50
    ) -> Dict[str, Any]:
        """
        Advances the simulation by 1 telemetry cycle:
        1. Generates telemetry payload
        2. Validates features
        3. Executes inference via LightGBM pipeline
        4. Evaluates state transitions and elevated-risk alerts
        5. Returns structured event record
        """
        active_model = model or self.model
        active_preprocessor = preprocessor or self.preprocessor
        
        if active_model is None or active_preprocessor is None:
            active_model, active_preprocessor = load_inference_pipeline()
            self.model = active_model
            self.preprocessor = active_preprocessor

        # 1. Produce and validate telemetry input
        raw_telemetry = self._generate_next_raw_telemetry()
        validated_input = validate_single_input(raw_telemetry)

        # 2. Execute inference through existing prediction engine
        pred_res = predict_single(
            validated_input,
            model=active_model,
            preprocessor=active_preprocessor,
            threshold=threshold
        )

        self.step_count += 1
        now_ts = datetime.now()
        ts_str = now_ts.strftime("%H:%M:%S")
        iso_str = now_ts.isoformat()

        # 3. Create consolidated event record
        event = {
            "step": self.step_count,
            "timestamp": ts_str,
            "iso_timestamp": iso_str,
            "mode": self.mode,
            "scenario": self.scenario if self.mode in [MODE_SCENARIO, "Controlled Simulation"] else "N/A",
            "Type": validated_input["Type"],
            "Air temperature [K]": validated_input["Air temperature [K]"],
            "Process temperature [K]": validated_input["Process temperature [K]"],
            "Rotational speed [rpm]": validated_input["Rotational speed [rpm]"],
            "Torque [Nm]": validated_input["Torque [Nm]"],
            "Tool wear [min]": validated_input["Tool wear [min]"],
            "failure_probability": pred_res["failure_probability"],
            "failure_probability_pct": pred_res["failure_probability_percentage"],
            "predicted_class": pred_res["predicted_class"],
            "status_label": pred_res["status_label"],
            "risk_level": pred_res["risk_level"],
            "decision_threshold": threshold,
            "diagnostic_flags": pred_res["diagnostic_flags"],
            "summary": pred_res["summary"],
            "recommended_actions": pred_res["recommended_actions"]
        }

        # 4. State-Change and Elevated Risk Alert Processing (Audit Item 3)
        curr_risk = pred_res["risk_level"]
        prev_risk = self._last_risk_level

        if prev_risk is not None and curr_risk != prev_risk:
            # Record general Risk Transition
            transition_entry = {
                "step": self.step_count,
                "timestamp": ts_str,
                "previous_risk": prev_risk,
                "current_risk": curr_risk,
                "type": "Risk Transition",
                "transition": f"{prev_risk.upper()} → {curr_risk.upper()}",
                "failure_probability": pred_res["failure_probability"],
                "probability_pct": pred_res["failure_probability_percentage"],
                "status_label": pred_res["status_label"]
            }
            self.risk_transitions.insert(0, transition_entry)
            if len(self.risk_transitions) > 50:
                self.risk_transitions = self.risk_transitions[:50]

            # Trigger Elevated Risk Alert ONLY when transitioning INTO High or Critical
            # Examples: MODERATE -> HIGH, LOW -> HIGH, MODERATE -> CRITICAL, LOW -> CRITICAL, HIGH -> CRITICAL
            # Do NOT alert on: LOW -> MODERATE, CRITICAL -> LOW, HIGH -> HIGH, CRITICAL -> CRITICAL
            is_elevated_entry = (
                (curr_risk in ["High", "Critical"] and prev_risk not in ["High", "Critical"]) or
                (prev_risk == "High" and curr_risk == "Critical")
            )

            if is_elevated_entry:
                alert_entry = {
                    "step": self.step_count,
                    "timestamp": ts_str,
                    "previous_risk": prev_risk,
                    "current_risk": curr_risk,
                    "alert_type": "Elevated Risk Alert",
                    "transition": f"{prev_risk.upper()} → {curr_risk.upper()}",
                    "failure_probability": pred_res["failure_probability"],
                    "probability_pct": pred_res["failure_probability_percentage"],
                    "status_label": pred_res["status_label"],
                    "diagnostic_flags": pred_res["diagnostic_flags"],
                    "telemetry_summary": f"Speed: {validated_input['Rotational speed [rpm]']:.0f} RPM, Torque: {validated_input['Torque [Nm]']:.1f} Nm, Wear: {validated_input['Tool wear [min]']:.0f} min, ΔT: {validated_input['Process temperature [K]'] - validated_input['Air temperature [K]']:.2f} K"
                }
                self.alerts.insert(0, alert_entry)
                if len(self.alerts) > 50:
                    self.alerts = self.alerts[:50]

        self._last_risk_level = curr_risk
        self._last_status = pred_res["status_label"]

        # 5. Append to rolling history (strictly enforce max_history limit)
        self.history.append(event)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

        return event

    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Returns the recent telemetry history as a list of event dictionaries."""
        if limit is not None and limit > 0:
            return self.history[-limit:]
        return self.history.copy()

    def get_history_df(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Returns the recent telemetry history as a structured pandas DataFrame."""
        if not self.history:
            return pd.DataFrame(columns=[
                "step", "timestamp", "Type", "Air temperature [K]", "Process temperature [K]",
                "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
                "failure_probability", "predicted_class", "status_label", "risk_level"
            ])
        
        df = pd.DataFrame(self.history)
        if limit is not None and limit > 0:
            df = df.iloc[-limit:]
        return df

    def get_alerts(self) -> List[Dict[str, Any]]:
        """Returns recorded elevated-risk alerts (entry into High or Critical)."""
        return self.alerts

    def get_risk_transitions(self) -> List[Dict[str, Any]]:
        """Returns recorded general risk transitions."""
        return self.risk_transitions

    def get_current_event(self) -> Optional[Dict[str, Any]]:
        """Returns the most recent telemetry event or None."""
        return self.history[-1] if self.history else None

    def get_export_df(self) -> pd.DataFrame:
        """
        Returns the demonstration session history formatted with Section 24 columns:
        Timestamp, Simulation Mode, Scenario, Type, Air Temperature, Process Temperature,
        RPM, Torque, Tool Wear, Failure Probability, Predicted Class, Risk Tier.
        """
        if not self.history:
            return pd.DataFrame(columns=[
                "Timestamp", "Simulation Mode", "Scenario", "Type",
                "Air Temperature", "Process Temperature", "RPM", "Torque",
                "Tool Wear", "Failure Probability", "Predicted Class", "Risk Tier"
            ])
        
        df = pd.DataFrame(self.history)
        export_df = pd.DataFrame()
        export_df["Timestamp"] = df["timestamp"]
        export_df["Simulation Mode"] = df["mode"]
        export_df["Scenario"] = df["scenario"]
        export_df["Type"] = df["Type"]
        export_df["Air Temperature"] = df["Air temperature [K]"]
        export_df["Process Temperature"] = df["Process temperature [K]"]
        export_df["RPM"] = df["Rotational speed [rpm]"]
        export_df["Torque"] = df["Torque [Nm]"]
        export_df["Tool Wear"] = df["Tool wear [min]"]
        export_df["Failure Probability"] = df["failure_probability"]
        export_df["Predicted Class"] = df["predicted_class"]
        export_df["Risk Tier"] = df["risk_level"]
        return export_df

    def export_session_log(self, directory: Optional[Path] = None) -> Path:
        """
        Exports the current demonstration session history to a timestamped CSV file
        in outputs/telemetry/telemetry_session_<timestamp>.csv.
        Automatically creates output directory, guarantees collision-safe filenames,
        and handles empty/active sessions safely.
        """
        out_dir = directory or (OUTPUTS_DIR / "telemetry")
        out_dir.mkdir(parents=True, exist_ok=True)

        ts_file = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
        export_path = out_dir / f"telemetry_session_{ts_file}.csv"

        df_out = self.get_export_df()
        df_out.to_csv(export_path, index=False)
        return export_path


def create_simulator(**kwargs) -> TelemetrySimulator:
    """Factory helper to instantiate a TelemetrySimulator."""
    return TelemetrySimulator(**kwargs)


def get_available_modes() -> List[str]:
    """Returns list of supported simulation modes."""
    return AVAILABLE_MODES.copy()


def get_available_scenarios() -> List[str]:
    """Returns list of supported controlled scenarios."""
    return AVAILABLE_SCENARIOS.copy()
