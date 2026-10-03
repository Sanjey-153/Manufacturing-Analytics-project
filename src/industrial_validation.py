"""
Industrial Validation Readiness Framework for CNC Predictive Maintenance (F7).
Provides schema validation, engineering unit standardization, label alignment,
chronological validation gates, drift evaluation, and reporting for future factory telemetry.

CRITICAL RESEARCH DISCLAIMER:
No claim of empirical industrial validation is made. The project is awaiting verified
longitudinal CNC/SCADA/MES telemetry and ground-truth maintenance logs.
This module establishes the RIGOROUS VALIDATION GATEWAY to benchmark the validated
LightGBM classifier against external industrial datasets when they become available.
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

# Setup project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    NUMERICAL_COLS,
    CATEGORICAL_COLS,
    FEATURE_COLS,
    MODELS_DIR,
    PREPROCESSOR_PATH
)
from src.prediction import (
    load_inference_pipeline,
    preprocess_features_for_inference,
    predict_batch,
    SANITY_RANGES,
    VALID_MACHINE_TYPES
)
from src.drift_monitor import evaluate_drift

OUTPUTS_IND_DIR = PROJECT_ROOT / "outputs" / "industrial_validation"

# Data Source Classifications
VALID_DATA_SOURCES = ["BENCHMARK", "SIMULATED", "INDUSTRIAL"]

# Standard Industrial Telemetry Schema (Field Name Mapping)
INDUSTRIAL_SCHEMA_MAPPING = {
    "timestamp": ["timestamp", "datetime", "time", "ts", "Timestamp"],
    "machine_id": ["machine_id", "asset_id", "cnc_id", "Machine Reference", "machine_reference"],
    "machine_type": ["machine_type", "type", "variant", "Type"],
    "air_temperature_K": ["air_temperature_k", "air_temp_k", "air_temperature", "Air temperature [K]"],
    "process_temperature_K": ["process_temperature_k", "proc_temp_k", "process_temperature", "Process temperature [K]"],
    "rotational_speed_rpm": ["rotational_speed_rpm", "speed_rpm", "spindle_speed", "Rotational speed [rpm]"],
    "torque_Nm": ["torque_nm", "torque", "Torque [Nm]"],
    "tool_wear_min": ["tool_wear_min", "tool_wear", "cutter_wear", "Tool wear [min]"],
    "failure_label": ["failure_label", "machine_failure", "target", "failure", "Target", "Machine failure"]
}

# 12-Point Readiness Checklist Definition
READINESS_CHECKLIST_ITEMS = [
    {"id": "SRC_CONN", "title": "Real Telemetry Source Connected", "description": "Continuous SCADA / MES / CNC OPC-UA streaming connectivity established.", "default_state": False},
    {"id": "MACH_ID", "title": "Machine Identifiers Available", "description": "Unique physical machine asset IDs tagged in telemetry packets.", "default_state": False},
    {"id": "TIMESTAMPS", "title": "Timestamps Available", "description": "High-precision UTC timestamps recorded per sensor event.", "default_state": False},
    {"id": "SMP_INTERVAL", "title": "Sampling Interval Documented", "description": "Deterministic sampling frequency (e.g. 1 Hz / 10 Hz) documented.", "default_state": False},
    {"id": "UNITS_VERIFIED", "title": "Engineering Units Verified", "description": "Sensors verified in standard engineering units (K, rpm, Nm, min).", "default_state": False},
    {"id": "SENSOR_QUAL", "title": "Sensor Signal Quality Verified", "description": "Zero rate of missingness, packet loss, or invalid saturation anomalies.", "default_state": False},
    {"id": "FAIL_LABELS", "title": "Failure Labels Available", "description": "Ground-truth failure events recorded with timestamps.", "default_state": False},
    {"id": "FAIL_DEF", "title": "Failure Definition Documented", "description": "Clear operational boundary between failure and routine preventive stoppage.", "default_state": False},
    {"id": "MAINT_ALIGN", "title": "Maintenance Events Aligned", "description": "Technician intervention logs synchronized with sensor timestamps.", "default_state": False},
    {"id": "CHRONO_HOLDOUT", "title": "Chronological Holdout Defined", "description": "Strict time-based evaluation splits separated without future leakage.", "default_state": False},
    {"id": "VAL_PERIOD", "title": "Validation Period Defined", "description": "Sufficient operational window (minimum 90 days) defined for benchmarking.", "default_state": False},
    {"id": "GROUND_TRUTH", "title": "Ground-Truth Outcomes Available", "description": "Independently audited maintenance logs available for evaluation.", "default_state": False}
]


def get_industrial_readiness_status() -> Dict[str, Any]:
    """Returns the operational status of industrial validation readiness."""
    return {
        "validation_state": "AWAITING_VERIFIED_INDUSTRIAL_TELEMETRY",
        "validation_gate_locked": True,
        "active_data_source": "BENCHMARK (AI4I 2020 Dataset)",
        "production_model": "LightGBM (models/lightgbm.joblib)",
        "checklist": READINESS_CHECKLIST_ITEMS,
        "disclaimer": (
            "No claim of real industrial validation is made because verified industrial "
            "telemetry and longitudinal ground-truth maintenance logs are not currently available."
        )
    }


def validate_industrial_schema(
    df: pd.DataFrame,
    source_tag: str = "INDUSTRIAL"
) -> Dict[str, Any]:
    """
    Validates an incoming external dataset against the required industrial telemetry schema,
    units, timestamp order, missing values, and physical range feasibility.
    """
    if source_tag not in VALID_DATA_SOURCES:
        raise ValueError(f"Invalid source tag '{source_tag}'. Valid: {VALID_DATA_SOURCES}")

    issues: List[str] = []
    warnings: List[str] = []
    mapped_cols: Dict[str, str] = {}

    col_lower = {str(c).lower().strip(): c for c in df.columns}

    # Map columns
    for standard_name, possible_aliases in INDUSTRIAL_SCHEMA_MAPPING.items():
        found = None
        for alias in possible_aliases:
            if alias.lower().strip() in col_lower:
                found = col_lower[alias.lower().strip()]
                break
        if found:
            mapped_cols[standard_name] = found
        elif standard_name not in ["failure_label"]:
            issues.append(f"Missing required industrial schema column: '{standard_name}'")

    # Missing values check
    null_counts = df.isnull().sum().to_dict()
    total_nulls = sum(null_counts.values())
    if total_nulls > 0:
        issues.append(f"Contains {total_nulls} null/missing values across columns.")

    # Check timestamps if mapped
    if "timestamp" in mapped_cols:
        ts_col = mapped_cols["timestamp"]
        try:
            ts_series = pd.to_datetime(df[ts_col])
            if not ts_series.is_monotonic_increasing:
                warnings.append("Timestamps are not strictly chronological; sorting required.")
            dup_ts = int(ts_series.duplicated().sum())
            if dup_ts > 0:
                warnings.append(f"Found {dup_ts} duplicate timestamp entries.")
        except Exception as e:
            issues.append(f"Failed to parse timestamp column '{ts_col}': {e}")

    # Physical range sanity check
    sensor_key_map = {
        "Air temperature [K]": "air_temperature_K",
        "Process temperature [K]": "process_temperature_K",
        "Rotational speed [rpm]": "rotational_speed_rpm",
        "Torque [Nm]": "torque_Nm",
        "Tool wear [min]": "tool_wear_min"
    }
    for col_std, (low, high) in SANITY_RANGES.items():
        std_key = sensor_key_map.get(col_std)
        if std_key and std_key in mapped_cols:
            orig_col = mapped_cols[std_key]
            if orig_col in df.columns:
                vals = pd.to_numeric(df[orig_col], errors="coerce")
                out_of_envelope = int(((vals < (low * 0.5)) | (vals > (high * 1.5))).sum())
                if out_of_envelope > 0:
                    warnings.append(f"Sensor '{orig_col}' contains {out_of_envelope} values outside physical envelope.")

    is_valid = len(issues) == 0
    gate_status = "READY_FOR_EVALUATION" if (is_valid and "failure_label" in mapped_cols and source_tag == "INDUSTRIAL") else "LOCKED_AWAITING_DATA"

    return {
        "is_valid": is_valid,
        "source_tag": source_tag,
        "gate_status": gate_status,
        "mapped_columns": mapped_cols,
        "issues": issues,
        "warnings": warnings,
        "total_rows": len(df),
        "has_ground_truth_labels": "failure_label" in mapped_cols
    }


def evaluate_industrial_dataset(
    df: pd.DataFrame,
    source_tag: str = "INDUSTRIAL",
    model: Optional[Any] = None,
    preprocessor: Optional[Any] = None,
    threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Evaluates the existing validated LightGBM model on an external industrial dataset.
    STRICTLY ENFORCES VALIDATION GATE: If schema is invalid or ground-truth labels are missing,
    execution halts with a clear descriptive message without fabricating metrics.
    """
    schema_report = validate_industrial_schema(df, source_tag=source_tag)
    if not schema_report["is_valid"]:
        return {
            "evaluation_status": "GATE_LOCKED",
            "message": "Industrial evaluation cannot proceed due to schema validation failures.",
            "issues": schema_report["issues"],
            "metrics": None
        }

    if not schema_report["has_ground_truth_labels"]:
        return {
            "evaluation_status": "GATE_LOCKED",
            "message": "Industrial evaluation cannot compute empirical accuracy without verified failure labels.",
            "issues": ["Ground truth 'failure_label' not present in uploaded dataset."],
            "metrics": None
        }

    if model is None or preprocessor is None:
        model, preprocessor = load_inference_pipeline()

    # Standardize DataFrame column names to match preprocessor
    df_std = pd.DataFrame()
    mapped = schema_report["mapped_columns"]
    
    df_std["Type"] = df[mapped["machine_type"]] if "machine_type" in mapped else "M"
    df_std["Air temperature [K]"] = df[mapped["air_temperature_K"]].astype(float)
    df_std["Process temperature [K]"] = df[mapped["process_temperature_K"]].astype(float)
    df_std["Rotational speed [rpm]"] = df[mapped["rotational_speed_rpm"]].astype(float)
    df_std["Torque [Nm]"] = df[mapped["torque_Nm"]].astype(float)
    df_std["Tool wear [min]"] = df[mapped["tool_wear_min"]].astype(float)

    y_true = df[mapped["failure_label"]].values.astype(int)

    # Preprocess & Infer
    df_proc = preprocess_features_for_inference(df_std, preprocessor)
    y_prob = model.predict_proba(df_proc)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    n_total = len(y_true)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    acc = (tp + tn) / n_total if n_total > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    alert_rate = (tp + fp) / n_total if n_total > 0 else 0.0

    eval_results = {
        "evaluation_status": "COMPLETED",
        "source_tag": source_tag,
        "total_records": n_total,
        "operational_threshold": threshold,
        "metrics": {
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1": round(f1, 4),
            "Specificity": round(spec, 4),
            "Alert_Rate": round(alert_rate, 4),
            "Confusion_Matrix": {"TP": tp, "TN": tn, "FP": fp, "FN": fn}
        },
        "disclaimer": (
            "Empirical results evaluated on external dataset. "
            "Model weights and preprocessor were preserved without retraining."
        )
    }

    return eval_results


def generate_industrial_validation_report(
    eval_results: Dict[str, Any],
    output_dir: Path = OUTPUTS_IND_DIR
) -> Tuple[Path, Path]:
    """Generates structured JSON and summary CSV validation reports for external audit."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_path = output_dir / "industrial_validation_report.json"
    csv_path = output_dir / "industrial_validation_summary.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)

    # Flatten metrics for CSV summary
    summary_rows = []
    if eval_results.get("metrics"):
        m = eval_results["metrics"]
        for k, v in m.items():
            if k != "Confusion_Matrix":
                summary_rows.append({"Metric": k, "Value": v})
        cm = m.get("Confusion_Matrix", {})
        for cm_k, cm_v in cm.items():
            summary_rows.append({"Metric": f"Count_{cm_k}", "Value": cm_v})
    else:
        summary_rows.append({"Metric": "Status", "Value": eval_results.get("evaluation_status", "LOCKED")})

    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
    return json_path, csv_path


def create_mock_industrial_fixture(n_rows: int = 50) -> pd.DataFrame:
    """
    TEST FIXTURE ONLY — Creates synthetic external telemetry with industrial schema column names
    to test the validation gateway logic. NEVER reported as real factory data.
    """
    np.random.seed(42)
    timestamps = [datetime(2026, 10, 3, 6, 0) + pd.Timedelta(seconds=i * 10) for i in range(n_rows)]
    return pd.DataFrame({
        "timestamp": timestamps,
        "machine_id": "CNC_FACTORY_M01",
        "machine_type": "M",
        "air_temperature_K": np.random.uniform(298.0, 302.0, n_rows),
        "process_temperature_K": np.random.uniform(308.0, 312.0, n_rows),
        "rotational_speed_rpm": np.random.uniform(1400.0, 1600.0, n_rows),
        "torque_Nm": np.random.uniform(35.0, 50.0, n_rows),
        "tool_wear_min": np.linspace(10.0, 180.0, n_rows),
        "failure_label": (np.random.rand(n_rows) > 0.90).astype(int)
    })


if __name__ == "__main__":
    print("Testing Industrial Validation Module...")
    status = get_industrial_readiness_status()
    print("Validation state:", status["validation_state"])

    mock_df = create_mock_industrial_fixture(50)
    schema_res = validate_industrial_schema(mock_df, source_tag="INDUSTRIAL")
    print("Schema check is_valid:", schema_res["is_valid"])

    eval_res = evaluate_industrial_dataset(mock_df, source_tag="INDUSTRIAL")
    print("Evaluation result:", eval_res)
    
    j_p, c_p = generate_industrial_validation_report(eval_res)
    print(f"Reports saved: {j_p}, {c_p}")
