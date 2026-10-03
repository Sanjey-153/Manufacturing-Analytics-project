"""
Advanced Time-Series Pipeline and Sequential Architecture for CNC Predictive Maintenance (F6).
Provides sequence window construction, chronological sorting, temporal feature engineering,
data-quality validation, chronological split design, and future temporal model interface.

IMPORTANT RESEARCH LIMITATION & ARCHITECTURAL DISCLAIMER:
The AI4I 2020 dataset is an independent cycle benchmark and does NOT represent genuine
longitudinal sequential telemetry from a single physical CNC milling machine.
This module establishes a TIME-SERIES READINESS ARCHITECTURE to ingest future genuine
sequential factory telemetry without fabricating validated temporal performance metrics.
The production baseline model remains the validated LightGBM classifier.
"""

import sys
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
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
    ID_COLS,
    FAILURE_TYPE_COLS,
    TARGET_COL
)

# Supported Configurable Window Sizes
SUPPORTED_SEQUENCE_LENGTHS = [5, 10, 20, 30, 50]
DEFAULT_SEQUENCE_LENGTH = 20

# Expected Sequential Telemetry Schema
REQUIRED_TEMPORAL_COLS = ["Timestamp", "Machine Reference", "Type"] + NUMERICAL_COLS


class TemporalPredictor(ABC):
    """
    Abstract Interface for future sequential machine learning models (LSTM, GRU, Temporal CNN, Transformers).
    """

    @abstractmethod
    def fit(self, X_seq: np.ndarray, y_seq: np.ndarray) -> "TemporalPredictor":
        """Fits the sequential model on 3D sequence tensors."""
        pass

    @abstractmethod
    def predict_proba(self, X_seq: np.ndarray) -> np.ndarray:
        """Predicts failure probability for sequence windows."""
        pass

    @abstractmethod
    def predict(self, X_seq: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predicts binary classification for sequence windows."""
        pass

    @abstractmethod
    def save(self, model_path: Union[str, Path]) -> None:
        """Serializes model weights."""
        pass

    @abstractmethod
    def load(self, model_path: Union[str, Path]) -> "TemporalPredictor":
        """Deserializes model weights."""
        pass


class MockTemporalPredictor(TemporalPredictor):
    """
    TEST MOCK ADAPTER — Clearly labeled unit-test mock for interface verification.
    NEVER used as a production or validated empirical model.
    """

    def __init__(self, sequence_length: int = 20):
        self.sequence_length = sequence_length
        self.is_fitted = False
        self.model_type = "TEST_MOCK_LSTM (Non-Production Unit-Test Fixture)"

    def fit(self, X_seq: np.ndarray, y_seq: np.ndarray) -> "MockTemporalPredictor":
        if len(X_seq.shape) != 3:
            raise ValueError(f"Expected 3D sequence tensor (N, L, F), got {X_seq.shape}")
        self.is_fitted = True
        return self

    def predict_proba(self, X_seq: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("MockTemporalPredictor must be fitted before predict_proba.")
        # Deterministic mock probability based on last timestep wear and torque
        n_samples = X_seq.shape[0]
        # Return mock probabilities bounded in [0.01, 0.99]
        return np.full((n_samples, 2), [0.85, 0.15])

    def predict(self, X_seq: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X_seq)[:, 1]
        return (probs >= threshold).astype(int)

    def save(self, model_path: Union[str, Path]) -> None:
        pass

    def load(self, model_path: Union[str, Path]) -> "MockTemporalPredictor":
        self.is_fitted = True
        return self


def get_temporal_model_status() -> Dict[str, Any]:
    """Returns the operational deployment status of sequential/temporal modeling."""
    return {
        "status": "AWAITING_VERIFIED_SEQUENTIAL_TELEMETRY",
        "production_model": "LightGBM (models/lightgbm.joblib)",
        "temporal_architecture_state": "ARCHITECTURE_READY",
        "validated_temporal_model_deployed": False,
        "limitation_note": (
            "AI4I 2020 benchmark consists of independent CNC milling records without "
            "genuine continuous temporal machine tracking. Temporal deep-learning models "
            "will be trained once high-frequency SCADA/MES sequential telemetry is collected."
        )
    }


def validate_temporal_telemetry(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Performs comprehensive temporal data-quality validation on sequential telemetry.
    Checks for: missing timestamps, duplicates, non-monotonic ordering, irregular intervals,
    missing sensor values, mixed machine IDs, and insufficient sequence length.
    """
    issues: List[str] = []
    warnings: List[str] = []

    # Check required columns
    missing_cols = [c for c in REQUIRED_TEMPORAL_COLS if c not in df.columns]
    if missing_cols:
        return {
            "status": "INVALID",
            "is_valid": False,
            "issues": [f"Missing required temporal columns: {missing_cols}"],
            "warnings": [],
            "total_records": len(df)
        }

    # Zero leakage check
    forbidden = [c for c in ID_COLS + FAILURE_TYPE_COLS if c in df.columns]
    if forbidden:
        issues.append(f"Leakage columns detected in telemetry sequence: {forbidden}")

    # Check timestamps
    if df["Timestamp"].isnull().any():
        issues.append(f"Contains {df['Timestamp'].isnull().sum()} missing/null timestamps.")

    try:
        ts_series = pd.to_datetime(df["Timestamp"])
    except Exception as e:
        issues.append(f"Failed to parse timestamps as datetime objects: {e}")
        ts_series = None

    if ts_series is not None:
        # Check duplicates
        dup_count = int(ts_series.duplicated().sum())
        if dup_count > 0:
            warnings.append(f"Detected {dup_count} duplicate timestamp records.")

        # Check monotonic ordering
        if not ts_series.is_monotonic_increasing:
            warnings.append("Timestamps are not strictly monotonically increasing; sorting required.")

        # Check irregular sampling intervals
        diffs = ts_series.diff().dropna()
        if len(diffs) > 1:
            median_dt = diffs.median()
            max_dt = diffs.max()
            if max_dt > (median_dt * 5):
                warnings.append(f"Detected potential telemetry time gap / irregular interval (Max gap: {max_dt}).")

    # Sensor nulls / infs
    for col in NUMERICAL_COLS:
        n_nulls = int(df[col].isnull().sum())
        if n_nulls > 0:
            issues.append(f"Sensor feature '{col}' contains {n_nulls} missing values.")

    # Sequence length check
    if len(df) < DEFAULT_SEQUENCE_LENGTH:
        warnings.append(f"Sequence length ({len(df)}) is less than default window ({DEFAULT_SEQUENCE_LENGTH}).")

    # Machine identifier continuity
    n_machines = df["Machine Reference"].nunique()
    if n_machines > 1:
        warnings.append(f"Data contains {n_machines} distinct machine references. Sequences must be built per-machine.")

    status = "INVALID" if len(issues) > 0 else ("WARNING" if len(warnings) > 0 else "VALID")

    return {
        "status": status,
        "is_valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "total_records": len(df),
        "unique_machines": int(n_machines)
    }


def compute_derived_temporal_features(
    df: pd.DataFrame,
    numerical_cols: Optional[List[str]] = None,
    rolling_window: int = 5
) -> pd.DataFrame:
    """
    Computes lag features, rolling statistics (mean, std, min, max), and first differences
    for continuous machine telemetry sensors.
    """
    if numerical_cols is None:
        numerical_cols = NUMERICAL_COLS

    df_out = df.copy()
    
    # Sort chronologically per machine before feature derivation
    if "Timestamp" in df_out.columns:
        df_out["Timestamp"] = pd.to_datetime(df_out["Timestamp"])
        if "Machine Reference" in df_out.columns:
            df_out = df_out.sort_values(["Machine Reference", "Timestamp"]).reset_index(drop=True)
        else:
            df_out = df_out.sort_values("Timestamp").reset_index(drop=True)

    for col in numerical_cols:
        if col not in df_out.columns:
            continue

        # Lags
        df_out[f"{col}_lag1"] = df_out[col].shift(1)
        df_out[f"{col}_lag2"] = df_out[col].shift(2)

        # First Difference (Velocity / Rate of change)
        df_out[f"{col}_first_diff"] = df_out[col].diff()

        # Rolling Statistics
        df_out[f"{col}_roll_mean"] = df_out[col].rolling(window=rolling_window, min_periods=1).mean()
        df_out[f"{col}_roll_std"] = df_out[col].rolling(window=rolling_window, min_periods=1).std().fillna(0.0)
        df_out[f"{col}_roll_min"] = df_out[col].rolling(window=rolling_window, min_periods=1).min()
        df_out[f"{col}_roll_max"] = df_out[col].rolling(window=rolling_window, min_periods=1).max()

    return df_out


def build_sequence_windows(
    df: pd.DataFrame,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    feature_cols: Optional[List[str]] = None,
    target_col: Optional[str] = TARGET_COL,
    stride: int = 1
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Constructs 3D sequence window tensors (N_sequences, sequence_length, N_features)
    for sequential telemetry. Strictly maintains per-machine boundaries.
    """
    if feature_cols is None:
        feature_cols = NUMERICAL_COLS

    if sequence_length not in SUPPORTED_SEQUENCE_LENGTHS:
        raise ValueError(f"Unsupported sequence length {sequence_length}. Supported: {SUPPORTED_SEQUENCE_LENGTHS}")

    X_list = []
    y_list = []

    # Process grouped by machine to avoid cross-machine sequence contamination
    machine_col = "Machine Reference" if "Machine Reference" in df.columns else None
    groups = df.groupby(machine_col) if machine_col else [("ALL", df)]

    for _, group in groups:
        # Chronological sort
        if "Timestamp" in group.columns:
            g_sorted = group.sort_values("Timestamp")
        else:
            g_sorted = group

        feature_mat = g_sorted[feature_cols].values
        has_target = target_col and target_col in g_sorted.columns
        target_vec = g_sorted[target_col].values if has_target else None

        n_rows = len(g_sorted)
        if n_rows < sequence_length:
            continue

        for start_idx in range(0, n_rows - sequence_length + 1, stride):
            end_idx = start_idx + sequence_length
            X_list.append(feature_mat[start_idx:end_idx, :])
            if has_target:
                # Target at end of sequence
                y_list.append(target_vec[end_idx - 1])

    if not X_list:
        empty_x = np.empty((0, sequence_length, len(feature_cols)))
        empty_y = np.empty((0,)) if target_col else None
        return empty_x, empty_y

    X_arr = np.array(X_list, dtype=np.float32)
    y_arr = np.array(y_list, dtype=np.int32) if y_list else None

    return X_arr, y_arr


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits sequential telemetry chronologically:
    - Train: Earliest period (0% to 70%)
    - Validation: Middle period (70% to 85%)
    - Test: Latest holdout period (85% to 100%)
    Strictly avoids random shuffling across time boundaries.
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0")

    if "Timestamp" in df.columns:
        df_sorted = df.sort_values("Timestamp").reset_index(drop=True)
    else:
        df_sorted = df.reset_index(drop=True)

    n = len(df_sorted)
    idx_train_end = int(n * train_ratio)
    idx_val_end = int(n * (train_ratio + val_ratio))

    df_train = df_sorted.iloc[:idx_train_end].copy()
    df_val = df_sorted.iloc[idx_train_end:idx_val_end].copy()
    df_test = df_sorted.iloc[idx_val_end:].copy()

    return df_train, df_val, df_test


def generate_synthetic_sequential_fixture(
    n_steps: int = 100,
    machine_id: str = "CNC-Mill-Seq01",
    start_time: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Generates a deterministic continuous sequential telemetry fixture for testing and UI demonstration.
    """
    if start_time is None:
        start_time = datetime(2026, 10, 3, 8, 0, 0)

    timestamps = [start_time + timedelta(seconds=i * 10) for i in range(n_steps)]
    
    # Smooth continuous physical signals
    t = np.linspace(0, 10, n_steps)
    air_temp = 298.0 + 2.0 * np.sin(t / 2)
    proc_temp = air_temp + 10.0 + 1.5 * np.cos(t / 3)
    speed = 1500.0 + 120.0 * np.sin(t)
    torque = 40.0 + 8.0 * np.cos(t / 1.5)
    tool_wear = np.linspace(5.0, 180.0, n_steps)

    df_synth = pd.DataFrame({
        "Timestamp": timestamps,
        "Machine Reference": machine_id,
        "Type": "M",
        "Air temperature [K]": np.round(air_temp, 2),
        "Process temperature [K]": np.round(proc_temp, 2),
        "Rotational speed [rpm]": np.round(speed, 1),
        "Torque [Nm]": np.round(torque, 2),
        "Tool wear [min]": np.round(tool_wear, 1),
        "Target": (tool_wear > 160).astype(int)
    })
    return df_synth


if __name__ == "__main__":
    print("Testing Time-Series Pipeline Module...")
    df_seq = generate_synthetic_sequential_fixture(100)
    print("Sequential fixture generated:", df_seq.shape)
    
    val_report = validate_temporal_telemetry(df_seq)
    print("Validation report:", val_report)
    
    df_feats = compute_derived_temporal_features(df_seq)
    print("Derived features shape:", df_feats.shape)

    X_win, y_win = build_sequence_windows(df_seq, sequence_length=20)
    print("Sequence windows shape:", X_win.shape, y_win.shape if y_win is not None else None)

    tr, val, ts = chronological_split(df_seq)
    print(f"Chronological split: Train={len(tr)}, Val={len(val)}, Test={len(ts)}")

    mock_model = MockTemporalPredictor()
    mock_model.fit(X_win, y_win)
    preds = mock_model.predict(X_win)
    print("Mock prediction shape:", preds.shape)
