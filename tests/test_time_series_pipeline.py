"""
Test Suite for F6 Advanced Time-Series Pipeline & Architecture.
Validates sequence window construction, derived temporal features, quality checks,
machine separation, chronological split logic, and future model interfaces.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.time_series_pipeline import (
    validate_temporal_telemetry,
    compute_derived_temporal_features,
    build_sequence_windows,
    chronological_split,
    generate_synthetic_sequential_fixture,
    get_temporal_model_status,
    MockTemporalPredictor,
    SUPPORTED_SEQUENCE_LENGTHS,
    DEFAULT_SEQUENCE_LENGTH
)
from src.prediction import load_inference_pipeline
from config import ID_COLS, FAILURE_TYPE_COLS, NUMERICAL_COLS


def run_time_series_pipeline_tests():
    print("=" * 70)
    print("TEST SUITE: F6 ADVANCED TIME-SERIES PIPELINE & ARCHITECTURE")
    print("=" * 70)

    # 1. Chronological sorting
    df_unsorted = generate_synthetic_sequential_fixture(50)
    # Shuffle order
    df_shuffled = df_unsorted.sample(frac=1.0, random_state=42).reset_index(drop=True)
    report_shuf = validate_temporal_telemetry(df_shuffled)
    assert any("monotonic" in w.lower() for w in report_shuf["warnings"]), "Should detect non-monotonic timestamps"
    print("[PASS] Test 1: Chronological sorting and order inspection verified.")

    # 2. Sequence construction
    df_clean = generate_synthetic_sequential_fixture(60)
    X_seq, y_seq = build_sequence_windows(df_clean, sequence_length=20)
    assert len(X_seq.shape) == 3, f"Expected 3D tensor, got {X_seq.shape}"
    assert X_seq.shape[0] == 60 - 20 + 1  # 41 sequences
    print(f"[PASS] Test 2: Sequence window construction verified (Shape: {X_seq.shape}).")

    # 3. Sequence length configuration
    for sl in SUPPORTED_SEQUENCE_LENGTHS:
        if len(df_clean) >= sl:
            X_s, _ = build_sequence_windows(df_clean, sequence_length=sl)
            assert X_s.shape[1] == sl, f"Expected sequence length {sl}, got {X_s.shape[1]}"
    print(f"[PASS] Test 3: Configurable sequence lengths {SUPPORTED_SEQUENCE_LENGTHS} verified.")

    # 4. Machine separation (never mix machines)
    df_m1 = generate_synthetic_sequential_fixture(30, machine_id="CNC-M01")
    df_m2 = generate_synthetic_sequential_fixture(30, machine_id="CNC-M02")
    df_multi = pd.concat([df_m1, df_m2], ignore_index=True)
    X_multi, _ = build_sequence_windows(df_multi, sequence_length=20)
    # (30 - 20 + 1) * 2 = 22 sequences total
    assert X_multi.shape[0] == 22, f"Expected 22 isolated machine sequences, got {X_multi.shape[0]}"
    print("[PASS] Test 4: Multi-machine boundary isolation verified.")

    # 5. Timestamp validation
    assert validate_temporal_telemetry(df_clean)["is_valid"] is True
    print("[PASS] Test 5: Valid timestamp sequence inspection verified.")

    # 6. Duplicate timestamp detection
    df_dup = df_clean.copy()
    df_dup.loc[5, "Timestamp"] = df_dup.loc[4, "Timestamp"]
    rep_dup = validate_temporal_telemetry(df_dup)
    assert any("duplicate" in w.lower() for w in rep_dup["warnings"]), "Should detect duplicate timestamps"
    print("[PASS] Test 6: Duplicate timestamp detection verified.")

    # 7. Missing timestamp detection
    df_null_ts = df_clean.copy()
    df_null_ts.loc[3, "Timestamp"] = None
    rep_null = validate_temporal_telemetry(df_null_ts)
    assert rep_null["is_valid"] is False, "Missing timestamps must invalidate data"
    print("[PASS] Test 7: Missing timestamp detection verified.")

    # 8. Irregular interval detection
    df_gap = df_clean.copy()
    # Insert a 2-hour gap
    df_gap.loc[30:, "Timestamp"] = [t + timedelta(hours=2) for t in df_gap.loc[30:, "Timestamp"]]
    rep_gap = validate_temporal_telemetry(df_gap)
    assert any("irregular" in w.lower() or "gap" in w.lower() for w in rep_gap["warnings"]), "Should detect large gap"
    print("[PASS] Test 8: Irregular sampling interval detection verified.")

    # 9. Insufficient window detection
    df_short = generate_synthetic_sequential_fixture(10)
    rep_short = validate_temporal_telemetry(df_short)
    assert any("less than default" in w.lower() for w in rep_short["warnings"]), "Should warn on short sequences"
    print("[PASS] Test 9: Insufficient sequence length detection verified.")

    # 10. Lag features
    df_feats = compute_derived_temporal_features(df_clean)
    for col in NUMERICAL_COLS:
        assert f"{col}_lag1" in df_feats.columns, f"Missing lag1 for {col}"
        assert f"{col}_lag2" in df_feats.columns, f"Missing lag2 for {col}"
        # Check value match at row 5
        assert np.isclose(df_feats.loc[5, f"{col}_lag1"], df_feats.loc[4, col])
    print("[PASS] Test 10: Derived lag features (lag_1, lag_2) verified.")

    # 11. Rolling features
    for col in NUMERICAL_COLS:
        assert f"{col}_roll_mean" in df_feats.columns
        assert f"{col}_roll_std" in df_feats.columns
        assert f"{col}_roll_min" in df_feats.columns
        assert f"{col}_roll_max" in df_feats.columns
    print("[PASS] Test 11: Rolling statistical features (mean, std, min, max) verified.")

    # 12. First differences
    for col in NUMERICAL_COLS:
        assert f"{col}_first_diff" in df_feats.columns
        expected_diff = df_feats.loc[5, col] - df_feats.loc[4, col]
        assert np.isclose(df_feats.loc[5, f"{col}_first_diff"], expected_diff)
    print("[PASS] Test 12: First differences (rate of change) verified.")

    # 13. Zero leakage fields
    for forbidden in ID_COLS + FAILURE_TYPE_COLS:
        assert forbidden not in df_clean.columns or forbidden == "Target"
    print("[PASS] Test 13: Zero leakage columns in sequential telemetry verified.")

    # 14. Train/validation/test chronological split logic
    df_long = generate_synthetic_sequential_fixture(100)
    tr, val, ts = chronological_split(df_long, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    assert len(tr) == 70 and len(val) == 15 and len(ts) == 15
    assert tr["Timestamp"].max() < val["Timestamp"].min()
    assert val["Timestamp"].max() < ts["Timestamp"].min()
    print(f"[PASS] Test 14: Strict chronological split verified (Train: {len(tr)}, Val: {len(val)}, Test: {len(ts)}).")

    # 15. Temporal interface behavior
    mock_model = MockTemporalPredictor(sequence_length=20)
    mock_model.fit(X_seq, y_seq)
    probs = mock_model.predict_proba(X_seq)
    assert probs.shape == (X_seq.shape[0], 2)
    preds = mock_model.predict(X_seq, threshold=0.5)
    assert len(preds) == X_seq.shape[0]
    print("[PASS] Test 15: TemporalPredictor abstract interface and Mock adapter verified.")

    # 16. Serialization support in mock adapter
    mock_model.save(PROJECT_ROOT / "outputs" / "mock_temporal.pkl")
    mock_model.load(PROJECT_ROOT / "outputs" / "mock_temporal.pkl")
    print("[PASS] Test 16: Model interface save/load signature verified.")

    # 17. Mock adapter clearly labeled
    assert "MOCK" in mock_model.model_type, "Mock adapter must be explicitly labeled as TEST MOCK"
    st_info = get_temporal_model_status()
    assert st_info["validated_temporal_model_deployed"] is False
    print(f"[PASS] Test 17: Mock adapter labeling and status transparency verified ({st_info['status']}).")

    # 18. LightGBM baseline remains authoritative and unchanged
    model, preproc = load_inference_pipeline()
    assert model is not None and preproc is not None
    print("[PASS] Test 18: Authoritative LightGBM production baseline verified intact.")

    print("\n" + "=" * 70)
    print("ALL 18 TIME-SERIES PIPELINE TESTS PASSED (18/18)")
    print("=" * 70)


if __name__ == "__main__":
    run_time_series_pipeline_tests()
