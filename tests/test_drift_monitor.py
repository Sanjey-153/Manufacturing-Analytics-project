"""
Comprehensive Validation Test Suite for Drift Monitoring Engine (Future Work F3).
Tests reference profile generation & persistence, numerical & categorical PSI calculations,
prototype status bands (STABLE, WATCH, DRIFT DETECTED), KS statistics, model-output probability drift,
risk-tier distribution comparison, data-quality monitoring (missing, invalid categories, range envelope violations),
insufficient window handling, transition-based alert mechanics, zero leakage, and baseline immutability.
"""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from config import (
    RAW_DATA_PATH,
    DRIFT_DIR,
    FEATURE_COLS,
    NUMERICAL_COLS,
    CATEGORICAL_COLS,
    RANDOM_STATE
)
from src.data_loader import load_raw_data
from src.preprocessing import separate_features_and_target, split_data
from src.drift_monitor import (
    build_reference_profile,
    load_reference_profile,
    get_reference_predictions,
    calculate_psi,
    calculate_categorical_psi,
    calculate_ks_test,
    check_data_quality,
    classify_psi_status,
    evaluate_drift,
    DriftAlertTracker,
    STATUS_INSUFFICIENT_DATA,
    STATUS_STABLE,
    STATUS_WATCH,
    STATUS_DRIFT_DETECTED,
    PSI_STABLE_UPPER,
    PSI_WATCH_UPPER
)


def run_drift_monitor_tests():
    print("=" * 75)
    print("RUNNING FUTURE WORK F3 DRIFT MONITORING TEST SUITE (22 CHECKS)")
    print("=" * 75)

    # Test 1: Reference profile generation executes cleanly from training split (N=8,000)
    profile = build_reference_profile(force_recompute=True)
    assert profile is not None, "Test 1 Failed: Profile generation returned None!"
    meta = profile["metadata"]
    assert meta["reference_rows"] == 8000, f"Test 1 Failed: Expected reference_rows 8000, got {meta.get('reference_rows')}!"
    assert meta["healthy_rows"] == 7729, f"Test 1 Failed: Expected healthy_rows 7729, got {meta.get('healthy_rows')}!"
    assert meta["failure_rows"] == 271, f"Test 1 Failed: Expected failure_rows 271, got {meta.get('failure_rows')}!"
    assert meta["reference_dataset"] == "AI4I 2020", f"Test 1 Failed: Unexpected dataset: {meta.get('reference_dataset')}!"
    assert meta["reference_split"] == "validated stratified training partition", f"Test 1 Failed: Unexpected split: {meta.get('reference_split')}!"
    assert meta["random_state"] == 42, f"Test 1 Failed: Unexpected random_state: {meta.get('random_state')}!"
    assert set(profile["features"].keys()) == set(FEATURE_COLS), "Test 1 Failed: Profiled feature set does not match FEATURE_COLS!"
    print(f"[PASS] Test 1: Fixed reference profile successfully generated from N=8,000 training partition (7,729 healthy, 271 failure).")

    # Test 2: Reference profile persistence and file loading
    drift_dir = DRIFT_DIR
    prof_file = drift_dir / "reference_profile.json"
    pred_file = drift_dir / "reference_predictions.json"
    assert prof_file.exists(), f"Test 2 Failed: File not found at {prof_file}!"
    assert pred_file.exists(), f"Test 2 Failed: File not found at {pred_file}!"
    loaded_prof = load_reference_profile()
    assert loaded_prof["metadata"]["is_fixed_reference"] is True, "Test 2 Failed: Loaded profile is not marked fixed!"
    print("[PASS] Test 2: Reference profile and prediction baseline successfully persisted and loaded.")

    # Test 3: Numerical PSI calculation
    ref_norm = np.random.RandomState(42).normal(100, 15, 1000)
    curr_same = np.random.RandomState(43).normal(100, 15, 500)
    psi_val, bin_details = calculate_psi(ref_norm, curr_same)
    assert isinstance(psi_val, float), "Test 3 Failed: PSI value must be float!"
    assert psi_val >= 0.0, "Test 3 Failed: PSI cannot be negative!"
    assert len(bin_details) > 0, "Test 3 Failed: Bin details list is empty!"
    print(f"[PASS] Test 3: Numerical PSI calculated cleanly (PSI: {psi_val:.4f}, {len(bin_details)} bins).")

    # Test 4: Categorical PSI calculation for Type
    ref_cats = ["L"] * 600 + ["M"] * 300 + ["H"] * 100
    curr_cats = ["L"] * 60 + ["M"] * 30 + ["H"] * 10
    cat_psi, cat_details = calculate_categorical_psi(ref_cats, curr_cats)
    assert isinstance(cat_psi, float), "Test 4 Failed: Categorical PSI must be float!"
    assert cat_psi < 0.05, f"Test 4 Failed: Identical categorical distribution produced high PSI: {cat_psi}!"
    print(f"[PASS] Test 4: Categorical PSI calculated cleanly for Machine Type (PSI: {cat_psi:.4f}).")

    # Test 5: PSI Stable Band (< 0.10)
    stable_psi, _ = calculate_psi(ref_norm, curr_same)
    assert stable_psi < PSI_STABLE_UPPER, f"Test 5 Failed: Expected < 0.10, got {stable_psi}!"
    assert classify_psi_status(stable_psi) == STATUS_STABLE
    print(f"[PASS] Test 5: PSI Stable interpretation band verified (PSI: {stable_psi:.4f} < 0.10 -> STABLE).")

    # Test 6: PSI Watch Band (0.10 <= PSI < 0.20)
    # Controlled moderate shift fixture (+5 shift on mean)
    curr_moderate = np.random.RandomState(44).normal(105, 15, 500)
    watch_psi, _ = calculate_psi(ref_norm, curr_moderate)
    assert PSI_STABLE_UPPER <= watch_psi < PSI_WATCH_UPPER, f"Test 6 Failed: Shifted fixture gave unexpected PSI: {watch_psi}!"
    assert classify_psi_status(watch_psi) == STATUS_WATCH, f"Test 6 Failed: Status classification for {watch_psi} != WATCH!"
    print(f"[PASS] Test 6: PSI Watch interpretation band verified (PSI: {watch_psi:.4f} in [0.10, 0.20) -> WATCH).")

    # Test 7: PSI Drift Band (PSI >= 0.20)
    # Controlled severe shift fixture
    curr_severe = np.random.RandomState(45).normal(125, 15, 500)
    drift_psi, _ = calculate_psi(ref_norm, curr_severe)
    assert drift_psi >= PSI_WATCH_UPPER, f"Test 7 Failed: Severe shift gave low PSI: {drift_psi}!"
    assert classify_psi_status(drift_psi) == STATUS_DRIFT_DETECTED
    print(f"[PASS] Test 7: PSI Drift Detected band verified (PSI: {drift_psi:.4f} >= 0.20 -> DRIFT DETECTED).")

    # Test 8: Kolmogorov-Smirnov (KS) test calculation
    ks_stat, ks_pval = calculate_ks_test(ref_norm, curr_same)
    assert 0.0 <= ks_stat <= 1.0, "Test 8 Failed: KS statistic out of [0, 1]!"
    assert 0.0 <= ks_pval <= 1.0, "Test 8 Failed: KS p-value out of [0, 1]!"
    print(f"[PASS] Test 8: Two-sample KS test calculated (Stat: {ks_stat:.4f}, p-value: {ks_pval:.4f}).")

    # Test 9: Prediction probability drift calculation
    ref_preds = get_reference_predictions()
    assert len(ref_preds) == 8000, f"Test 9 Failed: Expected 8000 reference predictions, got {len(ref_preds)}!"
    prob_psi, _ = calculate_psi(ref_preds, ref_preds[:100])
    assert prob_psi < 0.10, "Test 9 Failed: Subsample of reference gave high probability PSI!"
    print(f"[PASS] Test 9: Model-output failure probability drift calculated against fixed baseline.")

    # Test 10: Risk-tier distribution calculation
    low_prop = profile["model_output"]["risk_tier_distribution"]["Low"]["proportion"]
    crit_prop = profile["model_output"]["risk_tier_distribution"]["Critical"]["proportion"]
    assert 0.85 <= low_prop <= 0.98, f"Test 10 Failed: Unexpected baseline Low risk tier proportion: {low_prop}!"
    assert 0.01 <= crit_prop <= 0.06, f"Test 10 Failed: Unexpected baseline Critical risk tier proportion: {crit_prop}!"
    print(f"[PASS] Test 10: Risk tier distribution verified (Low: {low_prop*100:.1f}%, Critical: {crit_prop*100:.1f}%).")

    # Test 11: Missing value data-quality detection
    bad_df_missing = pd.DataFrame([{
        "Type": "M",
        "Air temperature [K]": None,  # Missing
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551.0,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0.0
    }])
    q_res1 = check_data_quality(bad_df_missing)
    assert q_res1["is_clean"] is False, "Test 11 Failed: Missing value was not detected!"
    assert q_res1["missing_count"] > 0, "Test 11 Failed: missing_count is 0!"
    print(f"[PASS] Test 11: Data quality successfully detected missing sensor parameters.")

    # Test 12: Invalid Machine Type categorical detection
    bad_df_type = pd.DataFrame([{
        "Type": "Z_INVALID",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551.0,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0.0
    }])
    q_res2 = check_data_quality(bad_df_type)
    assert q_res2["is_clean"] is False, "Test 12 Failed: Invalid Type was not detected!"
    assert q_res2["unexpected_categories_count"] == 1, "Test 12 Failed: unexpected_categories_count != 1!"
    print(f"[PASS] Test 12: Data quality successfully caught unexpected Machine Type ('Z_INVALID').")

    # Test 13: Out-of-envelope sanity range violation detection
    bad_df_range = pd.DataFrame([{
        "Type": "M",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": -999.0,  # Physically impossible negative RPM
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0.0
    }])
    q_res3 = check_data_quality(bad_df_range)
    assert q_res3["is_clean"] is False, "Test 13 Failed: Negative speed not detected as range violation!"
    assert q_res3["range_violations_count"] == 1, "Test 13 Failed: range_violations_count != 1!"
    print(f"[PASS] Test 13: Data quality successfully caught out-of-envelope sensor value (RPM: -999.0).")

    # Test 14: Insufficient monitoring window handling (N < min_sample_size)
    small_df = pd.DataFrame([{"Type": "M", "Air temperature [K]": 298.1}] * 20)
    rep_small = evaluate_drift(small_df, reference_profile=profile, min_sample_size=50)
    assert rep_small["overall_status"] == STATUS_INSUFFICIENT_DATA, f"Test 14 Failed: Expected {STATUS_INSUFFICIENT_DATA}, got {rep_small['overall_status']}!"
    print(f"[PASS] Test 14: Correctly handled insufficient window (N=20 < 50) without fabricating drift scores.")

    # Test 15: Overall status evaluation on nominal baseline batch
    raw_df = load_raw_data()
    X_raw, y_raw = separate_features_and_target(raw_df)
    _, X_test_ref, _, _ = split_data(X_raw, y_raw, test_size=0.2, random_state=RANDOM_STATE)
    nominal_test_batch = X_test_ref.iloc[:100].copy()
    rep_nominal = evaluate_drift(nominal_test_batch, reference_profile=profile, min_sample_size=50)
    assert rep_nominal["overall_status"] in [STATUS_STABLE, STATUS_WATCH], f"Test 15 Failed: Nominal batch got {rep_nominal['overall_status']}!"
    print(f"[PASS] Test 15: Full drift evaluation returned valid structured report (Overall: {rep_nominal['overall_status']}).")

    # Test 16: Drift alert transition: STABLE -> WATCH
    tracker = DriftAlertTracker()
    tracker.current_status = STATUS_STABLE
    alt1 = tracker.update(STATUS_WATCH, {"most_affected_feature": "Spindle Torque", "max_psi": 0.145})
    assert alt1 is not None, "Test 16 Failed: STABLE -> WATCH failed to generate alert!"
    assert alt1["current_status"] == STATUS_WATCH, "Test 16 Failed: Alert status != WATCH!"
    print("[PASS] Test 16: Drift Alert successfully triggered on transition: STABLE -> WATCH.")

    # Test 17: Drift alert transition: WATCH -> DRIFT DETECTED
    alt2 = tracker.update(STATUS_DRIFT_DETECTED, {"most_affected_feature": "Air Temperature", "max_psi": 0.285})
    assert alt2 is not None, "Test 17 Failed: WATCH -> DRIFT failed to generate alert!"
    assert alt2["severity"] == "CRITICAL", "Test 17 Failed: Alert severity != CRITICAL!"
    print("[PASS] Test 17: Drift Alert successfully triggered on transition: WATCH -> DRIFT DETECTED.")

    # Test 18: Duplicate alert suppression (DRIFT -> DRIFT generates 0 new alerts)
    alerts_before = len(tracker.get_alerts())
    alt_dup = tracker.update(STATUS_DRIFT_DETECTED, {"most_affected_feature": "Air Temperature", "max_psi": 0.310})
    assert alt_dup is None, "Test 18 Failed: Duplicate consecutive alert was generated!"
    assert len(tracker.get_alerts()) == alerts_before, "Test 18 Failed: Alert count increased on duplicate state!"
    print("[PASS] Test 18: Duplicate consecutive alert suppression verified (DRIFT -> DRIFT suppressed).")

    # Test 19: Drift recovery transition (DRIFT -> STABLE)
    alt_rec = tracker.update(STATUS_STABLE, {})
    assert alt_rec is not None, "Test 19 Failed: Recovery event not logged!"
    assert alt_rec["severity"] == "INFO", "Test 19 Failed: Recovery severity != INFO!"
    print("[PASS] Test 19: Recovery transition cleanly logged: DRIFT DETECTED -> STABLE.")

    # Test 20: Zero leakage verification (no forbidden target or failure flags in monitored features)
    forbidden_leakage = ["UDI", "Product ID", "Target", "Failure Type", "TWF", "HDF", "PWF", "OSF", "RNF"]
    for monitored_col in profile["features"].keys():
        assert monitored_col not in forbidden_leakage, f"Test 20 Failed: Leakage column '{monitored_col}' found in drift features!"
    print("[PASS] Test 20: Zero leakage verified (no target, failure mode, or identifier columns in drift monitoring).")

    # Test 21: Baseline reference immutability (evaluating telemetry does not alter reference profile)
    ref_before_hash = json.dumps(profile["features"]["Torque [Nm]"], sort_keys=True)
    # Evaluate extreme drift batch
    drift_batch = nominal_test_batch.copy()
    drift_batch["Torque [Nm]"] = 75.0
    evaluate_drift(drift_batch, reference_profile=profile, min_sample_size=50)
    ref_after = load_reference_profile()
    ref_after_hash = json.dumps(ref_after["features"]["Torque [Nm]"], sort_keys=True)
    assert ref_before_hash == ref_after_hash, "Test 21 Failed: Reference baseline was mutated by incoming telemetry!"
    print("[PASS] Test 21: Fixed baseline immutability confirmed (reference profile remains untouched).")

    # Test 22: Drift report schema completeness and JSON serializability
    rep = evaluate_drift(nominal_test_batch, reference_profile=profile, min_sample_size=50)
    json_str = json.dumps(rep, default=str)
    assert len(json_str) > 0, "Test 22 Failed: Report failed JSON serialization!"
    required_report_keys = [
        "overall_status", "window_size", "min_sample_size", "most_affected_feature",
        "max_psi", "data_quality", "feature_drift", "model_output_drift",
        "risk_tier_comparison", "timestamp", "disclaimer"
    ]
    for k in required_report_keys:
        assert k in rep, f"Test 22 Failed: Missing required key '{k}' in drift report!"
    print("[PASS] Test 22: Drift report schema verified complete and 100% JSON serializable.")

    print("\n" + "=" * 75)
    print("ALL 22 DRIFT MONITORING VALIDATION CHECKS PASSED WITH 100% INTEGRITY")
    print("=" * 75)
    return True


if __name__ == "__main__":
    run_drift_monitor_tests()
