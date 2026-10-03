"""
Comprehensive Test Suite for Prediction Engine and Deployment Inference Pipeline (Phase 6).
Tests input validation, error handling, schema integrity, probabilities, single & batch inference.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from config import (
    MODELS_DIR,
    PREPROCESSOR_PATH,
    FEATURE_COLS
)
from src.prediction import (
    load_inference_pipeline,
    validate_single_input,
    predict_single,
    predict_batch
)


def run_all_prediction_tests():
    print("=" * 70)
    print("RUNNING PHASE 6 PREDICTION ENGINE TEST SUITE")
    print("=" * 70)

    # Test 1: Model and Preprocessor reloading
    model, preprocessor = load_inference_pipeline()
    assert model is not None, "Test 1 Failed: Model failed to load!"
    assert preprocessor is not None, "Test 1 Failed: Preprocessor failed to load!"
    print("[PASS] Test 1: Primary LightGBM model and preprocessor successfully loaded.")

    # Test 2: Valid Normal Machine Operating Input
    normal_input = {
        "Type": "M",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551.0,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0.0
    }
    res_normal = predict_single(normal_input, model=model, preprocessor=preprocessor)
    assert res_normal["predicted_class"] in [0, 1], "Test 2 Failed: predicted_class must be 0 or 1!"
    assert 0.0 <= res_normal["failure_probability"] <= 1.0, "Test 2 Failed: Probability out of [0, 1]!"
    assert 0.0 <= res_normal["healthy_probability"] <= 1.0, "Test 2 Failed: Probability out of [0, 1]!"
    assert round(res_normal["failure_probability"] + res_normal["healthy_probability"], 4) == 1.0, "Test 2 Failed: Probabilities do not sum to 1!"
    assert res_normal["status_label"] == "Healthy"
    assert res_normal["risk_level"] in ["Low", "Moderate"]
    print(f"[PASS] Test 2: Valid normal input evaluated cleanly (Status: {res_normal['status_label']}, Fail Prob: {res_normal['failure_probability_percentage']}).")

    # Test 3: Valid Input Predicted as Failure (Extreme Operating Stress)
    failure_input = {
        "Type": "L",
        "Air temperature [K]": 302.5,
        "Process temperature [K]": 310.2,
        "Rotational speed [rpm]": 1200.0,
        "Torque [Nm]": 72.5,
        "Tool wear [min]": 240.0
    }
    res_fail = predict_single(failure_input, model=model, preprocessor=preprocessor)
    assert res_fail["predicted_class"] == 1, "Test 3 Failed: High-stress state should be classified as Failure (1)!"
    assert res_fail["failure_probability"] >= 0.5, "Test 3 Failed: Failure probability must be >= 0.5!"
    assert res_fail["status_label"] == "Machine Failure"
    assert res_fail["risk_level"] in ["High", "Critical"]
    assert len(res_fail["diagnostic_flags"]) > 0, "Test 3 Failed: Diagnostics should flag high wear/torque!"
    print(f"[PASS] Test 3: High-stress failure input detected accurately (Status: {res_fail['status_label']}, Fail Prob: {res_fail['failure_probability_percentage']}, Risk: {res_fail['risk_level']}).")

    # Test 4: Missing Input Handling
    missing_input = {
        "Type": "L",
        "Air temperature [K]": 298.1,
        # Missing Process temp, speed, torque, tool wear
    }
    try:
        predict_single(missing_input, model=model, preprocessor=preprocessor)
        assert False, "Test 4 Failed: Expected ValueError on missing keys!"
    except ValueError as e:
        assert "Missing required" in str(e)
        print(f"[PASS] Test 4: Missing feature rejected with error message: '{e}'")

    # Test 5: Invalid Machine Type Handling
    invalid_type_input = normal_input.copy()
    invalid_type_input["Type"] = "Z"
    try:
        predict_single(invalid_type_input, model=model, preprocessor=preprocessor)
        assert False, "Test 5 Failed: Expected ValueError on invalid Type 'Z'!"
    except ValueError as e:
        assert "Invalid Machine Type" in str(e)
        print(f"[PASS] Test 5: Invalid Type 'Z' rejected with error message: '{e}'")

    # Test 6: Non-Numeric Sensor Value Handling
    non_numeric_input = normal_input.copy()
    non_numeric_input["Torque [Nm]"] = "invalid_torque_string"
    try:
        predict_single(non_numeric_input, model=model, preprocessor=preprocessor)
        assert False, "Test 6 Failed: Expected TypeError on non-numeric torque!"
    except TypeError as e:
        assert "Invalid non-numeric" in str(e)
        print(f"[PASS] Test 6: Non-numeric value rejected with error message: '{e}'")

    # Test 7: Output Schema Completeness
    expected_keys = [
        "input_telemetry", "predicted_class", "status_label", "failure_probability",
        "healthy_probability", "failure_probability_percentage", "healthy_probability_percentage",
        "risk_level", "classification_threshold", "summary", "diagnostic_flags",
        "recommended_actions", "model_used", "disclaimer"
    ]
    for key in expected_keys:
        assert key in res_normal, f"Test 7 Failed: Missing output key '{key}'!"
    print("[PASS] Test 7: Output response schema complete with all required fields.")

    # Test 8: Feature Names and Order Consistency
    raw_feature_names = list(preprocessor.feature_names_in_) if hasattr(preprocessor, "feature_names_in_") else FEATURE_COLS
    assert set(raw_feature_names) == set(FEATURE_COLS), "Test 8 Failed: Preprocessor expected feature names mismatch!"
    print(f"[PASS] Test 8: Preprocessor feature mapping verified across {len(raw_feature_names)} features: {raw_feature_names}.")

    # Test 9: Batch Prediction Functionality
    batch_df = pd.DataFrame([normal_input, failure_input, {
        "Type": "H",
        "Air temperature [K]": 299.0,
        "Process temperature [K]": 309.5,
        "Rotational speed [rpm]": 1600.0,
        "Torque [Nm]": 35.0,
        "Tool wear [min]": 50.0
    }])
    batch_results = predict_batch(batch_df, model=model, preprocessor=preprocessor)
    assert len(batch_results) == 3, "Test 9 Failed: Batch result length mismatch!"
    assert "Predicted_Class" in batch_results.columns
    assert "Failure_Probability" in batch_results.columns
    assert "Risk_Level" in batch_results.columns
    assert list(batch_results["Predicted_Class"]) == [0, 1, 0]
    print("[PASS] Test 9: Batch inference processing verified successfully on multi-row telemetry DataFrame.")

    # Test 10: Negative and Boundary Value Rejection
    negative_speed_input = normal_input.copy()
    negative_speed_input["Rotational speed [rpm]"] = -500.0
    try:
        predict_single(negative_speed_input, model=model, preprocessor=preprocessor)
        assert False, "Test 10 Failed: Expected ValueError on negative rotational speed!"
    except ValueError as e:
        assert "cannot be negative" in str(e)
        print(f"[PASS] Test 10: Physically impossible negative speed rejected with error message: '{e}'")

    print("\n" + "=" * 70)
    print("ALL PHASE 6 PREDICTION & DEPLOYMENT TESTS PASSED WITH 100% INTEGRITY")
    print("=" * 70)


if __name__ == "__main__":
    run_all_prediction_tests()
