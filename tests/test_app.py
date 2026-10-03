"""
Automated Test Suite for Streamlit Dashboard and Application Components (Phase 7).
Tests imports, artifact availability, prediction integration, schema validation, and rendering readiness.
"""
import os
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
    FIGURES_DIR,
    METRICS_DIR,
    TABLES_DIR,
    FEATURE_COLS
)
from src.prediction import (
    load_inference_pipeline,
    predict_single,
    predict_batch,
    validate_single_input
)
from src.recommendations import generate_maintenance_recommendation, determine_risk_level


def run_app_tests():
    print("=" * 70)
    print("RUNNING PHASE 7 STREAMLIT APPLICATION TEST SUITE")
    print("=" * 70)

    # Test 1: Application Module Import
    import app.app as app_module
    assert app_module is not None, "Test 1 Failed: Could not import app.app!"
    print("[PASS] Test 1: Streamlit app module (app.app) imported successfully.")

    # Test 2 & 3: Model and Preprocessor Availability and Loading
    model, preprocessor = load_inference_pipeline()
    assert model is not None, "Test 2 Failed: LightGBM model failed to load!"
    assert preprocessor is not None, "Test 3 Failed: Preprocessor failed to load!"
    print(f"[PASS] Test 2 & 3: Model ({type(model).__name__}) and Preprocessor ({type(preprocessor).__name__}) loaded from disk.")

    # Test 4: Prediction Engine Functionality
    sample_normal = {
        "Type": "M",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551.0,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0.0
    }
    pred_res = predict_single(sample_normal, model=model, preprocessor=preprocessor)
    assert pred_res["predicted_class"] == 0
    assert pred_res["status_label"] == "Healthy"
    assert pred_res["risk_level"] in ["Low", "Moderate"]
    print(f"[PASS] Test 4: Single-instance prediction engine returned expected result (Status: {pred_res['status_label']}, Prob: {pred_res['failure_probability_percentage']}).")

    # Test 5: Failure Anomaly Detection
    sample_anomaly = {
        "Type": "L",
        "Air temperature [K]": 302.5,
        "Process temperature [K]": 310.2,
        "Rotational speed [rpm]": 1200.0,
        "Torque [Nm]": 72.5,
        "Tool wear [min]": 240.0
    }
    fail_res = predict_single(sample_anomaly, model=model, preprocessor=preprocessor)
    assert fail_res["predicted_class"] == 1
    assert fail_res["status_label"] == "Machine Failure"
    assert fail_res["risk_level"] in ["High", "Critical"]
    assert len(fail_res["diagnostic_flags"]) > 0
    print(f"[PASS] Test 5: Failure anomaly detected accurately (Status: {fail_res['status_label']}, Risk: {fail_res['risk_level']}).")

    # Test 6: Input Validation and Error Handling
    invalid_inputs = [
        ({"Type": "INVALID", "Air temperature [K]": 300, "Process temperature [K]": 310, "Rotational speed [rpm]": 1500, "Torque [Nm]": 40, "Tool wear [min]": 10}, ValueError),
        ({"Type": "M", "Air temperature [K]": "abc", "Process temperature [K]": 310, "Rotational speed [rpm]": 1500, "Torque [Nm]": 40, "Tool wear [min]": 10}, TypeError),
        ({"Type": "M", "Air temperature [K]": 300, "Process temperature [K]": 310, "Rotational speed [rpm]": -100, "Torque [Nm]": 40, "Tool wear [min]": 10}, ValueError)
    ]
    for inp, expected_err in invalid_inputs:
        try:
            predict_single(inp, model=model, preprocessor=preprocessor)
            assert False, f"Test 6 Failed: Input {inp} should have raised {expected_err}!"
        except expected_err:
            pass
    print("[PASS] Test 6: All invalid input cases correctly rejected with appropriate exceptions.")

    # Test 7: Metrics and Tables Artifact Verification
    required_metrics = [
        METRICS_DIR / "final_model_comparison.csv",
        METRICS_DIR / "deployment_summary.json",
        METRICS_DIR / "stacking_summary.json",
        METRICS_DIR / "eda_summary.json"
    ]
    for m_path in required_metrics:
        assert m_path.exists(), f"Test 7 Failed: Required metrics file missing: {m_path}"
    print(f"[PASS] Test 7: All {len(required_metrics)} deployment metrics and comparison tables detected.")

    # Test 8: Required EDA and Performance Figures Verification
    required_figures = [
        "01_machine_failure_distribution.png",
        "02_machine_type_distribution.png",
        "03_sensor_correlation_heatmap.png",
        "04_failure_mode_distribution.png",
        "05_sensor_distributions.png",
        "06_features_by_failure_boxplot.png",
        "07_machine_type_vs_failure.png",
        "08_feature_interactions_vs_failure.png",
        "final_model_comparison.png",
        "feature_importance_lightgbm.png",
        "confusion_matrix_lightgbm.png",
        "confusion_matrix_stacking_ensemble.png"
    ]
    for fig_name in required_figures:
        fig_path = FIGURES_DIR / fig_name
        assert fig_path.exists(), f"Test 8 Failed: Required figure missing: {fig_path}"
    print(f"[PASS] Test 8: All {len(required_figures)} publication figures detected in outputs/figures/.")

    # Test 9: Risk Level Heuristics
    assert determine_risk_level(0.05) == "Low"
    assert determine_risk_level(0.35) == "Moderate"
    assert determine_risk_level(0.65) == "High"
    assert determine_risk_level(0.95) == "Critical"
    print("[PASS] Test 9: Risk tier heuristics verified across all probability intervals.")

    # Test 10: Batch Inference Validation
    batch_df = pd.DataFrame([sample_normal, sample_anomaly])
    batch_out = predict_batch(batch_df, model=model, preprocessor=preprocessor)
    assert len(batch_out) == 2
    assert "Predicted_Class" in batch_out.columns
    assert list(batch_out["Predicted_Class"]) == [0, 1]
    print("[PASS] Test 10: Batch inference verified with matching class outputs [0, 1].")

    print("\n" + "=" * 70)
    print("ALL PHASE 7 STREAMLIT APPLICATION TESTS PASSED WITH 100% INTEGRITY")
    print("=" * 70)


if __name__ == "__main__":
    run_app_tests()
