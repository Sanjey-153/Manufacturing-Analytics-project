"""
Comprehensive Test Suite for SHAP Explainability Engine (Future Work F2).
Tests SHAP TreeExplainer initialization, transformed feature dimensions, feature name mapping,
local attributions for healthy and failure cases, probability consistency with authoritative prediction engine,
direction consistency, finite values, and absence of target/leakage features.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
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
    predict_single,
    VALID_MACHINE_TYPES
)
from src.explainability import (
    get_tree_explainer,
    explain_prediction,
    compute_global_shap_summary,
    get_transformed_feature_names,
    TRANSFORMED_TO_DISPLAY_NAMES
)


def run_explainability_tests():
    print("=" * 70)
    print("RUNNING FUTURE WORK F2 SHAP EXPLAINABILITY TEST SUITE")
    print("=" * 70)

    # Test 1: Explainability module imports cleanly
    assert get_tree_explainer is not None, "Test 1 Failed: get_tree_explainer is None!"
    assert explain_prediction is not None, "Test 1 Failed: explain_prediction is None!"
    assert compute_global_shap_summary is not None, "Test 1 Failed: compute_global_shap_summary is None!"
    print("[PASS] Test 1: Explainability module successfully imported with core functions.")

    # Test 2: Primary LightGBM model artifact loads
    lgb_path = MODELS_DIR / "lightgbm.joblib"
    assert lgb_path.exists(), f"Test 2 Failed: LightGBM model artifact not found at {lgb_path}!"
    model = joblib.load(lgb_path)
    assert model is not None, "Test 2 Failed: Model loaded as None!"
    assert hasattr(model, "predict_proba"), "Test 2 Failed: Loaded model lacks predict_proba method!"
    print("[PASS] Test 2: Authoritative LightGBM model artifact loaded successfully.")

    # Test 3: Preprocessor artifact loads and has valid transformers
    assert PREPROCESSOR_PATH.exists(), f"Test 3 Failed: Preprocessor artifact not found at {PREPROCESSOR_PATH}!"
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    assert preprocessor is not None, "Test 3 Failed: Preprocessor loaded as None!"
    assert hasattr(preprocessor, "transform"), "Test 3 Failed: Preprocessor lacks transform method!"
    print("[PASS] Test 3: Authoritative preprocessor artifact loaded successfully.")

    # Test 4: SHAP TreeExplainer initializes and caches
    explainer1 = get_tree_explainer(model)
    explainer2 = get_tree_explainer(model)
    assert explainer1 is not None, "Test 4 Failed: TreeExplainer initialized as None!"
    assert explainer1 is explainer2, "Test 4 Failed: TreeExplainer caching returned different instances!"
    print("[PASS] Test 4: SHAP TreeExplainer initialized and successfully cached.")

    # Test 5: Single valid input generates complete structured SHAP output
    nominal_input = {
        "Type": "M",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551.0,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0.0
    }
    shap_res = explain_prediction(nominal_input, model=model, preprocessor=preprocessor, threshold=0.50)
    assert isinstance(shap_res, dict), "Test 5 Failed: explain_prediction did not return a dict!"
    required_keys = [
        "predicted_class", "failure_probability", "risk_tier", "base_value",
        "feature_contributions", "top_contributions", "transformed_feature_names",
        "shap_values", "timestamp"
    ]
    for k in required_keys:
        assert k in shap_res, f"Test 5 Failed: Missing key '{k}' in SHAP explanation output!"
    print(f"[PASS] Test 5: Single valid input generated complete SHAP explanation structure ({len(shap_res['feature_contributions'])} features).")

    # Test 6: SHAP feature dimension matches transformed feature dimension
    feature_names = get_transformed_feature_names(preprocessor)
    assert len(shap_res["shap_values"]) == len(feature_names), (
        f"Test 6 Failed: SHAP values dimension ({len(shap_res['shap_values'])}) != "
        f"transformed features dimension ({len(feature_names)})!"
    )
    assert len(feature_names) == 8, f"Test 6 Failed: Expected 8 transformed features, got {len(feature_names)}!"
    print(f"[PASS] Test 6: SHAP feature dimension ({len(shap_res['shap_values'])}) matches preprocessed feature space exactly (8 features).")

    # Test 7: Feature-name dimension matches SHAP dimension and display mapping is comprehensive
    for feat_dict in shap_res["feature_contributions"]:
        assert "feature" in feat_dict, "Test 7 Failed: Missing 'feature' in contribution row!"
        assert "display_name" in feat_dict, "Test 7 Failed: Missing 'display_name' in contribution row!"
        assert "value" in feat_dict, "Test 7 Failed: Missing 'value' in contribution row!"
        assert "contribution" in feat_dict, "Test 7 Failed: Missing 'contribution' in contribution row!"
        assert "direction" in feat_dict, "Test 7 Failed: Missing 'direction' in contribution row!"
        assert feat_dict["feature"] in TRANSFORMED_TO_DISPLAY_NAMES, f"Test 7 Failed: Feature {feat_dict['feature']} not mapped in dictionary!"
    print("[PASS] Test 7: Feature names and human-readable display mappings verified for all 8 transformed dimensions.")

    # Test 8: No forbidden leakage columns or identifiers are present in SHAP explanation
    forbidden_cols = ["UDI", "Product ID", "Target", "Failure Type", "TWF", "HDF", "PWF", "OSF", "RNF"]
    for feat_dict in shap_res["feature_contributions"]:
        for forbidden in forbidden_cols:
            assert forbidden != feat_dict["feature"], f"Test 8 Failed: Forbidden leakage column '{forbidden}' found in explanation!"
            assert forbidden != feat_dict["display_name"], f"Test 8 Failed: Forbidden leakage display name '{forbidden}' found!"
    print("[PASS] Test 8: Zero leakage verified (no target, failure mode, or identifier features present).")

    # Test 9: Both healthy/low-risk and failure/high-risk examples can be explained cleanly
    high_risk_input = {
        "Type": "L",
        "Air temperature [K]": 302.5,
        "Process temperature [K]": 310.8,
        "Rotational speed [rpm]": 1320.0,
        "Torque [Nm]": 68.4,
        "Tool wear [min]": 215.0
    }
    shap_fail = explain_prediction(high_risk_input, model=model, preprocessor=preprocessor, threshold=0.50)
    assert shap_res["predicted_class"] == 0, "Test 9 Failed: Healthy case did not predict class 0!"
    assert shap_fail["predicted_class"] == 1, "Test 9 Failed: High-risk failure case did not predict class 1!"
    assert shap_fail["failure_probability"] > 0.50, "Test 9 Failed: Failure probability must be > 0.50 for high-risk input!"
    print(f"[PASS] Test 9: Successfully explained both Healthy (Prob: {shap_res['failure_probability']*100:.2f}%) and High-Risk Failure (Prob: {shap_fail['failure_probability']*100:.2f}%).")

    # Test 10: SHAP output contains strictly finite numeric values (no NaN or Inf)
    for res_obj in [shap_res, shap_fail]:
        for val in res_obj["shap_values"]:
            assert np.isfinite(val), f"Test 10 Failed: Non-finite SHAP value encountered: {val}!"
        assert np.isfinite(res_obj["base_value"]), f"Test 10 Failed: Non-finite base_value: {res_obj['base_value']}!"
    print("[PASS] Test 10: All computed SHAP attributions and baseline values are strictly finite real numbers.")

    # Test 11: Top contribution ordering is strictly based on descending absolute magnitude |SHAP|
    for res_obj in [shap_res, shap_fail]:
        contributions = res_obj["feature_contributions"]
        abs_vals = [c["abs_contribution"] for c in contributions]
        assert abs_vals == sorted(abs_vals, reverse=True), "Test 11 Failed: Contributions are not sorted by descending absolute magnitude!"
        assert len(res_obj["top_contributions"]) == 5, f"Test 11 Failed: Expected top 5 contributions, got {len(res_obj['top_contributions'])}!"
    print("[PASS] Test 11: Feature contribution ordering verified strictly by descending absolute magnitude |SHAP|.")

    # Test 12: Direction labeling is strictly consistent with SHAP sign
    for res_obj in [shap_res, shap_fail]:
        for c in res_obj["feature_contributions"]:
            if c["contribution"] > 0:
                assert c["direction"] == "Increases Failure Risk", f"Test 12 Failed: Positive contribution got direction '{c['direction']}'!"
            elif c["contribution"] < 0:
                assert c["direction"] == "Reduces Failure Risk", f"Test 12 Failed: Negative contribution got direction '{c['direction']}'!"
            else:
                assert c["direction"] == "Neutral", f"Test 12 Failed: Zero contribution got direction '{c['direction']}'!"
    print("[PASS] Test 12: Direction semantics verified (+ pushes higher failure risk, - pushes lower failure risk).")

    # Test 13: Probability and class from explainability workflow matches authoritative predict_single() exactly
    for test_payload in [nominal_input, high_risk_input]:
        pred_auth = predict_single(test_payload, model=model, preprocessor=preprocessor, threshold=0.50)
        shap_expl = explain_prediction(test_payload, model=model, preprocessor=preprocessor, threshold=0.50)
        assert pred_auth["predicted_class"] == shap_expl["predicted_class"], (
            f"Test 13 Failed: Class mismatch between prediction engine ({pred_auth['predicted_class']}) "
            f"and explainability ({shap_expl['predicted_class']})!"
        )
        assert abs(pred_auth["failure_probability"] - shap_expl["failure_probability"]) < 1e-6, (
            f"Test 13 Failed: Probability mismatch between prediction engine ({pred_auth['failure_probability']}) "
            f"and explainability ({shap_expl['failure_probability']})!"
        )
    print("[PASS] Test 13: Complete probability and predicted class consistency verified against authoritative predict_single().")

    # Test 14: Global SHAP population summary executes cleanly
    global_df = compute_global_shap_summary(model=model, preprocessor=preprocessor, n_samples=30)
    assert isinstance(global_df, pd.DataFrame), "Test 14 Failed: compute_global_shap_summary did not return a DataFrame!"
    assert len(global_df) == 8, f"Test 14 Failed: Expected 8 features in global summary, got {len(global_df)}!"
    assert "mean_abs_shap" in global_df.columns, "Test 14 Failed: Missing 'mean_abs_shap' column in global summary!"
    assert global_df["mean_abs_shap"].iloc[0] >= global_df["mean_abs_shap"].iloc[-1], "Test 14 Failed: Global summary not sorted descending!"
    print(f"[PASS] Test 14: Global SHAP population summary computed cleanly across benchmark instances (Top global: {global_df['display_name'].iloc[0]}).")

    print("\n" + "=" * 70)
    print("ALL 14 SHAP EXPLAINABILITY TESTS PASSED WITH ZERO REGRESSIONS")
    print("=" * 70)
    return True


if __name__ == "__main__":
    run_explainability_tests()
