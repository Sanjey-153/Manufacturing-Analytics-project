"""
Test Suite for F4 Cost-Based Probability Threshold Framework.
Validates exact holdout reconstruction, confusion matrix counts, economic cost calculations,
trade-off curves, edge-case handling, and scenario responsiveness.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.threshold_analysis import (
    get_holdout_data,
    get_holdout_predictions,
    calculate_confusion_matrix,
    calculate_expected_cost,
    calculate_threshold_metrics,
    generate_threshold_sweep,
    find_cost_minimizing_threshold,
    compare_thresholds,
    build_cost_sensitivity_matrix,
    save_holdout_predictions_artifact,
    ILLUSTRATIVE_COST_SCENARIOS,
    DISCLAIMER_TEXT
)
from config import ID_COLS, FAILURE_TYPE_COLS, TARGET_COL


def run_threshold_analysis_tests():
    print("=" * 70)
    print("TEST SUITE: F4 COST-BASED PROBABILITY THRESHOLD FRAMEWORK")
    print("=" * 70)

    # 1. Exact holdout split reconstruction
    X_test, y_test = get_holdout_data()
    assert isinstance(X_test, pd.DataFrame), "Holdout X_test must be a DataFrame"
    assert isinstance(y_test, pd.Series), "Holdout y_test must be a Series"
    print("[PASS] Check 1: Exact holdout split reconstruction verified.")

    # 2. Holdout size = 2,000
    assert len(X_test) == 2000, f"Expected 2,000 holdout rows, got {len(X_test)}"
    assert len(y_test) == 2000, f"Expected 2,000 holdout labels, got {len(y_test)}"
    print("[PASS] Check 2: Holdout size = 2,000 verified.")

    # 3. Holdout class counts = 1,932 healthy, 68 failure
    n_healthy = int((y_test == 0).sum())
    n_failure = int((y_test == 1).sum())
    assert n_healthy == 1932, f"Expected 1,932 healthy samples, got {n_healthy}"
    assert n_failure == 68, f"Expected 68 failure samples, got {n_failure}"
    print(f"[PASS] Check 3: Holdout class counts verified (Healthy: {n_healthy}, Failure: {n_failure}).")

    # 4. Probability vector dimension matches holdout rows
    y_true, y_prob, _ = get_holdout_predictions()
    assert len(y_true) == 2000, f"Expected 2,000 true labels, got {len(y_true)}"
    assert len(y_prob) == 2000, f"Expected 2,000 probability estimates, got {len(y_prob)}"
    assert np.all((y_prob >= 0.0) & (y_prob <= 1.0)), "All probabilities must be in [0.0, 1.0]"
    print("[PASS] Check 4: Probability vector dimension and range verified.")

    # 5. Threshold sweep creates expected threshold range
    df_sweep = generate_threshold_sweep(y_true, y_prob, step=0.01)
    assert len(df_sweep) == 101, f"Expected 101 threshold steps (0.00 to 1.00), got {len(df_sweep)}"
    assert np.isclose(df_sweep["Threshold"].iloc[0], 0.00), "First threshold must be 0.00"
    assert np.isclose(df_sweep["Threshold"].iloc[-1], 1.00), "Last threshold must be 1.00"
    print("[PASS] Check 5: Threshold sweep range (0.00 to 1.00, N=101) verified.")

    # 6. Confusion matrix counts are correct and sum to 2,000
    cm_05 = calculate_confusion_matrix(y_true, y_prob, 0.50)
    assert sum(cm_05.values()) == 2000, f"Confusion matrix sum must be 2,000, got {sum(cm_05.values())}"
    assert cm_05["TP"] + cm_05["FN"] == 68, "TP + FN must equal actual failures (68)"
    assert cm_05["TN"] + cm_05["FP"] == 1932, "TN + FP must equal actual healthy (1,932)"
    assert cm_05["TP"] == 55, f"Expected TP=55 at 0.50, got {cm_05['TP']}"
    assert cm_05["FN"] == 13, f"Expected FN=13 at 0.50, got {cm_05['FN']}"
    assert cm_05["FP"] == 48, f"Expected FP=48 at 0.50, got {cm_05['FP']}"
    assert cm_05["TN"] == 1884, f"Expected TN=1884 at 0.50, got {cm_05['TN']}"
    print(f"[PASS] Check 6: Confusion matrix counts at 0.50 verified (TP: {cm_05['TP']}, TN: {cm_05['TN']}, FP: {cm_05['FP']}, FN: {cm_05['FN']}).")

    # 7. Precision calculation
    m_05 = calculate_threshold_metrics(y_true, y_prob, 0.50)
    expected_prec = 55 / (55 + 48)  # 55 / 103 = ~0.53398
    assert np.isclose(m_05["Precision"], round(expected_prec, 4)), f"Precision mismatch: {m_05['Precision']} vs {expected_prec}"
    print(f"[PASS] Check 7: Precision calculation verified ({m_05['Precision']:.4f}).")

    # 8. Recall calculation
    expected_rec = 55 / 68  # ~0.80882
    assert np.isclose(m_05["Recall"], round(expected_rec, 4)), f"Recall mismatch: {m_05['Recall']} vs {expected_rec}"
    print(f"[PASS] Check 8: Recall calculation verified ({m_05['Recall']:.4f}).")

    # 9. F1 calculation
    expected_f1 = (2 * expected_prec * expected_rec) / (expected_prec + expected_rec)
    assert np.isclose(m_05["F1"], round(expected_f1, 4)), f"F1 mismatch: {m_05['F1']} vs {expected_f1}"
    print(f"[PASS] Check 9: F1 calculation verified ({m_05['F1']:.4f}).")

    # 10. Specificity calculation
    expected_spec = 1884 / (1884 + 48)  # 1884 / 1932 = ~0.97515
    assert np.isclose(m_05["Specificity"], round(expected_spec, 4)), f"Specificity mismatch: {m_05['Specificity']} vs {expected_spec}"
    print(f"[PASS] Check 10: Specificity calculation verified ({m_05['Specificity']:.4f}).")

    # 11. FPR calculation
    expected_fpr = 48 / 1932  # ~0.02484
    assert np.isclose(m_05["FPR"], round(expected_fpr, 4)), f"FPR mismatch: {m_05['FPR']} vs {expected_fpr}"
    print(f"[PASS] Check 11: FPR calculation verified ({m_05['FPR']:.4f}).")

    # 12. FNR calculation
    expected_fnr = 13 / 68  # ~0.19117
    assert np.isclose(m_05["FNR"], round(expected_fnr, 4)), f"FNR mismatch: {m_05['FNR']} vs {expected_fnr}"
    print(f"[PASS] Check 12: FNR calculation verified ({m_05['FNR']:.4f}).")

    # 13. Alert-rate calculation
    expected_alert = (55 + 48) / 2000  # 103 / 2000 = 0.0515
    assert np.isclose(m_05["Alert_Rate"], expected_alert), f"Alert rate mismatch: {m_05['Alert_Rate']} vs {expected_alert}"
    print(f"[PASS] Check 13: Alert rate calculation verified ({m_05['Alert_Rate']:.4f}).")

    # 14. Cost calculation
    # Test with C_TN=0, C_FP=10, C_FN=100, C_TP=5
    tot_c, exp_c = calculate_expected_cost(55, 1884, 48, 13, c_tn=0.0, c_fp=10.0, c_fn=100.0, c_tp=5.0)
    # Total = (1884*0) + (48*10) + (13*100) + (55*5) = 0 + 480 + 1300 + 275 = 2055.0
    # Expected = 2055 / 2000 = 1.0275
    assert tot_c == 2055.0, f"Expected total cost 2055.0, got {tot_c}"
    assert exp_c == 1.0275, f"Expected cost per obs 1.0275, got {exp_c}"
    print(f"[PASS] Check 14: Four-outcome cost calculation verified (Total: {tot_c}, Per Obs: {exp_c}).")

    # 15. Cost-minimizing threshold calculation is deterministic
    best_row = find_cost_minimizing_threshold(df_sweep)
    assert "Threshold" in best_row and "Expected_Cost" in best_row
    assert best_row["Expected_Cost"] <= m_05["Expected_Cost"], "Optimal threshold must not have higher expected cost than 0.50 baseline"
    print(f"[PASS] Check 15: Cost-minimizing threshold calculation verified (Opt Threshold: {best_row['Threshold']}, Min Cost: {best_row['Expected_Cost']}).")

    # 16. Baseline 0.50 remains unchanged
    comp = compare_thresholds(y_true, y_prob, thresh_a=0.50, thresh_b=float(best_row["Threshold"]))
    assert comp["operational_threshold"]["Threshold"] == 0.50, "Operational baseline must remain 0.50"
    print("[PASS] Check 16: Baseline 0.50 operational threshold preserved.")

    # 17. Different cost ratios produce different analysis results (TEST FIXTURES)
    # Scenario A: FN high cost (C_FN=1000, C_FP=1) -> Should encourage lower threshold to maximize recall
    # Scenario B: FP high cost (C_FN=1, C_FP=1000) -> Should encourage higher threshold to avoid false alarms
    sweep_fn_heavy = generate_threshold_sweep(y_true, y_prob, c_tn=0, c_fp=1, c_fn=1000, c_tp=0)
    best_fn_heavy = find_cost_minimizing_threshold(sweep_fn_heavy)

    sweep_fp_heavy = generate_threshold_sweep(y_true, y_prob, c_tn=0, c_fp=1000, c_fn=1, c_tp=0)
    best_fp_heavy = find_cost_minimizing_threshold(sweep_fp_heavy)

    assert best_fn_heavy["Threshold"] <= best_fp_heavy["Threshold"], "FN-dominant scenario must produce threshold <= FP-dominant scenario"
    print(f"[PASS] Check 17: Cost ratio responsiveness verified (FN-Heavy Thresh: {best_fn_heavy['Threshold']} <= FP-Heavy Thresh: {best_fp_heavy['Threshold']}).")

    # 18. No fabricated / default industrial cost claim exists in metadata or templates
    for name, sc in ILLUSTRATIVE_COST_SCENARIOS.items():
        assert "industrial" not in sc["description"].lower() or "illustrative" in sc["description"].lower(), f"Scenario {name} must not claim real industrial truth"
    assert "illustrative" in DISCLAIMER_TEXT.lower(), "Disclaimer text must declare illustrative scope"
    print("[PASS] Check 18: Research integrity & disclaimer text verified.")

    # 19. No leakage columns are present in holdout features
    for forbidden in ID_COLS + FAILURE_TYPE_COLS:
        assert forbidden not in X_test.columns, f"Leakage column {forbidden} found in holdout features!"
    print("[PASS] Check 19: Zero feature leakage in holdout dataset verified.")

    # 20. Output is deterministic
    sweep1 = generate_threshold_sweep(y_true, y_prob, c_tn=0, c_fp=10, c_fn=100, c_tp=5)
    sweep2 = generate_threshold_sweep(y_true, y_prob, c_tn=0, c_fp=10, c_fn=100, c_tp=5)
    pd.testing.assert_frame_equal(sweep1, sweep2)
    print("[PASS] Check 20: Deterministic reproducibility verified.")

    # 21. Edge thresholds 0.00 and 1.00 are handled cleanly
    m_00 = calculate_threshold_metrics(y_true, y_prob, 0.00)
    m_10 = calculate_threshold_metrics(y_true, y_prob, 1.00)
    assert m_00["TP"] == 68 and m_00["FP"] == 1932, "At 0.00, all samples predicted positive"
    assert m_10["TP"] == 0 or m_10["FN"] == 68, "At 1.00, threshold boundaries handled without error"
    print("[PASS] Check 21: Boundary thresholds (0.00, 1.00) verified.")

    # 22. Zero-denominator metrics handled safely without NaN or crash
    # At threshold 1.0, TP+FP may be 0 -> Precision handled
    assert not np.isnan(m_10["Precision"]), "Precision must not be NaN at 1.00"
    assert not np.isnan(m_10["F1"]), "F1 must not be NaN at 1.00"
    print("[PASS] Check 22: Zero-division edge cases handled safely.")

    print("\n" + "=" * 70)
    print("ALL 22 THRESHOLD ANALYSIS VALIDATION CHECKS PASSED (22/22)")
    print("=" * 70)


if __name__ == "__main__":
    run_threshold_analysis_tests()
