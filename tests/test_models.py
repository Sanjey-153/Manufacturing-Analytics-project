"""
Test Suite for ML Model Development and Validation (Phase 4).
Verifies serialization, test-set integrity, reproducibility, and metric calculations.
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
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    METRICS_DIR,
    TARGET_COL
)
from src.train_models import evaluate_model


def test_models_and_evaluation():
    print("=" * 70)
    print("RUNNING PHASE 4 VALIDATION TEST SUITE")
    print("=" * 70)

    # 1. Verify model files exist
    model_names = ["decision_tree", "random_forest", "xgboost", "lightgbm"]
    for m in model_names:
        m_path = MODELS_DIR / f"{m}.joblib"
        assert m_path.exists(), f"Model file missing: {m_path}"
    print("[PASS] Check 1: All 4 model files exist in models/.")

    # 2. Load test set and verify dimensions and class balance
    X_test = pd.read_csv(PROCESSED_DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(PROCESSED_DATA_DIR / "y_test.csv")[TARGET_COL]

    assert len(X_test) == 2000, f"Test size is {len(X_test)}, expected 2000!"
    assert len(y_test) == 2000, f"Test target size is {len(y_test)}, expected 2000!"
    assert int(y_test.sum()) == 68, f"Test failure count is {y_test.sum()}, expected 68!"
    assert int((y_test == 0).sum()) == 1932, f"Test healthy count is {(y_test == 0).sum()}, expected 1932!"
    print("[PASS] Check 2: Test set integrity confirmed (2,000 rows, 1,932 healthy, 68 failures).")

    # 3. Reload each model and verify predictions match recorded performance table
    perf_df = pd.read_csv(METRICS_DIR / "model_performance.csv")
    clean_cols = [c.replace("[", "(").replace("]", ")") for c in X_test.columns]
    X_test_clean = X_test.copy()
    X_test_clean.columns = clean_cols

    display_name_map = {
        "decision_tree": "Decision Tree",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM"
    }

    for m in model_names:
        m_path = MODELS_DIR / f"{m}.joblib"
        model = joblib.load(m_path)
        eval_res = evaluate_model(model, X_test_clean, y_test)
        disp_name = display_name_map[m]
        
        row = perf_df[perf_df["Model"] == disp_name].iloc[0]
        assert round(eval_res["accuracy"], 4) == row["Accuracy"], f"Accuracy mismatch for {disp_name}"
        assert round(eval_res["precision"], 4) == row["Precision"], f"Precision mismatch for {disp_name}"
        assert round(eval_res["recall"], 4) == row["Recall"], f"Recall mismatch for {disp_name}"
        assert round(eval_res["f1_score"], 4) == row["F1-Score"], f"F1-score mismatch for {disp_name}"
        assert eval_res["fn"] == row["False Negative"], f"FN mismatch for {disp_name}"
        assert eval_res["tp"] == row["True Positive"], f"TP mismatch for {disp_name}"
        print(f"[PASS] Check 3: Model '{disp_name}' successfully reloaded and re-evaluated with matching metrics.")

    print("\n" + "=" * 70)
    print("ALL PHASE 4 VALIDATION CHECKS PASSED WITH 100% INTEGRITY")
    print("=" * 70)


if __name__ == "__main__":
    test_models_and_evaluation()
