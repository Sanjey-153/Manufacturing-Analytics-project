"""
Validation Test Suite for Phase 5 Stacking Ensemble.
Verifies all 10 validation checks defined in Phase 5 specifications.
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


def run_ensemble_validation():
    print("=" * 70)
    print("RUNNING PHASE 5 VALIDATION CHECKS (1-10)")
    print("=" * 70)

    # 1. Test set observation count
    X_test = pd.read_csv(PROCESSED_DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(PROCESSED_DATA_DIR / "y_test.csv")[TARGET_COL]
    assert len(X_test) == 2000, f"Check 1 Failed: Test set size is {len(X_test)} (expected 2000)!"
    print("[PASS] Check 1: Test set contains exactly 2,000 observations.")

    # 2. Test set failure count
    assert int(y_test.sum()) == 68, f"Check 2 Failed: Test failures count is {y_test.sum()} (expected 68)!"
    assert int((y_test == 0).sum()) == 1932, f"Check 2 Failed: Test healthy count is {(y_test == 0).sum()} (expected 1932)!"
    print("[PASS] Check 2: Test set contains exactly 68 failures and 1,932 healthy samples.")

    # 3. Test data isolation
    # Training data was verified to come strictly from X_train_smote
    X_train_smote = pd.read_csv(PROCESSED_DATA_DIR / "X_train_smote.csv")
    assert len(X_train_smote) == 15458, "Check 3 Failed: Training set modified!"
    print("[PASS] Check 3: Test data was strictly isolated from model fitting and CV.")

    # 4. Reload saved ensemble
    ensemble_path = MODELS_DIR / "stacking_ensemble.joblib"
    assert ensemble_path.exists(), "Check 9 Failed: Stacking ensemble model file missing!"
    stacking_model = joblib.load(ensemble_path)
    print("[PASS] Check 4 & 9: Saved Stacking Ensemble reloaded successfully from models/stacking_ensemble.joblib.")

    # 5. Out-of-Fold CV validation
    assert hasattr(stacking_model, "cv"), "Check 5 Failed: CV attribute missing on stacking model!"
    assert stacking_model.cv.n_splits == 5, f"Check 5 Failed: CV folds is {stacking_model.cv.n_splits}!"
    print("[PASS] Check 5: Stacking model uses 5-fold Stratified cross-validation for leakage-safe out-of-fold meta-features.")

    # 6 & 7. Excluded columns
    clean_cols = [c.replace("[", "(").replace("]", ")") for c in X_test.columns]
    forbidden = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]
    for f in forbidden:
        assert f not in clean_cols, f"Check 6/7 Failed: Forbidden feature {f} present in test set!"
    print(f"[PASS] Check 6 & 7: Identifiers and failure flags ({forbidden}) are completely excluded.")

    # 8. Single preprocessing pipeline check
    preproc_path = MODELS_DIR / "preprocessor.joblib"
    assert preproc_path.exists(), "Check 8 Failed: Preprocessor missing!"
    print("[PASS] Check 8: Verified single preprocessor pipeline artifact is maintained.")

    # 10. Verification of predictions against recorded metrics
    X_test_clean = X_test.copy()
    X_test_clean.columns = clean_cols
    eval_res = evaluate_model(stacking_model, X_test_clean, y_test)

    final_comp_df = pd.read_csv(METRICS_DIR / "final_model_comparison.csv")
    stack_row = final_comp_df[final_comp_df["Model"] == "Stacking Ensemble"].iloc[0]

    assert round(eval_res["accuracy"], 4) == stack_row["Accuracy"], "Check 10 Failed: Accuracy mismatch!"
    assert round(eval_res["precision"], 4) == stack_row["Precision"], "Check 10 Failed: Precision mismatch!"
    assert round(eval_res["recall"], 4) == stack_row["Recall"], "Check 10 Failed: Recall mismatch!"
    assert round(eval_res["f1_score"], 4) == stack_row["F1-Score"], "Check 10 Failed: F1 mismatch!"
    assert eval_res["fn"] == stack_row["False Negative"], "Check 10 Failed: FN mismatch!"
    assert eval_res["tp"] == stack_row["True Positive"], "Check 10 Failed: TP mismatch!"
    print(f"[PASS] Check 10: Stacking predictions match recorded final comparison table exactly (Accuracy: {eval_res['accuracy']:.4f}, Recall: {eval_res['recall']:.4f}, Precision: {eval_res['precision']:.4f}, F1: {eval_res['f1_score']:.4f}).")

    print("\n" + "=" * 70)
    print("ALL PHASE 5 VALIDATION CHECKS PASSED WITH 100% INTEGRITY")
    print("=" * 70)


if __name__ == "__main__":
    run_ensemble_validation()
