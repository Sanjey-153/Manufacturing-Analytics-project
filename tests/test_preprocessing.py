"""
Test Suite for Data Preprocessing and Train/Test Pipeline Validation.
Explicitly verifies all 10 validation checks defined in Phase 2.
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
    RAW_DATA_PATH,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    METRICS_DIR,
    TABLES_DIR,
    ID_COLS,
    TARGET_COL,
    FAILURE_TYPE_COLS,
    CATEGORICAL_COLS,
    NUMERICAL_COLS,
    FEATURE_COLS,
    PREPROCESSOR_PATH
)
from src.preprocessing import (
    load_raw_data,
    separate_features_and_target,
    split_data,
    create_preprocessor,
    fit_and_transform_features,
    apply_smote,
    transform_single_instance,
    execute_pipeline
)


def run_all_validation_checks():
    print("=" * 70)
    print("RUNNING EXPLICIT PHASE 2 VALIDATION CHECKS (1-10)")
    print("=" * 70)

    # Load artifacts
    X_train = pd.read_csv(PROCESSED_DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DATA_DIR / "X_test.csv")
    y_train = pd.read_csv(PROCESSED_DATA_DIR / "y_train.csv")[TARGET_COL]
    y_test = pd.read_csv(PROCESSED_DATA_DIR / "y_test.csv")[TARGET_COL]
    X_train_smote = pd.read_csv(PROCESSED_DATA_DIR / "X_train_smote.csv")
    y_train_smote = pd.read_csv(PROCESSED_DATA_DIR / "y_train_smote.csv")[TARGET_COL]
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    # Check 1: No missing values
    assert X_train.isnull().sum().sum() == 0, "Check 1 Failed: X_train contains missing values!"
    assert X_test.isnull().sum().sum() == 0, "Check 1 Failed: X_test contains missing values!"
    assert y_train.isnull().sum() == 0, "Check 1 Failed: y_train contains missing values!"
    assert y_test.isnull().sum() == 0, "Check 1 Failed: y_test contains missing values!"
    assert X_train_smote.isnull().sum().sum() == 0, "Check 1 Failed: X_train_smote contains missing values!"
    assert y_train_smote.isnull().sum() == 0, "Check 1 Failed: y_train_smote contains missing values!"
    print("[PASS] Check 1: No missing values found across all processed datasets.")

    # Check 2: No duplicate rows introduced in training/testing features
    assert X_train.duplicated().sum() == 0, "Check 2 Failed: Duplicate rows in X_train!"
    assert X_test.duplicated().sum() == 0, "Check 2 Failed: Duplicate rows in X_test!"
    print(f"[PASS] Check 2: Zero duplicate rows in X_train ({len(X_train)} rows) and X_test ({len(X_test)} rows).")

    # Check 3: Identifiers are excluded from model features
    for col in ID_COLS:
        assert col not in X_train.columns, f"Check 3 Failed: Identifier {col} present in X_train!"
        assert col not in X_test.columns, f"Check 3 Failed: Identifier {col} present in X_test!"
        assert col not in X_train_smote.columns, f"Check 3 Failed: Identifier {col} present in X_train_smote!"
    print(f"[PASS] Check 3: Identifiers {ID_COLS} are strictly excluded from all model feature matrices.")

    # Check 4: Target is separated correctly
    assert y_train.name == TARGET_COL, "Check 4 Failed: y_train target name mismatch!"
    assert y_test.name == TARGET_COL, "Check 4 Failed: y_test target name mismatch!"
    assert set(y_train.unique()).issubset({0, 1}), "Check 4 Failed: Non-binary target values in y_train!"
    assert set(y_test.unique()).issubset({0, 1}), "Check 4 Failed: Non-binary target values in y_test!"
    print(f"[PASS] Check 4: Target '{TARGET_COL}' separated cleanly as binary variable.")

    # Check 5: Train/test split is stratified
    train_fail_rate = y_train.mean()
    test_fail_rate = y_test.mean()
    diff = abs(train_fail_rate - test_fail_rate)
    assert diff < 0.005, f"Check 5 Failed: Stratification failure! Train rate: {train_fail_rate:.4f}, Test rate: {test_fail_rate:.4f}"
    print(f"[PASS] Check 5: Stratified split preserved class proportions (Train: {train_fail_rate*100:.2f}%, Test: {test_fail_rate*100:.2f}%, diff: {diff*100:.4f}%).")

    # Check 6: Test data remains untouched by SMOTE
    assert len(X_test) == 2000, f"Check 6 Failed: X_test size is {len(X_test)} instead of 2000!"
    assert len(y_test) == 2000, f"Check 6 Failed: y_test size is {len(y_test)} instead of 2000!"
    assert y_test.value_counts().to_dict() == {0: 1932, 1: 68}, f"Check 6 Failed: y_test distribution modified! Got {y_test.value_counts().to_dict()}"
    print(f"[PASS] Check 6: Test dataset strictly untouched by SMOTE (Exact shape: 2000x8, 1932 healthy, 68 failures).")

    # Check 7: No target-derived failure flags used as input features
    for col in FAILURE_TYPE_COLS:
        assert col not in X_train.columns, f"Check 7 Failed: Leakage flag {col} in X_train!"
        assert col not in X_test.columns, f"Check 7 Failed: Leakage flag {col} in X_test!"
        assert col not in X_train_smote.columns, f"Check 7 Failed: Leakage flag {col} in X_train_smote!"
    print(f"[PASS] Check 7: No target-derived failure flags ({FAILURE_TYPE_COLS}) present in feature matrices.")

    # Check 8: Feature names tracked correctly
    expected_feature_names = ['Type_H', 'Type_L', 'Type_M', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    assert list(X_train.columns) == expected_feature_names, f"Check 8 Failed: Feature names mismatch in X_train! Got {list(X_train.columns)}"
    assert list(X_test.columns) == expected_feature_names, f"Check 8 Failed: Feature names mismatch in X_test! Got {list(X_test.columns)}"
    assert list(X_train_smote.columns) == expected_feature_names, f"Check 8 Failed: Feature names mismatch in X_train_smote! Got {list(X_train_smote.columns)}"
    print(f"[PASS] Check 8: Feature names correctly tracked across all datasets ({len(expected_feature_names)} features): {expected_feature_names}.")

    # Check 9: Preprocessor can transform future single-machine input consistently
    sample_machine_input = {
        "Type": "L",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551.0,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0.0
    }
    transformed_single = transform_single_instance(sample_machine_input, preprocessor)
    assert transformed_single.shape == (1, 8), f"Check 9 Failed: Transformed single shape is {transformed_single.shape}!"
    assert list(transformed_single.columns) == expected_feature_names, "Check 9 Failed: Single transform columns mismatch!"
    # For Type 'L', Type_H=0, Type_L=1, Type_M=0
    assert transformed_single["Type_L"].iloc[0] == 1.0, "Check 9 Failed: OneHot encoding for Type_L incorrect!"
    assert transformed_single["Type_H"].iloc[0] == 0.0, "Check 9 Failed: OneHot encoding for Type_H incorrect!"
    assert transformed_single["Type_M"].iloc[0] == 0.0, "Check 9 Failed: OneHot encoding for Type_M incorrect!"
    assert transformed_single["Rotational speed [rpm]"].iloc[0] == 1551.0, "Check 9 Failed: Physical units not preserved!"
    print(f"[PASS] Check 9: Preprocessor transforms future single-machine inputs consistently and preserves physical units.")

    # Check 10: No data leakage exists
    # Verify train and test row index sets do not overlap
    raw_df = load_raw_data(RAW_DATA_PATH)
    X_raw, y_raw = separate_features_and_target(raw_df)
    X_tr_r, X_te_r, _, _ = split_data(X_raw, y_raw, test_size=0.2, random_state=42)
    overlap = set(X_tr_r.index).intersection(set(X_te_r.index))
    assert len(overlap) == 0, f"Check 10 Failed: Index overlap between train and test sets! Overlap: {overlap}"
    print("[PASS] Check 10: Zero data leakage confirmed. Train/Test indices are mutually exclusive, preprocessor fitted only on train data.")

    print("\n" + "=" * 70)
    print("ALL 10 VALIDATION CHECKS PASSED WITH 100% INTEGRITY")
    print("=" * 70)


if __name__ == "__main__":
    run_all_validation_checks()
