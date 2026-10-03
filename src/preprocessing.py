"""
Preprocessing and Train/Test Pipeline Module for Manufacturing Analytics Project.
Handles feature/target separation, stratified splitting, leak-free preprocessing,
SMOTE oversampling (training only), artifact saving, and single-instance transformation.
"""
import os
import sys
import json
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    METRICS_DIR,
    TABLES_DIR,
    RANDOM_STATE,
    ID_COLS,
    TARGET_COL,
    FAILURE_TYPE_COLS,
    CATEGORICAL_COLS,
    NUMERICAL_COLS,
    FEATURE_COLS,
    PREPROCESSOR_PATH
)
from src.data_loader import load_raw_data, verify_dataset_integrity


def create_preprocessor(
    categorical_cols: Optional[List[str]] = None,
    numerical_cols: Optional[List[str]] = None
) -> ColumnTransformer:
    """
    Builds a scikit-learn ColumnTransformer that encodes categorical variables
    with OneHotEncoder and passes through numerical features without scaling
    (preserving original physical/engineering units).
    """
    if categorical_cols is None:
        categorical_cols = CATEGORICAL_COLS
    if numerical_cols is None:
        numerical_cols = NUMERICAL_COLS

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    sparse_output=False,
                    handle_unknown="ignore"
                ),
                categorical_cols,
            ),
            ("num", "passthrough", numerical_cols),
        ],
        verbose_feature_names_out=False
    )
    return preprocessor


def separate_features_and_target(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    target_col: str = TARGET_COL,
    exclude_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extracts predictive features and target variable from DataFrame.
    Strictly verifies that identifiers (UDI, Product ID) and failure mode flags
    (TWF, HDF, PWF, OSF, RNF) are excluded from the feature matrix to prevent data leakage.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS
    if exclude_cols is None:
        exclude_cols = ID_COLS + FAILURE_TYPE_COLS

    # Verify no excluded column is present in feature_cols
    forbidden_present = [col for col in exclude_cols if col in feature_cols]
    if forbidden_present:
        raise ValueError(
            f"Data leakage risk! Excluded columns found in feature list: {forbidden_present}"
        )

    # Validate target column exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    # Validate all feature columns exist
    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        raise ValueError(f"Feature columns missing from dataset: {missing_features}")

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Performs a reproducible stratified train/test split.
    Splits before any preprocessing or oversampling to avoid data leakage.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def fit_and_transform_features(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, ColumnTransformer, List[str]]:
    """
    Fits the preprocessor strictly on X_train, then transforms both X_train and X_test.
    Returns transformed DataFrames with correct feature names and preserved indices.
    """
    # Fit strictly on training data
    preprocessor.fit(X_train)

    feature_names = list(preprocessor.get_feature_names_out())

    # Transform training and test data
    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # Convert to DataFrames
    X_train_df = pd.DataFrame(
        X_train_trans,
        columns=feature_names,
        index=X_train.index
    )
    X_test_df = pd.DataFrame(
        X_test_trans,
        columns=feature_names,
        index=X_test.index
    )

    return X_train_df, X_test_df, preprocessor, feature_names


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Applies Synthetic Minority Over-sampling Technique (SMOTE) strictly on training data.
    Test set is never modified.
    """
    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    X_train_res_df = pd.DataFrame(
        X_train_res,
        columns=X_train.columns
    )
    y_train_res_series = pd.Series(
        y_train_res,
        name=y_train.name
    )

    return X_train_res_df, y_train_res_series


def transform_single_instance(
    instance_dict: Dict[str, Any],
    preprocessor: ColumnTransformer
) -> pd.DataFrame:
    """
    Transforms a single machine operating input dictionary into the preprocessed feature representation.
    Ensures future single-machine or batch inference consistency.
    """
    df_instance = pd.DataFrame([instance_dict])
    feature_names = list(preprocessor.get_feature_names_out())
    transformed = preprocessor.transform(df_instance)
    return pd.DataFrame(transformed, columns=feature_names)


def execute_pipeline() -> Dict[str, Any]:
    """
    Executes the complete Phase 2 data preprocessing and train/test pipeline:
    1. Loads raw dataset and validates data quality.
    2. Separates predictive features and target (strictly excluding IDs & failure flags).
    3. Performs stratified 80/20 train/test split.
    4. Fits ColumnTransformer strictly on X_train and transforms X_train and X_test.
    5. Applies SMOTE to X_train_processed to create balanced training data.
    6. Saves all processed datasets and preprocessor artifact.
    7. Generates summary metrics (JSON) and tables (CSV).
    8. Performs rigorous validation checks.
    """
    print("=" * 70)
    print("PHASE 2: DATA PREPROCESSING & TRAIN/TEST PIPELINE EXECUTION")
    print("=" * 70)

    # Step 1 & 2: Load raw data & verify quality
    df_raw = load_raw_data(RAW_DATA_PATH)
    integrity = verify_dataset_integrity(df_raw)
    print(f"Loaded raw dataset from: {RAW_DATA_PATH}")
    print(f"Raw shape: {df_raw.shape}")
    print(f"Integrity check: 0 nulls = {integrity['total_nulls'] == 0}, 0 duplicates = {integrity['duplicate_count'] == 0}")

    # Separate features and target
    X, y = separate_features_and_target(df_raw)
    print(f"Feature matrix shape: {X.shape}, Target shape: {y.shape}")
    print(f"Selected features: {list(X.columns)}")

    # Step 3: Stratified Train/Test split
    X_train_raw, X_test_raw, y_train, y_test = split_data(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # Split statistics
    train_total = len(y_train)
    test_total = len(y_test)
    train_failures = int(y_train.sum())
    train_healthy = int((y_train == 0).sum())
    test_failures = int(y_test.sum())
    test_healthy = int((y_test == 0).sum())
    train_fail_pct = (train_failures / train_total) * 100
    test_fail_pct = (test_failures / test_total) * 100

    print("\n--- Stratified Split Summary ---")
    print(f"Training samples: {train_total} (Healthy: {train_healthy}, Failures: {train_failures}, Failure Rate: {train_fail_pct:.2f}%)")
    print(f"Test samples:     {test_total} (Healthy: {test_healthy}, Failures: {test_failures}, Failure Rate: {test_fail_pct:.2f}%)")

    # Step 4: Leak-free Preprocessing (Fit on Train only)
    preprocessor = create_preprocessor()
    X_train_proc, X_test_proc, fitted_preprocessor, feature_names = fit_and_transform_features(
        preprocessor, X_train_raw, X_test_raw
    )
    print("\n--- Preprocessing Summary ---")
    print(f"Encoded feature names ({len(feature_names)}): {feature_names}")
    print(f"X_train_proc shape: {X_train_proc.shape}")
    print(f"X_test_proc shape:  {X_test_proc.shape}")

    # Step 5: SMOTE on Training Set only
    X_train_smote, y_train_smote = apply_smote(X_train_proc, y_train, random_state=RANDOM_STATE)
    smote_total = len(y_train_smote)
    smote_failures = int(y_train_smote.sum())
    smote_healthy = int((y_train_smote == 0).sum())
    smote_fail_pct = (smote_failures / smote_total) * 100

    print("\n--- SMOTE Oversampling Summary (Training Set Only) ---")
    print(f"Post-SMOTE Training samples: {smote_total} (Healthy: {smote_healthy}, Failures: {smote_failures}, Failure Rate: {smote_fail_pct:.2f}%)")
    print(f"Test Set untouched samples:  {len(y_test)} (Healthy: {test_healthy}, Failures: {test_failures}, Failure Rate: {test_fail_pct:.2f}%)")

    # Step 6: Save Artifacts
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Save datasets
    X_train_path = PROCESSED_DATA_DIR / "X_train.csv"
    X_test_path = PROCESSED_DATA_DIR / "X_test.csv"
    y_train_path = PROCESSED_DATA_DIR / "y_train.csv"
    y_test_path = PROCESSED_DATA_DIR / "y_test.csv"
    X_train_smote_path = PROCESSED_DATA_DIR / "X_train_smote.csv"
    y_train_smote_path = PROCESSED_DATA_DIR / "y_train_smote.csv"

    X_train_proc.to_csv(X_train_path, index=False)
    X_test_proc.to_csv(X_test_path, index=False)
    y_train.to_frame(name=TARGET_COL).to_csv(y_train_path, index=False)
    y_test.to_frame(name=TARGET_COL).to_csv(y_test_path, index=False)
    X_train_smote.to_csv(X_train_smote_path, index=False)
    y_train_smote.to_frame(name=TARGET_COL).to_csv(y_train_smote_path, index=False)

    # Save preprocessor
    joblib.dump(fitted_preprocessor, PREPROCESSOR_PATH)

    # Build Summary Metadata
    summary_data = {
        "dataset_name": "AI4I 2020 Predictive Maintenance Dataset",
        "random_state": RANDOM_STATE,
        "raw_total_samples": len(df_raw),
        "excluded_identifiers": ID_COLS,
        "excluded_leakage_columns": FAILURE_TYPE_COLS,
        "input_features_raw": FEATURE_COLS,
        "encoded_feature_names": feature_names,
        "num_input_features": len(FEATURE_COLS),
        "num_encoded_features": len(feature_names),
        "target_column": TARGET_COL,
        "train_samples": train_total,
        "test_samples": test_total,
        "train_healthy_count": train_healthy,
        "train_failure_count": train_failures,
        "train_failure_percentage": round(train_fail_pct, 4),
        "test_healthy_count": test_healthy,
        "test_failure_count": test_failures,
        "test_failure_percentage": round(test_fail_pct, 4),
        "smote_train_samples": smote_total,
        "smote_train_healthy_count": smote_healthy,
        "smote_train_failure_count": smote_failures,
        "smote_train_failure_percentage": round(smote_fail_pct, 4),
        "shapes": {
            "X_train": list(X_train_proc.shape),
            "X_test": list(X_test_proc.shape),
            "y_train": [len(y_train), 1],
            "y_test": [len(y_test), 1],
            "X_train_smote": list(X_train_smote.shape),
            "y_train_smote": [len(y_train_smote), 1]
        },
        "saved_artifacts": {
            "X_train": str(X_train_path.relative_to(Path(__file__).resolve().parent.parent)),
            "X_test": str(X_test_path.relative_to(Path(__file__).resolve().parent.parent)),
            "y_train": str(y_train_path.relative_to(Path(__file__).resolve().parent.parent)),
            "y_test": str(y_test_path.relative_to(Path(__file__).resolve().parent.parent)),
            "X_train_smote": str(X_train_smote_path.relative_to(Path(__file__).resolve().parent.parent)),
            "y_train_smote": str(y_train_smote_path.relative_to(Path(__file__).resolve().parent.parent)),
            "preprocessor": str(PREPROCESSOR_PATH.relative_to(Path(__file__).resolve().parent.parent))
        }
    }

    # Save JSON summary
    summary_json_path = METRICS_DIR / "preprocessing_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(summary_data, f, indent=4)

    # Save CSV summary table
    summary_table = pd.DataFrame([
        {"Metric / Attribute": "Raw Dataset Rows", "Value": str(len(df_raw))},
        {"Metric / Attribute": "Raw Dataset Columns", "Value": str(len(df_raw.columns))},
        {"Metric / Attribute": "Excluded ID Columns", "Value": ", ".join(ID_COLS)},
        {"Metric / Attribute": "Excluded Failure Flags (Leakage Prevention)", "Value": ", ".join(FAILURE_TYPE_COLS)},
        {"Metric / Attribute": "Input Features (Raw)", "Value": ", ".join(FEATURE_COLS)},
        {"Metric / Attribute": "Encoded Features", "Value": ", ".join(feature_names)},
        {"Metric / Attribute": "Train Sample Count", "Value": f"{train_total} (80.0%)"},
        {"Metric / Attribute": "Test Sample Count", "Value": f"{test_total} (20.0%)"},
        {"Metric / Attribute": "Train Class Distribution [0 / 1]", "Value": f"{train_healthy} / {train_failures} ({train_fail_pct:.2f}% failures)"},
        {"Metric / Attribute": "Test Class Distribution [0 / 1]", "Value": f"{test_healthy} / {test_failures} ({test_fail_pct:.2f}% failures)"},
        {"Metric / Attribute": "Post-SMOTE Train Class Distribution [0 / 1]", "Value": f"{smote_healthy} / {smote_failures} ({smote_fail_pct:.2f}% failures)"},
        {"Metric / Attribute": "X_train Shape", "Value": str(X_train_proc.shape)},
        {"Metric / Attribute": "X_test Shape", "Value": str(X_test_proc.shape)},
        {"Metric / Attribute": "X_train_smote Shape", "Value": str(X_train_smote.shape)},
        {"Metric / Attribute": "y_train Shape", "Value": f"({len(y_train)}, 1)"},
        {"Metric / Attribute": "y_test Shape", "Value": f"({len(y_test)}, 1)"},
        {"Metric / Attribute": "y_train_smote Shape", "Value": f"({len(y_train_smote)}, 1)"},
        {"Metric / Attribute": "Preprocessor Path", "Value": str(PREPROCESSOR_PATH.name)},
        {"Metric / Attribute": "Random Seed", "Value": str(RANDOM_STATE)}
    ])
    summary_csv_path = TABLES_DIR / "preprocessing_summary.csv"
    summary_table.to_csv(summary_csv_path, index=False)

    print("\n--- Artifacts Successfully Saved ---")
    print(f"X_train:              {X_train_path}")
    print(f"X_test:               {X_test_path}")
    print(f"y_train:              {y_train_path}")
    print(f"y_test:               {y_test_path}")
    print(f"X_train_smote:        {X_train_smote_path}")
    print(f"y_train_smote:        {y_train_smote_path}")
    print(f"Preprocessor:         {PREPROCESSOR_PATH}")
    print(f"Metrics JSON:         {summary_json_path}")
    print(f"Summary Table CSV:    {summary_csv_path}")

    return summary_data


if __name__ == "__main__":
    summary = execute_pipeline()
    print("\nExecution complete.")
