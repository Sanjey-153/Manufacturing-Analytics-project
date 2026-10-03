"""
Data Loader Module for Manufacturing Analytics Project.
Handles loading raw dataset, schema verification, and basic integrity checks.
"""
import os
import sys
from pathlib import Path
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    RAW_DATA_PATH,
    TARGET_COL,
    FAILURE_TYPE_COLS,
    NUMERICAL_COLS,
    CATEGORICAL_COLS,
    ID_COLS
)


def load_raw_data(filepath: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Loads the raw dataset from CSV.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")
    
    df = pd.read_csv(filepath)
    return df


def verify_dataset_integrity(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Inspects dataset structure, data types, missing values, duplicates, and column checks.
    """
    expected_cols = ID_COLS + CATEGORICAL_COLS + NUMERICAL_COLS + [TARGET_COL] + FAILURE_TYPE_COLS
    present_cols = df.columns.tolist()
    missing_expected_cols = [col for col in expected_cols if col not in present_cols]
    
    null_counts = df.isnull().sum().to_dict()
    total_nulls = int(df.isnull().sum().sum())
    duplicate_count = int(df.duplicated().sum())
    
    results = {
        "num_rows": len(df),
        "num_cols": len(df.columns),
        "columns": present_cols,
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_expected_columns": missing_expected_cols,
        "null_counts": null_counts,
        "total_nulls": total_nulls,
        "duplicate_count": duplicate_count,
        "is_valid": len(missing_expected_cols) == 0 and total_nulls == 0 and duplicate_count == 0
    }
    return results


def analyze_target_and_failures(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyzes target class distribution and specific failure modes.
    """
    target_counts = df[TARGET_COL].value_counts().to_dict()
    target_norm = df[TARGET_COL].value_counts(normalize=True).to_dict()
    
    # Class imbalance ratio
    neg_count = target_counts.get(0, 0)
    pos_count = target_counts.get(1, 0)
    imbalance_ratio = (neg_count / pos_count) if pos_count > 0 else float("inf")
    
    failure_mode_counts = {col: int(df[col].sum()) for col in FAILURE_TYPE_COLS if col in df.columns}
    
    return {
        "target_counts": target_counts,
        "target_percentages": {k: v * 100 for k, v in target_norm.items()},
        "imbalance_ratio": imbalance_ratio,
        "failure_mode_counts": failure_mode_counts
    }


if __name__ == "__main__":
    df = load_raw_data()
    print("Dataset Shape:", df.shape)
    integrity = verify_dataset_integrity(df)
    print("Integrity Summary:", integrity)
    stats = analyze_target_and_failures(df)
    print("Target & Failure Analysis:", stats)
