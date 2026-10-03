"""
Cost-Based Probability Threshold Framework for CNC Predictive Maintenance (F4).
Provides deterministic threshold sensitivity analysis, cost matrix evaluation,
trade-off curves, and scenario-based decision optimization on the validated holdout test set.

RESEARCH SCOPE & INTEGRITY DISCLAIMER:
This module provides a decision-support sensitivity framework. The project does not
contain verified real-world CNC maintenance cost data. Any cost values, templates,
or cost-minimizing thresholds are illustrative / user-defined assumptions and are NOT
validated industrial maintenance cost recommendations. The operational decision threshold
remains fixed at 0.50.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import pandas as pd

# Setup project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    RAW_DATA_PATH,
    TARGET_COL,
    FEATURE_COLS,
    RANDOM_STATE,
    MODELS_DIR,
    PREPROCESSOR_PATH
)
from src.data_loader import load_raw_data
from src.preprocessing import split_data, separate_features_and_target
from src.prediction import load_inference_pipeline, preprocess_features_for_inference

OUTPUTS_THRESHOLDS_DIR = PROJECT_ROOT / "outputs" / "thresholds"

# Pre-defined illustrative cost scenario templates (Non-industrial, educational fixtures)
ILLUSTRATIVE_COST_SCENARIOS = {
    "FN-Dominant (High Missed Failure Penalty)": {
        "description": "Prioritizes fault detection; missed machine failures carry higher relative penalty.",
        "C_TN": 0.0,
        "C_FP": 10.0,
        "C_FN": 100.0,
        "C_TP": 5.0,
        "cost_unit": "Relative Cost Units"
    },
    "Balanced (Equal Operational Tradeoffs)": {
        "description": "Balances false alarm consequences with missed failure avoidance.",
        "C_TN": 0.0,
        "C_FP": 25.0,
        "C_FN": 50.0,
        "C_TP": 5.0,
        "cost_unit": "Relative Cost Units"
    },
    "FP-Dominant (High False Alarm Penalty)": {
        "description": "Penalizes frequent false alarms and unnecessary inspection/maintenance actions.",
        "C_TN": 0.0,
        "C_FP": 80.0,
        "C_FN": 20.0,
        "C_TP": 10.0,
        "cost_unit": "Relative Cost Units"
    }
}

DISCLAIMER_TEXT = (
    "Illustrative / user-defined assumptions — not validated industrial costs. "
    "The operational classification threshold remains 0.50."
)


def get_holdout_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Reconstructs the exact validated stratified holdout test split (2,000 rows).
    Distribution: 1,932 Healthy (0), 68 Failure (1).
    """
    df_raw = load_raw_data(RAW_DATA_PATH)
    X, y = separate_features_and_target(df_raw)
    _, X_test, _, y_test = split_data(X, y, test_size=0.20, random_state=RANDOM_STATE)
    return X_test, y_test


def get_holdout_predictions(
    model: Optional[Any] = None,
    preprocessor: Optional[Any] = None
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Computes LightGBM failure probability predictions on the untouched holdout test set.
    Returns:
        y_true: Ground truth binary labels (2000,)
        y_prob: Predicted failure probabilities P(y=1) (2000,)
        X_test: Raw holdout feature DataFrame (2000, 6)
    """
    if model is None or preprocessor is None:
        model, preprocessor = load_inference_pipeline()

    X_test, y_test = get_holdout_data()
    X_test_proc = preprocess_features_for_inference(X_test, preprocessor)
    
    # Model inference
    probs = model.predict_proba(X_test_proc)[:, 1]
    y_true = y_test.values.astype(int)
    y_prob = probs.astype(float)

    return y_true, y_prob, X_test


def save_holdout_predictions_artifact(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_dir: Path = OUTPUTS_THRESHOLDS_DIR
) -> Path:
    """
    Saves holdout benchmark predictions artifact for reproducibility.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "holdout_predictions.csv"
    
    df_art = pd.DataFrame({
        "Index": np.arange(len(y_true)),
        "True_Label": y_true,
        "Failure_Probability": np.round(y_prob, 5)
    })
    df_art.to_csv(artifact_path, index=False)
    return artifact_path


def calculate_confusion_matrix(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float
) -> Dict[str, int]:
    """
    Calculates confusion matrix counts (TP, TN, FP, FN) for a given classification threshold.
    Handles exact edge cases at 0.00 and 1.00.
    """
    # Prediction: 1 if prob >= threshold else 0
    # Special handling for threshold 0.0: all predicted positive (1)
    # Special handling for threshold > 1.0: all predicted negative (0)
    preds = (y_prob >= threshold).astype(int)
    
    tp = int(np.sum((preds == 1) & (y_true == 1)))
    tn = int(np.sum((preds == 0) & (y_true == 0)))
    fp = int(np.sum((preds == 1) & (y_true == 0)))
    fn = int(np.sum((preds == 0) & (y_true == 1)))

    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def calculate_expected_cost(
    tp: int,
    tn: int,
    fp: int,
    fn: int,
    c_tn: float = 0.0,
    c_fp: float = 10.0,
    c_fn: float = 100.0,
    c_tp: float = 5.0
) -> Tuple[float, float]:
    """
    Calculates total cost and expected cost per observation using the 4-outcome decision-cost framework:
    - C_TN: Cost associated with correctly leaving a healthy machine unflagged.
    - C_FP: Cost of a false alarm or unnecessary inspection/maintenance action.
    - C_FN: Cost associated with a missed machine-failure case.
    - C_TP: Cost associated with correctly flagging a failure-risk case and the resulting planned response.

    Total Cost = (TN * C_TN) + (FP * C_FP) + (FN * C_FN) + (TP * C_TP)
    Expected Cost per observation = Total Cost / N
    """
    n_total = tp + tn + fp + fn
    if n_total == 0:
        return 0.0, 0.0

    total_cost = (tn * float(c_tn)) + (fp * float(c_fp)) + (fn * float(c_fn)) + (tp * float(c_tp))
    expected_cost = total_cost / float(n_total)
    return float(total_cost), float(expected_cost)


def calculate_threshold_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    c_tn: float = 0.0,
    c_fp: float = 10.0,
    c_fn: float = 100.0,
    c_tp: float = 5.0
) -> Dict[str, Any]:
    """
    Calculates comprehensive classification and decision-cost performance metrics for a single threshold.
    """
    cm = calculate_confusion_matrix(y_true, y_prob, threshold)
    tp, tn, fp, fn = cm["TP"], cm["TN"], cm["FP"], cm["FN"]
    n_total = len(y_true)

    # Classification Metrics with zero-division protection
    accuracy = (tp + tn) / n_total if n_total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    alert_rate = (tp + fp) / n_total if n_total > 0 else 0.0
    predicted_failure_rate = alert_rate

    total_cost, expected_cost = calculate_expected_cost(tp, tn, fp, fn, c_tn, c_fp, c_fn, c_tp)

    return {
        "Threshold": round(float(threshold), 4),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
        "Specificity": round(specificity, 4),
        "FPR": round(fpr, 4),
        "FNR": round(fnr, 4),
        "Alert_Rate": round(alert_rate, 4),
        "Predicted_Failure_Rate": round(predicted_failure_rate, 4),
        "Total_Cost": round(total_cost, 2),
        "Expected_Cost": round(expected_cost, 4)
    }


def generate_threshold_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    c_tn: float = 0.0,
    c_fp: float = 10.0,
    c_fn: float = 100.0,
    c_tp: float = 5.0,
    step: float = 0.01
) -> pd.DataFrame:
    """
    Evaluates a deterministic range of probability thresholds from 0.00 to 1.00.
    Returns a DataFrame containing metrics and expected costs for each threshold.
    """
    # Deterministic threshold array including 0.00 and 1.00
    num_steps = int(round(1.0 / step)) + 1
    thresholds = np.linspace(0.0, 1.0, num_steps)

    records = []
    for th in thresholds:
        metrics = calculate_threshold_metrics(
            y_true=y_true,
            y_prob=y_prob,
            threshold=float(th),
            c_tn=c_tn,
            c_fp=c_fp,
            c_fn=c_fn,
            c_tp=c_tp
        )
        records.append(metrics)

    df_sweep = pd.DataFrame(records)
    return df_sweep


def find_cost_minimizing_threshold(df_sweep: pd.DataFrame) -> Dict[str, Any]:
    """
    Identifies the cost-minimizing analysis threshold from the sweep DataFrame.
    """
    if df_sweep.empty:
        raise ValueError("Threshold sweep DataFrame is empty.")

    min_idx = df_sweep["Expected_Cost"].idxmin()
    best_row = df_sweep.loc[min_idx].to_dict()
    return best_row


def compare_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresh_a: float = 0.50,
    thresh_b: Optional[float] = None,
    c_tn: float = 0.0,
    c_fp: float = 10.0,
    c_fn: float = 100.0,
    c_tp: float = 5.0
) -> Dict[str, Any]:
    """
    Compares the baseline operational threshold (0.50) with an alternative analysis threshold.
    """
    if thresh_b is None:
        sweep = generate_threshold_sweep(y_true, y_prob, c_tn, c_fp, c_fn, c_tp)
        thresh_b = float(find_cost_minimizing_threshold(sweep)["Threshold"])

    metrics_a = calculate_threshold_metrics(y_true, y_prob, thresh_a, c_tn, c_fp, c_fn, c_tp)
    metrics_b = calculate_threshold_metrics(y_true, y_prob, thresh_b, c_tn, c_fp, c_fn, c_tp)

    cost_diff = metrics_b["Expected_Cost"] - metrics_a["Expected_Cost"]
    recall_diff = metrics_b["Recall"] - metrics_a["Recall"]
    prec_diff = metrics_b["Precision"] - metrics_a["Precision"]
    fn_diff = metrics_b["FN"] - metrics_a["FN"]
    fp_diff = metrics_b["FP"] - metrics_a["FP"]

    return {
        "operational_threshold": metrics_a,
        "analysis_threshold": metrics_b,
        "delta": {
            "Expected_Cost_Delta": round(cost_diff, 4),
            "Recall_Delta": round(recall_diff, 4),
            "Precision_Delta": round(prec_diff, 4),
            "FN_Delta": fn_diff,
            "FP_Delta": fp_diff
        }
    }


def build_cost_sensitivity_matrix(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    ratios: Optional[List[float]] = None,
    c_fp: float = 10.0,
    c_tn: float = 0.0,
    c_tp: float = 5.0,
    threshold_step: float = 0.05
) -> pd.DataFrame:
    """
    Generates a 2D sensitivity matrix mapping Cost Ratios (C_FN / C_FP) vs Thresholds to Expected Costs.
    Useful for visualizing how cost asymmetry shifts the optimal decision threshold.
    """
    if ratios is None:
        ratios = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

    num_steps = int(round(1.0 / threshold_step)) + 1
    thresholds = np.linspace(0.0, 1.0, num_steps)

    rows = []
    for ratio in ratios:
        c_fn = c_fp * ratio
        row_dict = {"Cost_Ratio (C_FN/C_FP)": f"{ratio:.1f}x"}
        for th in thresholds:
            m = calculate_threshold_metrics(
                y_true=y_true,
                y_prob=y_prob,
                threshold=float(th),
                c_tn=c_tn,
                c_fp=c_fp,
                c_fn=c_fn,
                c_tp=c_tp
            )
            row_dict[f"{th:.2f}"] = m["Expected_Cost"]
        rows.append(row_dict)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Testing Threshold Analysis Module...")
    y_t, y_p, _ = get_holdout_predictions()
    print(f"Holdout predictions loaded. Total: {len(y_t)}, Positives: {np.sum(y_t)}")
    
    art = save_holdout_predictions_artifact(y_t, y_p)
    print(f"Artifact saved: {art}")

    sweep = generate_threshold_sweep(y_t, y_p)
    best = find_cost_minimizing_threshold(sweep)
    print(f"Current 0.50 Metrics: {calculate_threshold_metrics(y_t, y_p, 0.50)}")
    print(f"Cost-Minimizing Threshold: {best['Threshold']} (Exp Cost: {best['Expected_Cost']})")
