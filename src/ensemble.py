"""
Stacking Ensemble Module for Manufacturing Predictive Maintenance.
Constructs, trains, serializes, and evaluates a leakage-free Stacking Classifier
combining Decision Tree, Random Forest, XGBoost, and LightGBM base learners
with a Logistic Regression meta-learner using 5-fold Stratified Cross-Validation.
"""
import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    FIGURES_DIR,
    TABLES_DIR,
    METRICS_DIR,
    RANDOM_STATE,
    TARGET_COL
)
from src.train_models import (
    load_processed_data,
    evaluate_model,
    plot_confusion_matrix
)

# Set publication plotting style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "semibold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.titleweight": "bold",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})


def build_base_estimators(random_state: int = RANDOM_STATE) -> List[Tuple[str, Any]]:
    """
    Builds list of base learners with fixed random states.
    """
    estimators = [
        (
            "decision_tree",
            DecisionTreeClassifier(
                random_state=random_state,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2
            )
        ),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=100,
                random_state=random_state,
                max_depth=12,
                n_jobs=-1
            )
        ),
        (
            "xgboost",
            XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=random_state,
                eval_metric="logloss",
                use_label_encoder=False,
                n_jobs=-1
            )
        ),
        (
            "lightgbm",
            LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=random_state,
                verbose=-1,
                n_jobs=-1
            )
        )
    ]
    return estimators


def build_stacking_ensemble(
    estimators: List[Tuple[str, Any]],
    n_splits: int = 5,
    random_state: int = RANDOM_STATE
) -> StackingClassifier:
    """
    Builds a StackingClassifier using StratifiedKFold cross-validation for out-of-fold
    meta-feature generation, avoiding target leakage into the meta-learner.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    meta_learner = LogisticRegression(random_state=random_state, max_iter=1000)

    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=cv,
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=-1
    )
    return stacking_clf


def plot_final_model_comparison(
    comparison_df: pd.DataFrame,
    save_path: Path
) -> None:
    """
    Generates a publication-quality grouped bar chart comparing all 5 models.
    """
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"]
    models = list(comparison_df["Model"])

    n_models = len(models)
    n_metrics = len(metrics)

    x = np.arange(n_models)
    width = 0.13

    palette = ["#2B5C8F", "#E67E22", "#27AE60", "#D9381E", "#8E44AD", "#16A085"]

    fig, ax = plt.subplots(figsize=(13, 6.5))

    for i, (metric, color) in enumerate(zip(metrics, palette)):
        values = comparison_df[metric].values
        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=metric,
            color=color,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.9
        )
        for bar, val in zip(bars, values):
            y_val = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_val + 0.015,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                rotation=90
            )

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight="bold", fontsize=10.5)
    ax.set_ylabel("Score (0.0 to 1.0)", labelpad=8)
    ax.set_ylim(0, 1.15)
    ax.set_title("Complete Predictive Maintenance Models Comparison (AI4I 2020 Test Set)", pad=14)
    ax.legend(loc="upper right", ncol=3, frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def run_stacking_pipeline() -> Dict[str, Any]:
    """
    Executes Phase 5 Stacking Ensemble workflow:
    1. Loads processed SMOTE training data and untouched test data.
    2. Constructs base estimators and StackingClassifier.
    3. Fits stacking model with 5-fold Stratified CV out-of-fold predictions.
    4. Serializes ensemble to models/stacking_ensemble.joblib.
    5. Evaluates ensemble on untouched test data.
    6. Combines with Phase 4 results and generates final_model_comparison.csv.
    7. Plots final comparison chart and confusion matrix.
    8. Generates stacking_summary.json.
    """
    print("=" * 70)
    print("PHASE 5: STACKING ENSEMBLE DEVELOPMENT & EVALUATION")
    print("=" * 70)

    # 1. Load data
    X_train, y_train, X_test, y_test, feature_names = load_processed_data()
    print(f"Loaded SMOTE Training Data: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
    print(f"Loaded Untouched Test Data:  {X_test.shape[0]:,} samples, {X_test.shape[1]} features")

    # 2. Build Stacking Ensemble
    base_estimators = build_base_estimators(RANDOM_STATE)
    stacking_model = build_stacking_ensemble(base_estimators, n_splits=5, random_state=RANDOM_STATE)

    print("\n--- Training Stacking Classifier (5-Fold Stratified CV Out-of-Fold) ---")
    stacking_model.fit(X_train, y_train)

    # 3. Save serialized model
    ensemble_path = MODELS_DIR / "stacking_ensemble.joblib"
    joblib.dump(stacking_model, ensemble_path)
    print(f"Serialized Stacking Ensemble saved to: {ensemble_path}")

    # 4. Evaluate on untouched test data
    eval_res = evaluate_model(stacking_model, X_test, y_test)
    print(f"\n--- Stacking Ensemble Evaluation on Test Set ---")
    print(f"Accuracy:  {eval_res['accuracy']:.4f}")
    print(f"Precision: {eval_res['precision']:.4f}")
    print(f"Recall:    {eval_res['recall']:.4f}")
    print(f"F1-Score:  {eval_res['f1_score']:.4f}")
    print(f"ROC-AUC:   {eval_res['roc_auc']:.4f}")
    print(f"PR-AUC:    {eval_res['pr_auc']:.4f}")
    print(f"Confusion Matrix: TN={eval_res['tn']}, FP={eval_res['fp']}, FN={eval_res['fn']}, TP={eval_res['tp']}")

    # 5. Plot Stacking Confusion Matrix
    cm_path = FIGURES_DIR / "confusion_matrix_stacking_ensemble.png"
    plot_confusion_matrix(eval_res["confusion_matrix"], "Stacking Ensemble", cm_path)
    print(f"[Generated Figure] {cm_path.name}")

    # 6. Load Phase 4 performance table and append Stacking results
    phase4_perf_path = METRICS_DIR / "model_performance.csv"
    if phase4_perf_path.exists():
        phase4_df = pd.read_csv(phase4_perf_path)
    else:
        raise FileNotFoundError("Phase 4 model_performance.csv not found!")

    stacking_record = {
        "Model": "Stacking Ensemble",
        "Accuracy": round(eval_res["accuracy"], 4),
        "Precision": round(eval_res["precision"], 4),
        "Recall": round(eval_res["recall"], 4),
        "F1-Score": round(eval_res["f1_score"], 4),
        "ROC-AUC": round(eval_res["roc_auc"], 4),
        "PR-AUC": round(eval_res["pr_auc"], 4),
        "True Negative": eval_res["tn"],
        "False Positive": eval_res["fp"],
        "False Negative": eval_res["fn"],
        "True Positive": eval_res["tp"]
    }

    # Combine ensuring no duplicates
    combined_rows = [r for r in phase4_df.to_dict(orient="records") if r["Model"] != "Stacking Ensemble"]
    combined_rows.append(stacking_record)
    final_comparison_df = pd.DataFrame(combined_rows)

    # Save final comparison CSVs
    final_csv_metrics = METRICS_DIR / "final_model_comparison.csv"
    final_csv_tables = TABLES_DIR / "final_model_comparison.csv"
    final_comparison_df.to_csv(final_csv_metrics, index=False)
    final_comparison_df.to_csv(final_csv_tables, index=False)
    print(f"[Saved Table] {final_csv_metrics}")
    print(f"[Saved Table] {final_csv_tables}")

    # 7. Plot final model comparison figure
    final_fig_path = FIGURES_DIR / "final_model_comparison.png"
    plot_final_model_comparison(final_comparison_df, final_fig_path)
    print(f"[Generated Figure] {final_fig_path.name}")

    # 8. Compare Stacking vs LightGBM
    lgbm_row = final_comparison_df[final_comparison_df["Model"] == "LightGBM"].iloc[0]
    stack_row = final_comparison_df[final_comparison_df["Model"] == "Stacking Ensemble"].iloc[0]

    comparison_vs_lgbm = {
        "Recall": {
            "LightGBM": float(lgbm_row["Recall"]),
            "Stacking": float(stack_row["Recall"]),
            "Delta": round(float(stack_row["Recall"]) - float(lgbm_row["Recall"]), 4)
        },
        "F1-Score": {
            "LightGBM": float(lgbm_row["F1-Score"]),
            "Stacking": float(stack_row["F1-Score"]),
            "Delta": round(float(stack_row["F1-Score"]) - float(lgbm_row["F1-Score"]), 4)
        },
        "False_Negatives": {
            "LightGBM": int(lgbm_row["False Negative"]),
            "Stacking": int(stack_row["False Negative"]),
            "Delta": int(stack_row["False Negative"]) - int(lgbm_row["False Negative"])
        },
        "Precision": {
            "LightGBM": float(lgbm_row["Precision"]),
            "Stacking": float(stack_row["Precision"]),
            "Delta": round(float(stack_row["Precision"]) - float(lgbm_row["Precision"]), 4)
        },
        "PR-AUC": {
            "LightGBM": float(lgbm_row["PR-AUC"]),
            "Stacking": float(stack_row["PR-AUC"]),
            "Delta": round(float(stack_row["PR-AUC"]) - float(lgbm_row["PR-AUC"]), 4)
        }
    }

    # Meta-learner coefficients
    meta_clf = stacking_model.final_estimator_
    meta_coefs = {}
    if hasattr(meta_clf, "coef_"):
        for (est_name, _), coef in zip(base_estimators, meta_clf.coef_[0]):
            meta_coefs[est_name] = round(float(coef), 4)

    # 9. Save Summary JSON
    summary_data = {
        "ensemble_architecture": {
            "base_models": [name for name, _ in base_estimators],
            "meta_learner": "LogisticRegression",
            "cv_strategy": "StratifiedKFold",
            "n_splits": 5,
            "random_state": RANDOM_STATE,
            "stack_method": "predict_proba",
            "passthrough": False,
            "meta_learner_weights": meta_coefs
        },
        "training_data": {
            "samples": len(X_train),
            "features": len(feature_names),
            "feature_names": feature_names
        },
        "test_evaluation": {
            "test_samples": len(X_test),
            "test_failures": int(y_test.sum()),
            "metrics": {
                "accuracy": round(eval_res["accuracy"], 4),
                "precision": round(eval_res["precision"], 4),
                "recall": round(eval_res["recall"], 4),
                "f1_score": round(eval_res["f1_score"], 4),
                "roc_auc": round(eval_res["roc_auc"], 4),
                "pr_auc": round(eval_res["pr_auc"], 4),
                "tn": eval_res["tn"],
                "fp": eval_res["fp"],
                "fn": eval_res["fn"],
                "tp": eval_res["tp"]
            }
        },
        "comparison_vs_lightgbm": comparison_vs_lgbm,
        "all_models_performance": final_comparison_df.to_dict(orient="records")
    }

    summary_json_path = METRICS_DIR / "stacking_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(summary_data, f, indent=4)
    print(f"[Saved JSON Metrics] {summary_json_path.name}")

    print("\n" + "=" * 70)
    print("PHASE 5 STACKING ENSEMBLE COMPLETED SUCCESSFULLY")
    print("=" * 70)

    return summary_data


if __name__ == "__main__":
    run_stacking_pipeline()
