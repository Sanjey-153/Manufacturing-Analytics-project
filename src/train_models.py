"""
Model Training and Evaluation Pipeline for Manufacturing Predictive Maintenance.
Trains and evaluates 4 tree-based machine learning classifiers on SMOTE-balanced training data:
1. Decision Tree
2. Random Forest
3. XGBoost
4. LightGBM
Evaluates models on the untouched test set using Accuracy, Precision, Recall, F1-score, ROC-AUC, PR-AUC,
confusion matrices, and feature importances.
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
from sklearn.ensemble import RandomForestClassifier
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


def load_processed_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str]]:
    """
    Loads processed SMOTE training data and untouched test data.
    Verifies feature names and shapes.
    """
    X_train_path = PROCESSED_DATA_DIR / "X_train_smote.csv"
    y_train_path = PROCESSED_DATA_DIR / "y_train_smote.csv"
    X_test_path = PROCESSED_DATA_DIR / "X_test.csv"
    y_test_path = PROCESSED_DATA_DIR / "y_test.csv"

    if not X_train_path.exists() or not X_test_path.exists():
        raise FileNotFoundError("Processed datasets not found in data/processed/. Run Phase 2 first.")

    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)[TARGET_COL]
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)[TARGET_COL]

    feature_names = list(X_train.columns)
    if list(X_test.columns) != feature_names:
        raise ValueError("Feature mismatch between training and test sets!")

    # Sanitize feature names for XGBoost and LightGBM compatibility (no [, ], or < allowed)
    clean_feature_names = [col.replace("[", "(").replace("]", ")") for col in feature_names]
    X_train.columns = clean_feature_names
    X_test.columns = clean_feature_names

    return X_train, y_train, X_test, y_test, clean_feature_names


def get_models(random_state: int = RANDOM_STATE) -> Dict[str, Any]:
    """
    Instantiates the 4 baseline classifiers with fixed random states and standard configurations.
    """
    models = {
        "Decision Tree": DecisionTreeClassifier(
            random_state=random_state,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            max_depth=12,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=random_state,
            eval_metric="logloss",
            use_label_encoder=False,
            n_jobs=-1
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=random_state,
            verbose=-1,
            n_jobs=-1
        )
    }
    return models


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Any]:
    """
    Evaluates a trained model on test data, calculating classification metrics,
    probabilities, ROC-AUC, PR-AUC, and confusion matrix.
    """
    y_pred = model.predict(X_test)
    
    # Probability estimates for positive class (1)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_test)
    else:
        y_prob = y_pred

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "y_pred": y_pred,
        "y_prob": y_prob
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    save_path: Path
) -> None:
    """
    Plots a publication-quality confusion matrix heatmap.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        linewidths=1.5,
        linecolor="black",
        annot_kws={"size": 14, "weight": "bold"},
        ax=ax
    )

    ax.set_title(f"Confusion Matrix — {model_name}", pad=14)
    ax.set_xlabel("Predicted Label", labelpad=8)
    ax.set_ylabel("True Label", labelpad=8)
    ax.set_xticklabels(["Healthy (0)", "Failure (1)"], fontweight="semibold")
    ax.set_yticklabels(["Healthy (0)", "Failure (1)"], fontweight="semibold")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    model_name: str,
    save_path: Path
) -> pd.DataFrame:
    """
    Extracts and plots feature importance for tree-based models.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        return pd.DataFrame()

    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(
        fi_df["Feature"],
        fi_df["Importance"],
        color="#2B5C8F",
        edgecolor="black",
        linewidth=1.1,
        height=0.55
    )

    for bar, val in zip(bars, fi_df["Importance"]):
        ax.text(
            val + 0.008,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f} ({val*100:.1f}%)",
            va="center",
            ha="left",
            fontsize=9.5,
            fontweight="bold"
        )

    ax.set_xlim(0, max(fi_df["Importance"]) * 1.25)
    ax.set_xlabel("Relative Feature Importance (Gini / Gain)")
    ax.set_title(f"Feature Importance — {model_name}", pad=14)
    ax.grid(axis="x", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return fi_df.sort_values("Importance", ascending=False)


def plot_performance_comparison(
    perf_df: pd.DataFrame,
    save_path: Path
) -> None:
    """
    Creates a grouped bar chart comparing Accuracy, Precision, Recall, and F1-Score across all models.
    """
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"]
    models = list(perf_df["Model"])
    
    n_models = len(models)
    n_metrics = len(metrics)
    
    x = np.arange(n_models)
    width = 0.13
    
    palette = ["#2B5C8F", "#E67E22", "#27AE60", "#D9381E", "#8E44AD", "#34495E"]

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (metric, color) in enumerate(zip(metrics, palette)):
        values = perf_df[metric].values
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
    ax.set_xticklabels(models, fontweight="bold", fontsize=11)
    ax.set_ylabel("Score (0.0 to 1.0)", labelpad=8)
    ax.set_ylim(0, 1.15)
    ax.set_title("Machine Learning Models Performance Comparison (AI4I 2020 Test Set)", pad=14)
    ax.legend(loc="upper right", ncol=3, frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def train_and_evaluate_all() -> Dict[str, Any]:
    """
    Executes the entire training and evaluation workflow:
    1. Loads processed data.
    2. Trains Decision Tree, Random Forest, XGBoost, and LightGBM.
    3. Serializes models to models/.
    4. Evaluates on test set.
    5. Saves performance tables, confusion matrices, and feature importance figures.
    """
    print("=" * 70)
    print("PHASE 4: MACHINE LEARNING MODEL DEVELOPMENT & EVALUATION")
    print("=" * 70)

    # Ensure directories
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    X_train, y_train, X_test, y_test, feature_names = load_processed_data()
    print(f"Loaded SMOTE Training Data: {X_train.shape[0]:,} rows, {X_train.shape[1]} features")
    print(f"Loaded Untouched Test Data:  {X_test.shape[0]:,} rows, {X_test.shape[1]} features")
    print(f"Test Set Class Counts: {y_test.value_counts().to_dict()} (Failures: {y_test.sum()})")
    print(f"Feature Names: {feature_names}")

    models = get_models(RANDOM_STATE)
    
    performance_records = []
    training_summary = {
        "dataset_info": {
            "training_samples": len(X_train),
            "training_features": len(feature_names),
            "feature_list": feature_names,
            "test_samples": len(X_test),
            "test_failures": int(y_test.sum()),
            "test_healthy": int((y_test == 0).sum()),
            "random_seed": RANDOM_STATE
        },
        "models": {}
    }

    file_prefix_map = {
        "Decision Tree": "decision_tree",
        "Random Forest": "random_forest",
        "XGBoost": "xgboost",
        "LightGBM": "lightgbm"
    }

    trained_models = {}

    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        model.fit(X_train, y_train)
        trained_models[name] = model

        # Save model
        prefix = file_prefix_map[name]
        model_path = MODELS_DIR / f"{prefix}.joblib"
        joblib.dump(model, model_path)
        print(f"Serialized model saved to: {model_path}")

        # Evaluate model
        eval_res = evaluate_model(model, X_test, y_test)
        print(f"Evaluation: Accuracy={eval_res['accuracy']:.4f}, Precision={eval_res['precision']:.4f}, "
              f"Recall={eval_res['recall']:.4f}, F1={eval_res['f1_score']:.4f}, "
              f"ROC-AUC={eval_res['roc_auc']:.4f}, PR-AUC={eval_res['pr_auc']:.4f}")
        print(f"Confusion Matrix: TN={eval_res['tn']}, FP={eval_res['fp']}, FN={eval_res['fn']}, TP={eval_res['tp']}")

        # Plot Confusion Matrix
        cm_path = FIGURES_DIR / f"confusion_matrix_{prefix}.png"
        plot_confusion_matrix(eval_res["confusion_matrix"], name, cm_path)

        # Feature Importance
        fi_path = FIGURES_DIR / f"feature_importance_{prefix}.png"
        fi_df = plot_feature_importance(model, feature_names, name, fi_path)

        # Performance table record
        performance_records.append({
            "Model": name,
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
        })

        # Training metadata summary
        training_summary["models"][name] = {
            "model_file": str(model_path.name),
            "status": "Trained and Evaluated Successfully",
            "hyperparameters": {k: str(v) for k, v in model.get_params().items() if k in [
                "max_depth", "n_estimators", "learning_rate", "min_samples_split",
                "min_samples_leaf", "random_state", "eval_metric"
            ]},
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
            },
            "top_features": fi_df.to_dict(orient="records") if not fi_df.empty else []
        }

    # Save performance table CSV
    perf_df = pd.DataFrame(performance_records)
    perf_csv_metrics = METRICS_DIR / "model_performance.csv"
    perf_csv_tables = TABLES_DIR / "model_performance.csv"
    perf_df.to_csv(perf_csv_metrics, index=False)
    perf_df.to_csv(perf_csv_tables, index=False)
    print(f"\n[Saved Table] {perf_csv_metrics}")
    print(f"[Saved Table] {perf_csv_tables}")

    # Plot performance comparison
    comp_fig_path = FIGURES_DIR / "model_performance_comparison.png"
    plot_performance_comparison(perf_df, comp_fig_path)
    print(f"[Generated Figure] {comp_fig_path.name}")

    # Save JSON training summary
    summary_json_path = METRICS_DIR / "model_training_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(training_summary, f, indent=4)
    print(f"[Saved JSON Metrics] {summary_json_path.name}")

    print("\n" + "=" * 70)
    print("PHASE 4 ML MODEL DEVELOPMENT COMPLETED SUCCESSFULLY")
    print("=" * 70)

    return {
        "performance_df": perf_df,
        "training_summary": training_summary
    }


if __name__ == "__main__":
    train_and_evaluate_all()
