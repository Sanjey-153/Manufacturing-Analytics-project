"""
Exploratory Data Analysis (EDA) Module for Manufacturing Analytics Project.
Performs comprehensive, publication-quality descriptive analytics on the AI4I 2020 dataset,
analyzing sensor distributions, failure relationships, machine types, correlations,
failure modes, and physical operating regimes.
"""
import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from config import (
    RAW_DATA_PATH,
    FIGURES_DIR,
    TABLES_DIR,
    METRICS_DIR,
    TARGET_COL,
    FAILURE_TYPE_COLS,
    CATEGORICAL_COLS,
    NUMERICAL_COLS,
    ID_COLS
)
from src.data_loader import load_raw_data

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

# Custom palette
PALETTE_BINARY = {0: "#2B5C8F", 1: "#D9381E", "0": "#2B5C8F", "1": "#D9381E"}  # Blue (Healthy), Red (Failure)
PALETTE_TYPES = {"L": "#4A90E2", "M": "#50E3C2", "H": "#F5A623"}
PALETTE_FAILURES = ["#E74C3C", "#E67E22", "#F1C40F", "#9B59B6", "#1ABC9C"]


def plot_target_distribution(df: pd.DataFrame, save_path: Path) -> Dict[str, Any]:
    """
    Analyzes and plots target class distribution (Machine failure).
    """
    counts = df[TARGET_COL].value_counts()
    percentages = df[TARGET_COL].value_counts(normalize=True) * 100

    healthy_count = int(counts.get(0, 0))
    failure_count = int(counts.get(1, 0))
    healthy_pct = float(percentages.get(0, 0.0))
    failure_pct = float(percentages.get(1, 0.0))

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        ["Healthy (0)", "Machine Failure (1)"],
        [healthy_count, failure_count],
        color=[PALETTE_BINARY[0], PALETTE_BINARY[1]],
        width=0.5,
        edgecolor="black",
        linewidth=1.2
    )

    for bar, count, pct in zip(bars, [healthy_count, failure_count], [healthy_pct, failure_pct]):
        y_val = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_val + 120,
            f"{count:,}\n({pct:.2f}%)",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold"
        )

    ax.set_ylim(0, 10500)
    ax.set_ylabel("Number of Observations", labelpad=8)
    ax.set_title("Machine Failure Target Distribution (AI4I 2020)", pad=14)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Annotate class imbalance ratio
    imbalance_ratio = healthy_count / failure_count if failure_count > 0 else 0
    ax.text(
        0.95, 0.90,
        f"Total Samples: {len(df):,}\nClass Ratio: {imbalance_ratio:.2f} : 1\nImbalance: Extreme ({failure_pct:.2f}% failure)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#ced4da", alpha=0.9),
        fontsize=10
    )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return {
        "healthy_count": healthy_count,
        "failure_count": failure_count,
        "healthy_percentage": round(healthy_pct, 4),
        "failure_percentage": round(failure_pct, 4),
        "imbalance_ratio": round(imbalance_ratio, 2)
    }


def plot_machine_type_distribution(df: pd.DataFrame, save_path: Path) -> Dict[str, Any]:
    """
    Analyzes and plots machine Type distribution ('L', 'M', 'H').
    """
    counts = df["Type"].value_counts().reindex(["L", "M", "H"])
    percentages = (df["Type"].value_counts(normalize=True) * 100).reindex(["L", "M", "H"])

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        ["L (Low - 50%)", "M (Medium - 30%)", "H (High - 20%)"],
        counts.values,
        color=[PALETTE_TYPES["L"], PALETTE_TYPES["M"], PALETTE_TYPES["H"]],
        width=0.55,
        edgecolor="black",
        linewidth=1.2
    )

    for bar, count, pct in zip(bars, counts.values, percentages.values):
        y_val = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_val + 100,
            f"{count:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold"
        )

    ax.set_ylim(0, 6200)
    ax.set_ylabel("Number of Observations", labelpad=8)
    ax.set_title("Machine Product Type Distribution (L, M, H Variants)", pad=14)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return {
        "type_counts": counts.to_dict(),
        "type_percentages": {k: round(v, 2) for k, v in percentages.to_dict().items()}
    }


def plot_sensor_distributions(df: pd.DataFrame, save_path: Path) -> None:
    """
    Plots histograms with KDE curves for all 5 continuous numerical sensor features.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for i, col in enumerate(NUMERICAL_COLS):
        ax = axes[i]
        sns.histplot(
            df[col],
            kde=True,
            ax=ax,
            color="#2B5C8F",
            edgecolor="black",
            linewidth=0.8,
            alpha=0.6,
            bins=35
        )
        mean_val = df[col].mean()
        median_val = df[col].median()
        std_val = df[col].std()

        ax.axvline(mean_val, color="#D9381E", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.2f}")
        ax.axvline(median_val, color="#27AE60", linestyle="-.", linewidth=1.5, label=f"Median: {median_val:.2f}")

        ax.set_title(col, pad=8)
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        ax.legend(loc="upper right", frameon=True)
        ax.grid(True, linestyle="--", alpha=0.5)

    # Hide 6th empty subplot
    fig.delaxes(axes[5])

    fig.suptitle("Univariate Distributions of CNC Machine Sensor Parameters", y=0.98, fontsize=15)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(save_path)
    plt.close()


def plot_sensor_correlation_heatmap(df: pd.DataFrame, save_path: Path) -> pd.DataFrame:
    """
    Calculates Pearson correlation matrix among numerical features and plots heatmap.
    """
    corr_matrix = df[NUMERICAL_COLS].corr()

    fig, ax = plt.subplots(figsize=(8, 6.5))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".3f",
        cmap=cmap,
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient (r)"},
        ax=ax
    )

    ax.set_title("Sensor Parameter Correlation Matrix", pad=14)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return corr_matrix


def plot_failure_mode_distribution(df: pd.DataFrame, save_path: Path) -> Dict[str, Any]:
    """
    Analyzes and plots occurrence counts for each failure mode flag (TWF, HDF, PWF, OSF, RNF).
    """
    failure_counts = {col: int(df[col].sum()) for col in FAILURE_TYPE_COLS}
    failure_pct = {col: (df[col].sum() / len(df)) * 100 for col in FAILURE_TYPE_COLS}
    
    total_failures_with_flag = int(df[FAILURE_TYPE_COLS].sum(axis=1).gt(0).sum())
    total_machine_failures = int(df[TARGET_COL].sum())
    failures_without_flag = int(((df[TARGET_COL] == 1) & (df[FAILURE_TYPE_COLS].sum(axis=1) == 0)).sum())
    multi_flag_failures = int((df[FAILURE_TYPE_COLS].sum(axis=1) > 1).sum())

    labels_map = {
        "HDF": "HDF (Heat Dissipation Failure)",
        "PWF": "PWF (Power Failure)",
        "OSF": "OSF (Overstrain Failure)",
        "TWF": "TWF (Tool Wear Failure)",
        "RNF": "RNF (Random Failure)"
    }
    
    sorted_modes = sorted(failure_counts.items(), key=lambda x: x[1], reverse=True)
    names = [labels_map[k] for k, v in sorted_modes]
    vals = [v for k, v in sorted_modes]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.barh(names, vals, color=PALETTE_FAILURES, edgecolor="black", linewidth=1.1, height=0.55)

    for bar, val in zip(bars, vals):
        pct = (val / len(df)) * 100
        ax.text(
            val + 3,
            bar.get_y() + bar.get_height() / 2,
            f"{val} ({pct:.2f}%)",
            va="center",
            ha="left",
            fontsize=10.5,
            fontweight="bold"
        )

    ax.set_xlim(0, 140)
    ax.set_xlabel("Number of Occurrences (out of 10,000 total cycles)")
    ax.set_title("Distribution of Specific Failure Modes (AI4I 2020)", pad=14)
    ax.grid(axis="x", linestyle="--", alpha=0.7)

    # Informative annotation
    info_text = (
        f"Total Machine Failures: {total_machine_failures}\n"
        f"Failures with >=1 Mode Flag: {total_failures_with_flag}\n"
        f"Unlabeled Machine Failures: {failures_without_flag}\n"
        f"Multi-Mode Failures: {multi_flag_failures}"
    )
    ax.text(
        0.95, 0.25,
        info_text,
        transform=ax.transAxes,
        ha="right",
        va="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#ced4da", alpha=0.9),
        fontsize=9.5
    )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return {
        "failure_mode_counts": failure_counts,
        "failure_mode_percentages": {k: round(v, 4) for k, v in failure_pct.items()},
        "total_machine_failures": total_machine_failures,
        "failures_with_flag": total_failures_with_flag,
        "failures_without_flag": failures_without_flag,
        "multi_flag_failures": multi_flag_failures
    }


def plot_failure_wise_boxplots(df: pd.DataFrame, save_path: Path) -> pd.DataFrame:
    """
    Compares numerical feature distributions between Healthy (0) and Failed (1) machines using Boxplots.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, col in enumerate(NUMERICAL_COLS):
        ax = axes[i]
        sns.boxplot(
            x=TARGET_COL,
            y=col,
            hue=TARGET_COL,
            data=df,
            palette=PALETTE_BINARY,
            ax=ax,
            width=0.45,
            fliersize=3,
            legend=False,
            boxprops=dict(edgecolor="black", linewidth=1.2),
            whiskerprops=dict(linewidth=1.2),
            medianprops=dict(color="yellow", linewidth=2.0)
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Healthy (0)", "Failure (1)"], fontweight="semibold")
        ax.set_title(col, pad=8)
        ax.set_xlabel("Machine Failure Status")
        ax.set_ylabel(col)
        ax.grid(True, linestyle="--", alpha=0.5)

    # Delete 6th empty plot
    fig.delaxes(axes[5])

    fig.suptitle("Comparison of Sensor Operating Parameters by Machine Failure Status", y=0.98, fontsize=15)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(save_path)
    plt.close()

    # Calculate summary comparison table
    summary_by_failure = df.groupby(TARGET_COL)[NUMERICAL_COLS].agg([
        "count", "mean", "std", "median", lambda x: np.percentile(x, 75) - np.percentile(x, 25)
    ]).rename(columns={"<lambda_0>": "IQR"})
    
    return summary_by_failure


def plot_machine_type_vs_failure(df: pd.DataFrame, save_path: Path) -> pd.DataFrame:
    """
    Analyzes and plots failure counts and failure rates across machine product types (L, M, H).
    """
    grouped = df.groupby("Type")[TARGET_COL].agg(
        Total="count",
        Failures="sum",
        Healthy=lambda x: (x == 0).sum(),
        Failure_Rate_Pct=lambda x: (x.sum() / len(x)) * 100
    ).reindex(["L", "M", "H"])

    fig, ax1 = plt.subplots(figsize=(8, 5.5))

    x = np.arange(len(grouped))
    width = 0.45

    bars = ax1.bar(
        x,
        grouped["Failures"],
        width=width,
        color="#D9381E",
        edgecolor="black",
        linewidth=1.1,
        alpha=0.85,
        label="Failure Count"
    )
    ax1.set_ylabel("Failure Count (Observed)", color="#D9381E", labelpad=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Type {t}\n(N={grouped.loc[t, 'Total']:,})" for t in grouped.index], fontweight="semibold")
    ax1.set_ylim(0, 300)

    for bar, val in zip(bars, grouped["Failures"]):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            val + 5,
            f"{val:,}",
            ha="center",
            va="bottom",
            fontweight="bold",
            color="#D9381E"
        )

    # Secondary axis for failure percentage
    ax2 = ax1.twinx()
    line = ax2.plot(
        x,
        grouped["Failure_Rate_Pct"],
        color="#2B5C8F",
        marker="o",
        linewidth=2.5,
        markersize=8,
        label="Failure Rate (%)"
    )
    ax2.set_ylabel("Failure Rate (%)", color="#2B5C8F", labelpad=8)
    ax2.set_ylim(0, 5.0)
    ax2.grid(False)

    for i, rate in enumerate(grouped["Failure_Rate_Pct"]):
        ax2.text(
            i,
            rate + 0.18,
            f"{rate:.2f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            color="#2B5C8F"
        )

    plt.title("Observed Failure Counts & Rates Across Machine Types (L, M, H)", pad=14)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return grouped


def plot_feature_interactions_vs_failure(df: pd.DataFrame, save_path: Path) -> None:
    """
    Plots multi-variable operating regimes that reflect physical failure modes:
    1. Rotational Speed vs Torque (Power / Overstrain Regime)
    2. Air Temp vs Process Temp (Thermal Dissipation Regime)
    3. Tool Wear vs Torque (Mechanical Wear Regime)
    4. Temperature Difference vs Rotational Speed (Heat Dissipation Regime)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    df_copy = df.copy()
    df_copy["Temp_Diff_K"] = df_copy["Process temperature [K]"] - df_copy["Air temperature [K]"]

    # 1. Rotational Speed vs Torque
    ax = axes[0, 0]
    sns.scatterplot(
        data=df_copy,
        x="Rotational speed [rpm]",
        y="Torque [Nm]",
        hue=TARGET_COL,
        palette=PALETTE_BINARY,
        alpha=0.6,
        s=25,
        ax=ax
    )
    ax.set_title("Rotational Speed vs. Torque (Power & Overstrain Boundary)", pad=10)
    ax.set_xlabel("Rotational speed [rpm]")
    ax.set_ylabel("Torque [Nm]")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=["Healthy (0)", "Failure (1)"], loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)

    # 2. Air Temperature vs Process Temperature
    ax = axes[0, 1]
    sns.scatterplot(
        data=df_copy,
        x="Air temperature [K]",
        y="Process temperature [K]",
        hue=TARGET_COL,
        palette=PALETTE_BINARY,
        alpha=0.6,
        s=25,
        ax=ax
    )
    ax.set_title("Air Temperature vs. Process Temperature", pad=10)
    ax.set_xlabel("Air temperature [K]")
    ax.set_ylabel("Process temperature [K]")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=["Healthy (0)", "Failure (1)"], loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)

    # 3. Tool Wear vs Torque
    ax = axes[1, 0]
    sns.scatterplot(
        data=df_copy,
        x="Tool wear [min]",
        y="Torque [Nm]",
        hue=TARGET_COL,
        palette=PALETTE_BINARY,
        alpha=0.6,
        s=25,
        ax=ax
    )
    ax.set_title("Tool Wear vs. Torque (Tool Wear & Overstrain Regimes)", pad=10)
    ax.set_xlabel("Tool wear [min]")
    ax.set_ylabel("Torque [Nm]")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=["Healthy (0)", "Failure (1)"], loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)

    # 4. Temp Difference vs Rotational Speed
    ax = axes[1, 1]
    sns.scatterplot(
        data=df_copy,
        x="Rotational speed [rpm]",
        y="Temp_Diff_K",
        hue=TARGET_COL,
        palette=PALETTE_BINARY,
        alpha=0.6,
        s=25,
        ax=ax
    )
    ax.set_title("Rotational Speed vs. Temperature Difference (ΔT = Process - Air)", pad=10)
    ax.set_xlabel("Rotational speed [rpm]")
    ax.set_ylabel("Temperature Difference ΔT [K]")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=["Healthy (0)", "Failure (1)"], loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("Machine Operating Regimes and Failure Clusters", y=0.98, fontsize=15)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig(save_path)
    plt.close()


def compute_outlier_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes IQR-based outlier boundaries and counts for all numerical features.
    """
    records = []
    for col in NUMERICAL_COLS:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_count = int(outliers_mask.sum())
        outlier_pct = (outlier_count / len(df)) * 100
        
        # Check how many outliers are associated with failure
        failures_in_outliers = int(df.loc[outliers_mask, TARGET_COL].sum())
        failure_rate_in_outliers = (failures_in_outliers / outlier_count * 100) if outlier_count > 0 else 0.0

        records.append({
            "Feature": col,
            "Min": round(df[col].min(), 2),
            "Q1 (25%)": round(q1, 2),
            "Median": round(df[col].median(), 2),
            "Q3 (75%)": round(q3, 2),
            "Max": round(df[col].max(), 2),
            "IQR": round(iqr, 2),
            "Lower Bound": round(lower_bound, 2),
            "Upper Bound": round(upper_bound, 2),
            "Outlier Count": outlier_count,
            "Outlier %": round(outlier_pct, 2),
            "Failures in Outliers": failures_in_outliers,
            "Failure Rate in Outliers (%)": round(failure_rate_in_outliers, 2)
        })

    return pd.DataFrame(records)


def run_eda_pipeline() -> Dict[str, Any]:
    """
    Executes complete EDA pipeline, generates figures, CSV tables, and summary metrics.
    """
    print("=" * 70)
    print("PHASE 3: EXPLORATORY DATA ANALYSIS (EDA) EXECUTION")
    print("=" * 70)

    # Ensure output directories exist
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load raw data
    df_raw = load_raw_data(RAW_DATA_PATH)
    print(f"Loaded raw dataset: {df_raw.shape[0]:,} rows, {df_raw.shape[1]} columns")

    # 2. Target Distribution
    fig01_path = FIGURES_DIR / "01_machine_failure_distribution.png"
    target_stats = plot_target_distribution(df_raw, fig01_path)
    print(f"[Generated] {fig01_path.name}: {target_stats}")

    # 3. Machine Type Distribution
    fig02_path = FIGURES_DIR / "02_machine_type_distribution.png"
    type_stats = plot_machine_type_distribution(df_raw, fig02_path)
    print(f"[Generated] {fig02_path.name}: {type_stats}")

    # 4. Sensor Correlation Heatmap
    fig03_path = FIGURES_DIR / "03_sensor_correlation_heatmap.png"
    corr_matrix = plot_sensor_correlation_heatmap(df_raw, fig03_path)
    print(f"[Generated] {fig03_path.name}")

    # 5. Failure Mode Distribution
    fig04_path = FIGURES_DIR / "04_failure_mode_distribution.png"
    failure_mode_stats = plot_failure_mode_distribution(df_raw, fig04_path)
    print(f"[Generated] {fig04_path.name}: {failure_mode_stats['failure_mode_counts']}")

    # 6. Sensor Distributions
    fig05_path = FIGURES_DIR / "05_sensor_distributions.png"
    plot_sensor_distributions(df_raw, fig05_path)
    print(f"[Generated] {fig05_path.name}")

    # 7. Failure-wise Boxplots
    fig06_path = FIGURES_DIR / "06_features_by_failure_boxplot.png"
    summary_by_failure_df = plot_failure_wise_boxplots(df_raw, fig06_path)
    print(f"[Generated] {fig06_path.name}")

    # 8. Machine Type vs Failure
    fig07_path = FIGURES_DIR / "07_machine_type_vs_failure.png"
    type_failure_df = plot_machine_type_vs_failure(df_raw, fig07_path)
    print(f"[Generated] {fig07_path.name}")

    # 9. Feature Interactions & Operating Regimes
    fig08_path = FIGURES_DIR / "08_feature_interactions_vs_failure.png"
    plot_feature_interactions_vs_failure(df_raw, fig08_path)
    print(f"[Generated] {fig08_path.name}")

    # 10. Compute Numerical Summaries & Tables
    # A. Descriptive Statistics
    desc_stats = df_raw[NUMERICAL_COLS].describe().T
    desc_stats["median"] = df_raw[NUMERICAL_COLS].median()
    desc_stats["skewness"] = df_raw[NUMERICAL_COLS].skew()
    desc_stats["kurtosis"] = df_raw[NUMERICAL_COLS].kurtosis()
    desc_stats_path = TABLES_DIR / "eda_descriptive_statistics.csv"
    desc_stats.to_csv(desc_stats_path)
    print(f"[Saved Table] {desc_stats_path.name}")

    # B. Failure Rate by Machine Type
    type_fail_path = TABLES_DIR / "failure_rate_by_machine_type.csv"
    type_failure_df.to_csv(type_fail_path)
    print(f"[Saved Table] {type_fail_path.name}")

    # C. Feature Summary by Failure
    feat_fail_path = TABLES_DIR / "feature_summary_by_failure.csv"
    summary_by_failure_df.to_csv(feat_fail_path)
    print(f"[Saved Table] {feat_fail_path.name}")

    # D. Failure Mode Counts
    total_failures = failure_mode_stats["total_machine_failures"]
    modes_df = pd.DataFrame([
        {
            "Failure Mode Code": mode,
            "Description": {
                "HDF": "Heat Dissipation Failure",
                "PWF": "Power Failure",
                "OSF": "Overstrain Failure",
                "TWF": "Tool Wear Failure",
                "RNF": "Random Failure"
            }[mode],
            "Occurrence Count": failure_mode_stats["failure_mode_counts"][mode],
            "Prevalence (% of 10,000)": failure_mode_stats["failure_mode_percentages"][mode],
            "Share of Total Failures (%)": round((failure_mode_stats["failure_mode_counts"][mode] / total_failures) * 100, 2)
        }
        for mode in FAILURE_TYPE_COLS
    ])
    modes_path = TABLES_DIR / "failure_mode_counts.csv"
    modes_df.to_csv(modes_path, index=False)
    print(f"[Saved Table] {modes_path.name}")

    # E. Outlier Summary
    outliers_df = compute_outlier_summary(df_raw)
    outliers_path = TABLES_DIR / "eda_outlier_summary.csv"
    outliers_df.to_csv(outliers_path, index=False)
    print(f"[Saved Table] {outliers_path.name}")

    # 11. Compile Comprehensive JSON Summary
    eda_summary = {
        "dataset_metadata": {
            "total_observations": len(df_raw),
            "total_features": len(df_raw.columns),
            "identifiers": ID_COLS,
            "categorical_features": CATEGORICAL_COLS,
            "numerical_features": NUMERICAL_COLS,
            "target_column": TARGET_COL,
            "failure_mode_flags": FAILURE_TYPE_COLS
        },
        "target_distribution": target_stats,
        "machine_type_distribution": type_stats,
        "failure_rate_by_machine_type": {
            k: {
                "total": int(type_failure_df.loc[k, "Total"]),
                "failures": int(type_failure_df.loc[k, "Failures"]),
                "healthy": int(type_failure_df.loc[k, "Healthy"]),
                "failure_rate_pct": round(float(type_failure_df.loc[k, "Failure_Rate_Pct"]), 2)
            }
            for k in type_failure_df.index
        },
        "sensor_correlations": {
            col1: {col2: round(corr_matrix.loc[col1, col2], 4) for col2 in NUMERICAL_COLS}
            for col1 in NUMERICAL_COLS
        },
        "failure_modes_summary": failure_mode_stats,
        "outlier_analysis": outliers_df.to_dict(orient="records"),
        "generated_figures": [
            str(fig01_path.name),
            str(fig02_path.name),
            str(fig03_path.name),
            str(fig04_path.name),
            str(fig05_path.name),
            str(fig06_path.name),
            str(fig07_path.name),
            str(fig08_path.name)
        ],
        "generated_tables": [
            str(desc_stats_path.name),
            str(type_fail_path.name),
            str(feat_fail_path.name),
            str(modes_path.name),
            str(outliers_path.name)
        ]
    }

    metrics_json_path = METRICS_DIR / "eda_summary.json"
    with open(metrics_json_path, "w") as f:
        json.dump(eda_summary, f, indent=4)
    print(f"[Saved JSON Metrics] {metrics_json_path.name}")

    print("=" * 70)
    print("PHASE 3 EDA COMPLETED SUCCESSFULLY")
    print("=" * 70)

    return eda_summary


if __name__ == "__main__":
    run_eda_pipeline()
