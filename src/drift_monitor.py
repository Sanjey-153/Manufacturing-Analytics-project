"""
Drift Monitoring Engine for Manufacturing Telemetry Analytics (Future Work F3).
Tracks input feature distribution drift, categorical distribution drift, model-output probability drift,
risk-tier distribution drift, and data-quality anomalies against the fixed authoritative reference baseline.
"""
import os
import sys
import json
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    RAW_DATA_PATH,
    OUTPUTS_DIR,
    DRIFT_DIR,
    FEATURE_COLS,
    CATEGORICAL_COLS,
    NUMERICAL_COLS,
    RANDOM_STATE
)
from src.data_loader import load_raw_data
from src.preprocessing import separate_features_and_target, split_data
from src.prediction import (
    load_inference_pipeline,
    preprocess_features_for_inference,
    predict_single,
    predict_batch,
    SANITY_RANGES,
    VALID_MACHINE_TYPES
)

# Prototype Drift Threshold Constants (Heuristic monitoring thresholds for prototype)
PSI_STABLE_UPPER = 0.10
PSI_WATCH_UPPER = 0.20

STATUS_INSUFFICIENT_DATA = "INSUFFICIENT DATA"
STATUS_STABLE = "STABLE"
STATUS_WATCH = "WATCH"
STATUS_DRIFT_DETECTED = "DRIFT DETECTED"

# Human-friendly feature labels & units
FEATURE_METADATA = {
    "Type": {"name": "Machine Type", "unit": "Category (H/M/L)", "type": "categorical"},
    "Air temperature [K]": {"name": "Air Temperature", "unit": "K", "type": "numerical"},
    "Process temperature [K]": {"name": "Process Temperature", "unit": "K", "type": "numerical"},
    "Rotational speed [rpm]": {"name": "Rotational Speed", "unit": "rpm", "type": "numerical"},
    "Torque [Nm]": {"name": "Spindle Torque", "unit": "Nm", "type": "numerical"},
    "Tool wear [min]": {"name": "Tool Wear Duration", "unit": "min", "type": "numerical"},
    "failure_probability": {"name": "Model Output Probability", "unit": "Probability [0, 1]", "type": "probability"}
}

# Cached profile instances in memory
_CACHED_REFERENCE_PROFILE: Optional[Dict[str, Any]] = None
_CACHED_REFERENCE_PREDICTIONS: Optional[List[float]] = None


def get_drift_dir() -> Path:
    """Returns the path to the drift output directory, ensuring it exists."""
    d_dir = DRIFT_DIR if DRIFT_DIR is not None else (OUTPUTS_DIR / "drift")
    d_dir.mkdir(parents=True, exist_ok=True)
    return d_dir


def calculate_psi(
    reference_data: Union[np.ndarray, List[float], pd.Series],
    current_data: Union[np.ndarray, List[float], pd.Series],
    bin_edges: Optional[Union[np.ndarray, List[float]]] = None,
    num_bins: int = 10,
    epsilon: float = 1e-4
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Computes Population Stability Index (PSI) between reference and current numerical distributions
    using fixed reference quantile bin edges.
    
    Formula: PSI = sum((actual_prop - expected_prop) * ln(actual_prop / expected_prop))
    """
    ref_arr = np.asarray(reference_data, dtype=float)
    curr_arr = np.asarray(current_data, dtype=float)

    # Filter out non-finite values for statistical calculation
    ref_clean = ref_arr[np.isfinite(ref_arr)]
    curr_clean = curr_arr[np.isfinite(curr_arr)]

    if len(ref_clean) == 0 or len(curr_clean) == 0:
        return 0.0, []

    # Derive bin edges from reference if not explicitly provided
    if bin_edges is None:
        quantiles = np.linspace(0, 100, num_bins + 1)
        raw_edges = np.percentile(ref_clean, quantiles)
        # Deduplicate edges to handle identical quantiles
        unique_edges = np.unique(raw_edges)
        if len(unique_edges) < 2:
            unique_edges = np.array([ref_clean.min() - 1.0, ref_clean.max() + 1.0])
        bin_edges = unique_edges

    # Set outer boundaries to -inf and +inf for robust binning
    edges = list(bin_edges)
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref_clean, bins=edges)
    curr_counts, _ = np.histogram(curr_clean, bins=edges)

    ref_total = len(ref_clean)
    curr_total = len(curr_clean)

    # Proportions with epsilon smoothing to avoid log(0) or divide by 0
    ref_props = np.maximum(ref_counts / ref_total, epsilon)
    curr_props = np.maximum(curr_counts / curr_total, epsilon)

    # Normalize after smoothing
    ref_props /= ref_props.sum()
    curr_props /= curr_props.sum()

    # Calculate PSI per bin
    bin_psi = (curr_props - ref_props) * np.log(curr_props / ref_props)
    total_psi = float(np.sum(bin_psi))

    bin_details = []
    for i in range(len(ref_counts)):
        low_label = "-inf" if i == 0 else f"{edges[i]:.2f}"
        high_label = "+inf" if i == len(ref_counts) - 1 else f"{edges[i+1]:.2f}"
        bin_details.append({
            "bin_index": i,
            "range": f"[{low_label}, {high_label})",
            "ref_count": int(ref_counts[i]),
            "ref_prop": float(round(ref_props[i], 4)),
            "curr_count": int(curr_counts[i]),
            "curr_prop": float(round(curr_props[i], 4)),
            "psi_contribution": float(round(bin_psi[i], 5))
        })

    return round(total_psi, 4), bin_details


def calculate_categorical_psi(
    reference_categories: Union[List[str], pd.Series],
    current_categories: Union[List[str], pd.Series],
    valid_categories: Optional[List[str]] = None,
    epsilon: float = 1e-4
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Computes Categorical PSI for discrete variables (e.g. Type: 'H', 'M', 'L').
    """
    if valid_categories is None:
        valid_categories = ["H", "M", "L"]

    ref_s = pd.Series(reference_categories).astype(str)
    curr_s = pd.Series(current_categories).astype(str)

    all_cats = list(valid_categories)
    # Include any unexpected categories found in current data
    for cat in curr_s.unique():
        if cat not in all_cats:
            all_cats.append(cat)

    ref_counts = ref_s.value_counts()
    curr_counts = curr_s.value_counts()

    ref_total = len(ref_s) if len(ref_s) > 0 else 1
    curr_total = len(curr_s) if len(curr_s) > 0 else 1

    ref_props_list = []
    curr_props_list = []

    for cat in all_cats:
        ref_props_list.append(ref_counts.get(cat, 0) / ref_total)
        curr_props_list.append(curr_counts.get(cat, 0) / curr_total)

    ref_props = np.maximum(np.array(ref_props_list), epsilon)
    curr_props = np.maximum(np.array(curr_props_list), epsilon)

    ref_props /= ref_props.sum()
    curr_props /= curr_props.sum()

    cat_psi = (curr_props - ref_props) * np.log(curr_props / ref_props)
    total_psi = float(np.sum(cat_psi))

    cat_details = []
    for i, cat in enumerate(all_cats):
        cat_details.append({
            "category": cat,
            "ref_count": int(ref_counts.get(cat, 0)),
            "ref_prop": float(round(ref_props[i], 4)),
            "curr_count": int(curr_counts.get(cat, 0)),
            "curr_prop": float(round(curr_props[i], 4)),
            "psi_contribution": float(round(cat_psi[i], 5)),
            "is_valid_type": cat in valid_categories
        })

    return round(total_psi, 4), cat_details


def calculate_ks_test(
    reference_data: Union[np.ndarray, List[float], pd.Series],
    current_data: Union[np.ndarray, List[float], pd.Series]
) -> Tuple[float, float]:
    """
    Computes the two-sample Kolmogorov-Smirnov test statistic and p-value.
    """
    ref_clean = np.asarray(reference_data, dtype=float)
    ref_clean = ref_clean[np.isfinite(ref_clean)]
    curr_clean = np.asarray(current_data, dtype=float)
    curr_clean = curr_clean[np.isfinite(curr_clean)]

    if len(ref_clean) == 0 or len(curr_clean) == 0:
        return 0.0, 1.0

    ks_res = stats.ks_2samp(ref_clean, curr_clean)
    return round(float(ks_res.statistic), 4), round(float(ks_res.pvalue), 6)


# Application Input Validation Envelopes (F1 / System-wide Operational Bounds)
APPLICATION_INPUT_ENVELOPES = {
    "Air temperature [K]": (280.0, 320.0),
    "Process temperature [K]": (290.0, 330.0),
    "Rotational speed [rpm]": (500.0, 3500.0),
    "Torque [Nm]": (0.0, 120.0),
    "Tool wear [min]": (0.0, 350.0)
}


def check_data_quality(df_current: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluates telemetry data quality across missing values, invalid types,
    unexpected categories, and out-of-envelope sensor readings.
    Distinguishes invalid inputs (violating application validation envelopes)
    from valid values that merely deviate from the historical reference distribution.
    """
    total_events = len(df_current)
    if total_events == 0:
        return {
            "total_events": 0,
            "is_clean": True,
            "quality_status": "CLEAN",
            "missing_count": 0,
            "nan_count": 0,
            "infinite_count": 0,
            "unexpected_categories_count": 0,
            "range_violations_count": 0,
            "total_issues": 0,
            "issue_details": []
        }

    issues: List[str] = []
    missing_count = 0
    nan_count = 0
    inf_count = 0
    unexpected_cat_count = 0
    range_violation_count = 0

    # 1. Missing columns or missing values
    for col in FEATURE_COLS:
        if col not in df_current.columns:
            missing_count += total_events
            issues.append(f"Missing required feature column '{col}'")
        else:
            col_nulls = int(df_current[col].isnull().sum())
            if col_nulls > 0:
                missing_count += col_nulls
                issues.append(f"Feature '{col}' has {col_nulls} null/missing values")

    # 2. NaN and Inf in numerical features
    for col in NUMERICAL_COLS:
        if col in df_current.columns:
            col_data = pd.to_numeric(df_current[col], errors='coerce')
            nans = int(col_data.isna().sum())
            if nans > 0:
                nan_count += nans
                issues.append(f"Feature '{col}' contains {nans} non-numeric/NaN values")
            infs = int(np.isinf(col_data).sum()) if hasattr(np, 'isinf') else 0
            if infs > 0:
                inf_count += infs
                issues.append(f"Feature '{col}' contains {infs} infinite values")

    # 3. Categorical validation for Type
    if "Type" in df_current.columns:
        invalid_types = df_current[~df_current["Type"].astype(str).isin(VALID_MACHINE_TYPES)]
        if len(invalid_types) > 0:
            unexpected_cat_count = len(invalid_types)
            unique_invalids = invalid_types["Type"].unique().tolist()
            issues.append(f"Unexpected Machine Type categories found: {unique_invalids} ({unexpected_cat_count} occurrences)")

    # 4. Out-of-envelope application input validation limits
    for col, (min_val, max_val) in APPLICATION_INPUT_ENVELOPES.items():
        if col in df_current.columns:
            numeric_vals = pd.to_numeric(df_current[col], errors='coerce').dropna()
            out_of_bounds = numeric_vals[(numeric_vals < min_val) | (numeric_vals > max_val)]
            if len(out_of_bounds) > 0:
                cnt = len(out_of_bounds)
                range_violation_count += cnt
                issues.append(f"Feature '{col}' has {cnt} values outside configured application envelope [{min_val}, {max_val}]")

    total_issues = missing_count + nan_count + inf_count + unexpected_cat_count + range_violation_count
    is_clean = (total_issues == 0)

    if total_issues == 0:
        quality_status = "CLEAN"
    elif unexpected_cat_count > 0 or missing_count > 0 or nan_count > 0:
        quality_status = "ERROR"
    else:
        quality_status = "WARNING"

    return {
        "total_events": total_events,
        "is_clean": is_clean,
        "quality_status": quality_status,
        "missing_count": missing_count,
        "nan_count": nan_count,
        "infinite_count": inf_count,
        "unexpected_categories_count": unexpected_cat_count,
        "range_violations_count": range_violation_count,
        "total_issues": total_issues,
        "issue_details": issues
    }


def classify_psi_status(psi_value: float) -> str:
    """Classifies a PSI value into prototype interpretation bands."""
    if psi_value < PSI_STABLE_UPPER:
        return STATUS_STABLE
    elif psi_value < PSI_WATCH_UPPER:
        return STATUS_WATCH
    else:
        return STATUS_DRIFT_DETECTED


def build_reference_profile(force_recompute: bool = False) -> Dict[str, Any]:
    """
    Constructs and persists the authoritative fixed reference profile from the validated
    training split (80% stratified sample of ai4i2020.csv, N=8,000 observations).
    """
    global _CACHED_REFERENCE_PROFILE, _CACHED_REFERENCE_PREDICTIONS
    drift_dir = get_drift_dir()
    profile_path = drift_dir / "reference_profile.json"
    pred_path = drift_dir / "reference_predictions.json"

    if not force_recompute and profile_path.exists() and pred_path.exists():
        try:
            with open(profile_path, "r", encoding="utf-8") as f:
                profile = json.load(f)
            with open(pred_path, "r", encoding="utf-8") as f:
                preds = json.load(f)
            _CACHED_REFERENCE_PROFILE = profile
            _CACHED_REFERENCE_PREDICTIONS = preds
            return profile
        except Exception:
            pass  # Recompute if reading failed

    # 1. Load raw dataset and separate features and target
    raw_df = load_raw_data()
    X, y = separate_features_and_target(raw_df)

    # 2. Extract strictly the training partition (8,000 observations)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=RANDOM_STATE)

    # 3. Load authoritative model and preprocessor to generate reference predictions
    model, preprocessor = load_inference_pipeline()
    X_train_trans = preprocess_features_for_inference(X_train, preprocessor)
    ref_probs = model.predict_proba(X_train_trans)[:, 1].tolist()

    # 4. Build feature statistical profiles
    features_profile: Dict[str, Any] = {}

    # Categorical Type profile
    type_counts = X_train["Type"].value_counts().to_dict()
    type_props = X_train["Type"].value_counts(normalize=True).to_dict()
    features_profile["Type"] = {
        "feature_name": "Type",
        "display_name": "Machine Type",
        "type": "categorical",
        "unit": "Category (H/M/L)",
        "sample_count": len(X_train),
        "categories": ["H", "M", "L"],
        "category_counts": type_counts,
        "category_proportions": {k: round(float(v), 4) for k, v in type_props.items()}
    }

    # Numerical features profiles
    for col in NUMERICAL_COLS:
        col_vals = X_train[col].to_numpy(dtype=float)
        quantiles = np.percentile(col_vals, np.linspace(0, 100, 11))
        unique_edges = np.unique(quantiles)
        if len(unique_edges) < 2:
            unique_edges = np.array([col_vals.min() - 1.0, col_vals.max() + 1.0])

        features_profile[col] = {
            "feature_name": col,
            "display_name": FEATURE_METADATA.get(col, {}).get("name", col),
            "type": "numerical",
            "unit": FEATURE_METADATA.get(col, {}).get("unit", ""),
            "sample_count": len(col_vals),
            "mean": round(float(np.mean(col_vals)), 4),
            "std": round(float(np.std(col_vals)), 4),
            "median": round(float(np.median(col_vals)), 4),
            "min": round(float(np.min(col_vals)), 4),
            "max": round(float(np.max(col_vals)), 4),
            "p25": round(float(np.percentile(col_vals, 25)), 4),
            "p75": round(float(np.percentile(col_vals, 75)), 4),
            "bin_edges": [round(float(e), 4) for e in unique_edges]
        }

    # Model Output failure probability profile
    prob_arr = np.asarray(ref_probs, dtype=float)
    prob_quantiles = np.percentile(prob_arr, np.linspace(0, 100, 11))
    prob_edges = np.unique(prob_quantiles)
    if len(prob_edges) < 2:
        prob_edges = np.array([0.0, 0.5, 1.0])

    # Risk tier distribution in reference baseline
    low_count = int(np.sum(prob_arr < 0.20))
    mod_count = int(np.sum((prob_arr >= 0.20) & (prob_arr < 0.50)))
    high_count = int(np.sum((prob_arr >= 0.50) & (prob_arr < 0.80)))
    crit_count = int(np.sum(prob_arr >= 0.80))
    total_ref = len(prob_arr)

    model_output_profile = {
        "feature_name": "failure_probability",
        "display_name": "Model Output Failure Probability",
        "type": "probability",
        "unit": "Probability [0, 1]",
        "sample_count": total_ref,
        "mean": round(float(np.mean(prob_arr)), 4),
        "std": round(float(np.std(prob_arr)), 4),
        "median": round(float(np.median(prob_arr)), 4),
        "min": round(float(np.min(prob_arr)), 4),
        "max": round(float(np.max(prob_arr)), 4),
        "bin_edges": [round(float(e), 4) for e in prob_edges],
        "risk_tier_distribution": {
            "Low": {"count": low_count, "proportion": round(low_count / total_ref, 4)},
            "Moderate": {"count": mod_count, "proportion": round(mod_count / total_ref, 4)},
            "High": {"count": high_count, "proportion": round(high_count / total_ref, 4)},
            "Critical": {"count": crit_count, "proportion": round(crit_count / total_ref, 4)}
        }
    }

    healthy_count = int((y_train == 0).sum())
    failure_count = int((y_train == 1).sum())

    profile_doc = {
        "metadata": {
            "project": "AI-Based Predictive Maintenance System for CNC Machines",
            "reference_dataset": "AI4I 2020",
            "reference_split": "validated stratified training partition",
            "random_state": RANDOM_STATE,
            "reference_rows": len(X_train),
            "reference_sample_size": len(X_train),
            "healthy_rows": healthy_count,
            "failure_rows": failure_count,
            "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "is_fixed_reference": True
        },
        "features": features_profile,
        "model_output": model_output_profile
    }

    # Persist reference files
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile_doc, f, indent=2)

    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(ref_probs, f)

    _CACHED_REFERENCE_PROFILE = profile_doc
    _CACHED_REFERENCE_PREDICTIONS = ref_probs
    return profile_doc


def load_reference_profile() -> Dict[str, Any]:
    """Loads and caches the fixed reference profile."""
    global _CACHED_REFERENCE_PROFILE
    if _CACHED_REFERENCE_PROFILE is not None:
        return _CACHED_REFERENCE_PROFILE
    return build_reference_profile(force_recompute=False)


def get_reference_predictions() -> List[float]:
    """Returns the cached list of baseline reference failure probabilities."""
    global _CACHED_REFERENCE_PREDICTIONS
    if _CACHED_REFERENCE_PREDICTIONS is not None:
        return _CACHED_REFERENCE_PREDICTIONS
    build_reference_profile(force_recompute=False)
    if _CACHED_REFERENCE_PREDICTIONS is not None:
        return _CACHED_REFERENCE_PREDICTIONS
    return []


def evaluate_drift(
    current_data: Union[pd.DataFrame, List[Dict[str, Any]]],
    reference_profile: Optional[Dict[str, Any]] = None,
    min_sample_size: int = 50,
    model: Optional[Any] = None,
    preprocessor: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Evaluates data drift, model-output probability drift, and data quality on a window of telemetry observations.
    
    Returns structured drift report:
    - overall_status: INSUFFICIENT DATA / STABLE / WATCH / DRIFT DETECTED
    - window_size: number of events evaluated
    - data_quality: quality metrics and issues
    - feature_drift: dictionary of per-feature PSI, KS, and status
    - model_output_drift: probability PSI, KS, and status
    - risk_tier_comparison: reference vs current risk tier proportions
    - most_affected_feature: feature with peak PSI
    """
    if reference_profile is None:
        reference_profile = load_reference_profile()

    # Convert to DataFrame if list of dicts
    if isinstance(current_data, list):
        df_curr = pd.DataFrame(current_data)
    else:
        df_curr = current_data.copy()

    total_events = len(df_curr)

    # 1. Data Quality Check
    quality_report = check_data_quality(df_curr)

    # 2. Check for Insufficient Data
    if total_events < min_sample_size:
        return {
            "overall_status": STATUS_INSUFFICIENT_DATA,
            "window_size": total_events,
            "min_sample_size": min_sample_size,
            "monitoring_status_code": "INSUFFICIENT_DATA",
            "message": f"Window size ({total_events}) is below minimum sample size ({min_sample_size}). Telemetry is accumulating.",
            "data_quality": quality_report,
            "feature_drift": {},
            "model_output_drift": {},
            "risk_tier_comparison": {},
            "most_affected_feature": None,
            "max_psi": 0.0,
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "disclaimer": (
                "Drift monitoring compares the current telemetry distribution with a fixed reference distribution. "
                "It does not establish causal degradation, true industrial process drift, or model-performance deterioration "
                "without validated longitudinal outcomes."
            )
        }

    # 3. Load Raw Reference Training Data for KS tests
    raw_df = load_raw_data()
    X_raw, y_raw = separate_features_and_target(raw_df)
    X_train_ref, _, _, _ = split_data(X_raw, y_raw, test_size=0.2, random_state=RANDOM_STATE)
    ref_predictions = get_reference_predictions()

    # 4. Feature Drift Calculation
    feature_drift_results: Dict[str, Any] = {}
    psi_values: List[Tuple[str, float]] = []

    # A. Categorical Type Drift
    if "Type" in df_curr.columns:
        type_psi, type_bins = calculate_categorical_psi(
            reference_categories=X_train_ref["Type"],
            current_categories=df_curr["Type"],
            valid_categories=VALID_MACHINE_TYPES
        )
        type_status = classify_psi_status(type_psi)
        feature_drift_results["Type"] = {
            "feature": "Type",
            "display_name": "Machine Type",
            "type": "categorical",
            "unit": "Category",
            "psi": type_psi,
            "status": type_status,
            "ks_statistic": None,
            "ks_pvalue": None,
            "ref_distribution": reference_profile["features"]["Type"]["category_proportions"],
            "curr_distribution": {
                k: round(float(v), 4)
                for k, v in df_curr["Type"].value_counts(normalize=True).to_dict().items()
            },
            "bin_details": type_bins
        }
        psi_values.append(("Machine Type", type_psi))

    # B. Numerical Features Drift
    for col in NUMERICAL_COLS:
        if col in df_curr.columns:
            ref_col_data = X_train_ref[col].to_numpy(dtype=float)
            curr_col_data = pd.to_numeric(df_curr[col], errors='coerce').dropna().to_numpy(dtype=float)
            ref_feat_meta = reference_profile["features"].get(col, {})
            bin_edges = ref_feat_meta.get("bin_edges")

            # Calculate PSI
            col_psi, col_bins = calculate_psi(
                reference_data=ref_col_data,
                current_data=curr_col_data,
                bin_edges=bin_edges
            )
            # Calculate KS Test
            ks_stat, ks_pval = calculate_ks_test(ref_col_data, curr_col_data)
            col_status = classify_psi_status(col_psi)

            feature_drift_results[col] = {
                "feature": col,
                "display_name": FEATURE_METADATA.get(col, {}).get("name", col),
                "type": "numerical",
                "unit": FEATURE_METADATA.get(col, {}).get("unit", ""),
                "psi": col_psi,
                "status": col_status,
                "ks_statistic": ks_stat,
                "ks_pvalue": ks_pval,
                "ref_mean": ref_feat_meta.get("mean"),
                "ref_median": ref_feat_meta.get("median"),
                "curr_mean": round(float(np.mean(curr_col_data)), 4) if len(curr_col_data) > 0 else 0.0,
                "curr_median": round(float(np.median(curr_col_data)), 4) if len(curr_col_data) > 0 else 0.0,
                "bin_details": col_bins
            }
            psi_values.append((FEATURE_METADATA.get(col, {}).get("name", col), col_psi))

    # 5. Model Output (Failure Probability) Drift
    if "failure_probability" in df_curr.columns:
        curr_probs = pd.to_numeric(df_curr["failure_probability"], errors='coerce').dropna().to_numpy(dtype=float)
    else:
        # Run inference if probabilities not present in dataframe
        if model is None or preprocessor is None:
            model, preprocessor = load_inference_pipeline()
        curr_trans = preprocess_features_for_inference(df_curr[FEATURE_COLS], preprocessor)
        curr_probs = model.predict_proba(curr_trans)[:, 1]

    ref_prob_meta = reference_profile.get("model_output", {})
    prob_bin_edges = ref_prob_meta.get("bin_edges")

    prob_psi, prob_bins = calculate_psi(
        reference_data=ref_predictions,
        current_data=curr_probs,
        bin_edges=prob_bin_edges
    )
    prob_ks_stat, prob_ks_pval = calculate_ks_test(ref_predictions, curr_probs)
    prob_status = classify_psi_status(prob_psi)

    model_output_drift = {
        "feature": "failure_probability",
        "display_name": "Model Output Failure Probability",
        "psi": prob_psi,
        "status": prob_status,
        "ks_statistic": prob_ks_stat,
        "ks_pvalue": prob_ks_pval,
        "ref_mean": ref_prob_meta.get("mean"),
        "ref_median": ref_prob_meta.get("median"),
        "curr_mean": round(float(np.mean(curr_probs)), 4) if len(curr_probs) > 0 else 0.0,
        "curr_median": round(float(np.median(curr_probs)), 4) if len(curr_probs) > 0 else 0.0,
        "bin_details": prob_bins
    }
    psi_values.append(("Model Output Probability", prob_psi))

    # 6. Risk Tier Distribution Comparison
    curr_low = int(np.sum(curr_probs < 0.20))
    curr_mod = int(np.sum((curr_probs >= 0.20) & (curr_probs < 0.50)))
    curr_high = int(np.sum((curr_probs >= 0.50) & (curr_probs < 0.80)))
    curr_crit = int(np.sum(curr_probs >= 0.80))
    curr_total_probs = len(curr_probs) if len(curr_probs) > 0 else 1

    ref_risk_tiers = ref_prob_meta.get("risk_tier_distribution", {})
    risk_tier_comparison = {}
    for tier, curr_count in [("Low", curr_low), ("Moderate", curr_mod), ("High", curr_high), ("Critical", curr_crit)]:
        curr_prop = round(curr_count / curr_total_probs, 4)
        ref_prop = ref_risk_tiers.get(tier, {}).get("proportion", 0.0)
        diff_pp = round((curr_prop - ref_prop) * 100, 2)  # Percentage points
        risk_tier_comparison[tier] = {
            "tier": tier,
            "ref_proportion": ref_prop,
            "ref_percentage": f"{ref_prop * 100:.1f}%",
            "curr_count": curr_count,
            "curr_proportion": curr_prop,
            "curr_percentage": f"{curr_prop * 100:.1f}%",
            "difference_pp": diff_pp
        }

    # 7. Overall Drift Status Classification
    # Rule:
    # DRIFT DETECTED if any feature PSI >= 0.20 or probability PSI >= 0.20
    # WATCH if any PSI >= 0.10 or data quality warning
    # STABLE if all PSI < 0.10 and clean data quality
    max_psi_item = max(psi_values, key=lambda x: x[1]) if psi_values else ("None", 0.0)
    most_affected_feature, max_psi = max_psi_item

    has_drift = any(psi >= PSI_WATCH_UPPER for _, psi in psi_values)
    has_watch = any(psi >= PSI_STABLE_UPPER for _, psi in psi_values)
    has_quality_issue = not quality_report["is_clean"]

    if has_drift:
        overall_status = STATUS_DRIFT_DETECTED
    elif has_watch or has_quality_issue:
        overall_status = STATUS_WATCH
    else:
        overall_status = STATUS_STABLE

    return {
        "overall_status": overall_status,
        "window_size": total_events,
        "min_sample_size": min_sample_size,
        "monitoring_status_code": overall_status.replace(" ", "_"),
        "most_affected_feature": most_affected_feature,
        "max_psi": max_psi,
        "data_quality": quality_report,
        "feature_drift": feature_drift_results,
        "model_output_drift": model_output_drift,
        "risk_tier_comparison": risk_tier_comparison,
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "disclaimer": (
            "Drift monitoring compares the current telemetry distribution with a fixed reference distribution. "
            "It does not establish causal degradation, true industrial process drift, or model-performance deterioration "
            "without validated longitudinal outcomes."
        )
    }


class DriftAlertTracker:
    """
    Manages transition-based drift alerts, suppressing consecutive duplicate alerts
    and recording status transition histories.
    """
    def __init__(self):
        self.current_status: str = STATUS_INSUFFICIENT_DATA
        self.previous_status: Optional[str] = None
        self.alerts_history: List[Dict[str, Any]] = []
        self.transition_log: List[Dict[str, Any]] = []

    def update(self, new_status: str, drift_report: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Updates tracker state with new drift assessment status.
        Generates an alert ONLY on meaningful state transitions.
        """
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

        # Initial state setup
        if self.previous_status is None and new_status == STATUS_INSUFFICIENT_DATA:
            self.current_status = new_status
            return None

        # State transition occurred
        if new_status != self.current_status:
            old_status = self.current_status
            self.previous_status = old_status
            self.current_status = new_status

            transition_record = {
                "timestamp": timestamp,
                "from_status": old_status,
                "to_status": new_status,
                "transition_label": f"{old_status} ➔ {new_status}"
            }
            self.transition_log.insert(0, transition_record)

            # Generate alert on elevated drift transitions (e.g. STABLE -> WATCH, WATCH -> DRIFT, STABLE -> DRIFT)
            if new_status in [STATUS_WATCH, STATUS_DRIFT_DETECTED]:
                alert = {
                    "alert_id": f"DRIFT_ALT_{len(self.alerts_history)+1:03d}",
                    "timestamp": timestamp,
                    "previous_status": old_status,
                    "current_status": new_status,
                    "severity": "CRITICAL" if new_status == STATUS_DRIFT_DETECTED else "WARNING",
                    "most_affected_feature": drift_report.get("most_affected_feature") if drift_report else "Unknown",
                    "max_psi": drift_report.get("max_psi", 0.0) if drift_report else 0.0,
                    "message": (
                        f"Drift Alert: System transitioned from {old_status} to {new_status}. "
                        f"Primary driver: {drift_report.get('most_affected_feature', 'Unknown')} "
                        f"(PSI: {drift_report.get('max_psi', 0.0):.4f})."
                    ) if drift_report else f"Drift Alert: Transitioned from {old_status} to {new_status}."
                }
                self.alerts_history.insert(0, alert)
                return alert
            elif old_status in [STATUS_WATCH, STATUS_DRIFT_DETECTED] and new_status == STATUS_STABLE:
                # Recovery transition recorded
                recovery_event = {
                    "alert_id": f"DRIFT_REC_{len(self.alerts_history)+1:03d}",
                    "timestamp": timestamp,
                    "previous_status": old_status,
                    "current_status": new_status,
                    "severity": "INFO",
                    "message": f"Drift Recovery: System normalized back to {STATUS_STABLE}."
                }
                self.alerts_history.insert(0, recovery_event)
                return recovery_event

        return None

    def get_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        return self.alerts_history[:limit]

    def get_transitions(self, limit: int = 10) -> List[Dict[str, Any]]:
        return self.transition_log[:limit]

    def reset(self):
        self.current_status = STATUS_INSUFFICIENT_DATA
        self.previous_status = None
        self.alerts_history.clear()
        self.transition_log.clear()
