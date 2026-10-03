"""
Explainable AI (XAI) Module using SHAP (SHapley Additive exPlanations) for LightGBM.
Provides local per-prediction feature attribution, directional contribution analysis,
and global model explainability without modifying the underlying prediction pipeline.
"""
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import joblib
import shap

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    MODELS_DIR,
    PREPROCESSOR_PATH,
    FEATURE_COLS,
    RAW_DATA_PATH
)
from src.prediction import (
    load_inference_pipeline,
    validate_single_input,
    preprocess_features_for_inference,
    predict_single
)
from src.recommendations import determine_risk_level

# Cached SHAP TreeExplainer instance
_CACHED_EXPLAINER: Optional[shap.TreeExplainer] = None
_CACHED_MODEL: Optional[Any] = None

# Feature display mapping dictionary (internal transformed name -> friendly engineering name & unit)
FEATURE_DISPLAY_MAP = {
    "Type_H": {"label": "Machine Quality Type (H - High)", "unit": "Binary Flag (1=True, 0=False)", "raw_feature": "Type"},
    "Type_L": {"label": "Machine Quality Type (L - Low)", "unit": "Binary Flag (1=True, 0=False)", "raw_feature": "Type"},
    "Type_M": {"label": "Machine Quality Type (M - Medium)", "unit": "Binary Flag (1=True, 0=False)", "raw_feature": "Type"},
    "Air temperature (K)": {"label": "Air Temperature", "unit": "K", "raw_feature": "Air temperature [K]"},
    "Process temperature (K)": {"label": "Process Temperature", "unit": "K", "raw_feature": "Process temperature [K]"},
    "Rotational speed (rpm)": {"label": "Rotational Speed", "unit": "rpm", "raw_feature": "Rotational speed [rpm]"},
    "Torque (Nm)": {"label": "Spindle Torque", "unit": "Nm", "raw_feature": "Torque [Nm]"},
    "Tool wear (min)": {"label": "Tool Wear Duration", "unit": "min", "raw_feature": "Tool wear [min]"}
}

TRANSFORMED_TO_DISPLAY_NAMES = {
    k: v["label"] for k, v in FEATURE_DISPLAY_MAP.items()
}


def get_transformed_feature_names(preprocessor: Any) -> List[str]:
    """
    Extracts the sanitized transformed feature column names from the fitted preprocessor.
    """
    try:
        raw_names = list(preprocessor.get_feature_names_out())
        cleaned_names = []
        for name in raw_names:
            col = name.split("__")[-1]
            col = col.replace("[", "(").replace("]", ")")
            cleaned_names.append(col)
        return cleaned_names
    except Exception:
        # Authoritative known transformed ordering
        return [
            "Type_H", "Type_L", "Type_M",
            "Air temperature (K)", "Process temperature (K)",
            "Rotational speed (rpm)", "Torque (Nm)", "Tool wear (min)"
        ]


def get_tree_explainer(model: Any) -> shap.TreeExplainer:
    """
    Initializes or returns the cached shap.TreeExplainer for the serialized LightGBM model.
    """
    global _CACHED_EXPLAINER, _CACHED_MODEL
    if _CACHED_EXPLAINER is not None and _CACHED_MODEL is model:
        return _CACHED_EXPLAINER

    _CACHED_EXPLAINER = shap.TreeExplainer(model)
    _CACHED_MODEL = model
    return _CACHED_EXPLAINER


def explain_prediction(
    input_data: Dict[str, Any],
    model: Optional[Any] = None,
    preprocessor: Optional[Any] = None,
    threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Computes per-prediction SHAP feature attributions for a single machine operating record.
    Preserves exact consistency with the validated predict_single() pipeline.
    
    Returns:
    - input_telemetry: validated raw inputs
    - predicted_class: 0 (Healthy) or 1 (Machine Failure)
    - failure_probability: float probability [0.0, 1.0]
    - risk_level / risk_tier: Low / Moderate / High / Critical
    - base_value: model expected value (log-odds bias)
    - feature_contributions / contributions: ordered list of feature contribution dictionaries
    - top_contributions: top contributing features sorted by absolute SHAP magnitude
    - shap_values: list of raw class-1 SHAP contribution floats
    - disclaimer: research & interpretation notice
    """
    # 1. Load pipeline if not provided
    if model is None or preprocessor is None:
        model, preprocessor = load_inference_pipeline()

    # 2. Validate input and preprocess using the authoritative prediction engine
    validated_input = validate_single_input(input_data)
    df_input = pd.DataFrame([validated_input])
    df_transformed = preprocess_features_for_inference(df_input, preprocessor)

    # 3. Compute authoritative prediction & probabilities (reused from model)
    prob_failure = float(model.predict_proba(df_transformed)[0, 1])
    pred_class = int(prob_failure >= threshold)
    risk_level = determine_risk_level(prob_failure, threshold=threshold)
    status_label = "Machine Failure" if pred_class == 1 else "Healthy"

    # 4. Generate SHAP values via TreeExplainer
    explainer = get_tree_explainer(model)
    shap_raw = explainer.shap_values(df_transformed)

    # For binary classification TreeExplainer returns a list of ndarrays [class_0_shap, class_1_shap]
    if isinstance(shap_raw, list):
        shap_values_class1 = shap_raw[1][0]
    else:
        shap_values_class1 = shap_raw[0]

    # Base value (model expected log-odds value)
    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = float(base_val[1] if len(base_val) > 1 else base_val[0])
    else:
        base_val = float(base_val)

    # 5. Map transformed features to friendly display format
    feature_cols = list(df_transformed.columns)
    contributions: List[Dict[str, Any]] = []
    shap_floats: List[float] = []

    for col_name, shap_val in zip(feature_cols, shap_values_class1):
        shap_float = float(shap_val)
        shap_floats.append(shap_float)
        meta = FEATURE_DISPLAY_MAP.get(col_name, {
            "label": col_name,
            "unit": "",
            "raw_feature": col_name
        })

        # Determine current input value string
        raw_feat = meta["raw_feature"]
        if col_name.startswith("Type_"):
            target_type = col_name.split("_")[1]
            actual_type = validated_input.get("Type", "")
            current_val_str = f"Type={actual_type} ({'Active' if actual_type == target_type else 'Inactive'})"
            numeric_val = 1.0 if actual_type == target_type else 0.0
        else:
            val_num = validated_input.get(raw_feat, df_transformed[col_name].iloc[0])
            unit_str = meta["unit"]
            numeric_val = float(val_num)
            if isinstance(val_num, float):
                current_val_str = f"{val_num:.2f} {unit_str}".strip()
            else:
                current_val_str = f"{val_num} {unit_str}".strip()

        # Directional attribution labeling
        if shap_float > 0.0001:
            direction = "Increases Failure Risk"
            direction_code = "RISK_INCREASE"
        elif shap_float < -0.0001:
            direction = "Reduces Failure Risk"
            direction_code = "RISK_REDUCE"
        else:
            direction = "Neutral"
            direction_code = "NEUTRAL"

        contributions.append({
            "feature": col_name,
            "transformed_feature": col_name,
            "display_name": meta["label"],
            "raw_feature": raw_feat,
            "value": numeric_val,
            "current_value": current_val_str,
            "unit": meta["unit"],
            "contribution": round(shap_float, 4),
            "shap_value": round(shap_float, 4),
            "abs_contribution": round(abs(shap_float), 4),
            "abs_shap_value": round(abs(shap_float), 4),
            "direction": direction,
            "direction_code": direction_code
        })

    # 6. Sort contributions by absolute SHAP magnitude descending
    contributions_sorted = sorted(contributions, key=lambda x: x["abs_shap_value"], reverse=True)
    top_5 = contributions_sorted[:5]

    return {
        "input_telemetry": validated_input,
        "predicted_class": pred_class,
        "status_label": status_label,
        "failure_probability": round(prob_failure, 4),
        "failure_probability_percentage": f"{prob_failure * 100:.2f}%",
        "risk_level": risk_level,
        "risk_tier": risk_level,
        "classification_threshold": threshold,
        "base_value": round(base_val, 4),
        "feature_contributions": contributions_sorted,
        "contributions": contributions_sorted,
        "top_contributions": top_5,
        "transformed_feature_names": feature_cols,
        "shap_values": shap_floats,
        "explainer_type": "shap.TreeExplainer (LightGBM)",
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "disclaimer": (
            "SHAP estimates how individual input features contribute to the model's output "
            "for this benchmark observation. Positive contributions push the prediction toward "
            "higher failure risk; negative contributions push it toward lower risk. "
            "SHAP values reflect model behavior and do not represent physical causality or certified CNC mechanics."
        )
    }


def compute_global_shap_summary(
    sample_df: Optional[pd.DataFrame] = None,
    model: Optional[Any] = None,
    preprocessor: Optional[Any] = None,
    n_samples: int = 100
) -> pd.DataFrame:
    """
    Computes global SHAP feature importance (mean absolute SHAP values) across a representative sample.
    Returns a pandas DataFrame sorted in descending order of mean |SHAP|.
    """
    if model is None or preprocessor is None:
        model, preprocessor = load_inference_pipeline()

    if sample_df is None and RAW_DATA_PATH.exists():
        raw_df = pd.read_csv(RAW_DATA_PATH)
        sample_df = raw_df[FEATURE_COLS].iloc[:n_samples].copy()
    elif sample_df is None:
        raise FileNotFoundError("Raw benchmark dataset not found for global SHAP calculation.")

    df_trans = preprocess_features_for_inference(sample_df, preprocessor)
    explainer = get_tree_explainer(model)
    shap_raw = explainer.shap_values(df_trans)

    if isinstance(shap_raw, list):
        shap_vals_c1 = shap_raw[1]
    else:
        shap_vals_c1 = shap_raw

    # Mean absolute SHAP per transformed feature
    mean_abs_shap = np.mean(np.abs(shap_vals_c1), axis=0)
    feature_names = list(df_trans.columns)

    global_importance = []
    for name, score in zip(feature_names, mean_abs_shap):
        meta = FEATURE_DISPLAY_MAP.get(name, {"label": name, "unit": ""})
        global_importance.append({
            "feature": name,
            "transformed_feature": name,
            "display_name": meta["label"],
            "mean_abs_shap": round(float(score), 4)
        })

    global_df = pd.DataFrame(global_importance)
    global_df = global_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return global_df
