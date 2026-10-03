"""
Prediction Engine and Deployment Inference Pipeline for Manufacturing Predictive Maintenance.
Provides robust input validation, preprocessing transformation, model inference (LightGBM),
probability estimation, risk categorization, and batch processing capabilities.
"""
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Union, Tuple, Optional
import pandas as pd
import numpy as np
import joblib

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from config import (
    MODELS_DIR,
    PREPROCESSOR_PATH,
    FEATURE_COLS,
    CATEGORICAL_COLS,
    NUMERICAL_COLS
)
from src.recommendations import generate_maintenance_recommendation, determine_risk_level

PRIMARY_MODEL_PATH = MODELS_DIR / "lightgbm.joblib"

# Dataset-observed sanity ranges (used for soft warnings/sanity checks, not hard physical limits)
SANITY_RANGES = {
    "Air temperature [K]": (280.0, 320.0),
    "Process temperature [K]": (290.0, 330.0),
    "Rotational speed [rpm]": (500.0, 3500.0),
    "Torque [Nm]": (0.0, 120.0),
    "Tool wear [min]": (0.0, 350.0)
}

VALID_MACHINE_TYPES = ["L", "M", "H"]


def load_inference_pipeline(
    model_path: Path = PRIMARY_MODEL_PATH,
    preprocessor_path: Path = PREPROCESSOR_PATH
) -> Tuple[Any, Any]:
    """
    Loads and returns the serialized prediction model (LightGBM) and preprocessing pipeline.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at: {model_path}")
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor artifact not found at: {preprocessor_path}")

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor


def validate_single_input(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates a single machine telemetry input record.
    Checks for required keys, correct data types, allowed categorical values,
    and positive physical bounds.
    """
    if not isinstance(input_dict, dict):
        raise TypeError(f"Input data must be a dictionary, got {type(input_dict).__name__}")

    # Standardize column name keys (support minor naming variations if needed)
    standardized: Dict[str, Any] = {}
    
    # Check Type
    if "Type" not in input_dict:
        raise ValueError("Missing required feature: 'Type'. Expected one of ['L', 'M', 'H'].")
    
    type_val = str(input_dict["Type"]).strip().upper()
    if type_val not in VALID_MACHINE_TYPES:
        raise ValueError(
            f"Invalid Machine Type '{type_val}'. Valid types are: {VALID_MACHINE_TYPES} (L=Low, M=Medium, H=High)."
        )
    standardized["Type"] = type_val

    # Check Numerical features
    for col in NUMERICAL_COLS:
        if col not in input_dict:
            raise ValueError(f"Missing required sensor feature: '{col}'")
        
        raw_val = input_dict[col]
        try:
            val = float(raw_val)
        except (ValueError, TypeError):
            raise TypeError(f"Invalid non-numeric value for '{col}': {raw_val} (type {type(raw_val).__name__})")
        
        if np.isnan(val) or np.isinf(val):
            raise ValueError(f"Sensor value for '{col}' cannot be NaN or Infinite: {val}")

        # Basic physical feasibility checks
        if "temperature" in col.lower() and val <= 0:
            raise ValueError(f"Temperature value for '{col}' must be in positive Kelvin (> 0 K), got: {val}")
        if "speed" in col.lower() and val < 0:
            raise ValueError(f"Rotational speed '{col}' cannot be negative, got: {val}")
        if "torque" in col.lower() and val < 0:
            raise ValueError(f"Torque '{col}' cannot be negative, got: {val}")
        if "wear" in col.lower() and val < 0:
            raise ValueError(f"Tool wear '{col}' cannot be negative, got: {val}")

        standardized[col] = val

    return standardized


def preprocess_features_for_inference(
    df_raw: pd.DataFrame,
    preprocessor: Any
) -> pd.DataFrame:
    """
    Transforms raw input DataFrame through the fitted preprocessor
    and sanitizes column names for LightGBM/tree models.
    """
    transformed = preprocessor.transform(df_raw[FEATURE_COLS])
    feature_names = list(preprocessor.get_feature_names_out())
    
    # Clean brackets for LightGBM compatibility
    clean_feature_names = [col.replace("[", "(").replace("]", ")") for col in feature_names]
    
    transformed_df = pd.DataFrame(
        transformed,
        columns=clean_feature_names,
        index=df_raw.index
    )
    return transformed_df


def predict_single(
    input_data: Dict[str, Any],
    model: Optional[Any] = None,
    preprocessor: Optional[Any] = None,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    End-to-end prediction for a single machine operating record.
    Returns predicted class, probabilities, risk tier, diagnostics, and recommendations.
    """
    # Load pipeline if not provided
    if model is None or preprocessor is None:
        model, preprocessor = load_inference_pipeline()

    # Validate input
    validated_input = validate_single_input(input_data)

    # Convert to DataFrame
    df_input = pd.DataFrame([validated_input])

    # Preprocess
    df_transformed = preprocess_features_for_inference(df_input, preprocessor)

    # Generate prediction & probability
    prob_failure = float(model.predict_proba(df_transformed)[0, 1])
    prob_healthy = float(1.0 - prob_failure)
    pred_class = int(prob_failure >= threshold)

    # Generate recommendation
    rec_info = generate_maintenance_recommendation(
        predicted_class=pred_class,
        failure_probability=prob_failure,
        input_data=validated_input,
        threshold=threshold
    )

    result = {
        "input_telemetry": validated_input,
        "predicted_class": pred_class,
        "status_label": rec_info["status_label"],
        "failure_probability": round(prob_failure, 4),
        "healthy_probability": round(prob_healthy, 4),
        "failure_probability_percentage": f"{prob_failure * 100:.2f}%",
        "healthy_probability_percentage": f"{prob_healthy * 100:.2f}%",
        "risk_level": rec_info["risk_level"],
        "classification_threshold": threshold,
        "summary": rec_info["summary"],
        "diagnostic_flags": rec_info["diagnostic_flags"],
        "recommended_actions": rec_info["recommended_actions"],
        "model_used": "LightGBM Classifier (models/lightgbm.joblib)",
        "disclaimer": rec_info["disclaimer"]
    }

    return result


def predict_batch(
    df: pd.DataFrame,
    model: Optional[Any] = None,
    preprocessor: Optional[Any] = None,
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    Generates batch predictions and probability estimates for a DataFrame of machine records.
    """
    if model is None or preprocessor is None:
        model, preprocessor = load_inference_pipeline()

    # Check required columns
    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Batch input DataFrame is missing required columns: {missing_cols}")

    # Validate batch rows
    validated_rows = []
    for idx, row in df[FEATURE_COLS].iterrows():
        try:
            validated_row = validate_single_input(row.to_dict())
            validated_rows.append(validated_row)
        except Exception as e:
            raise ValueError(f"Validation error at row {idx}: {str(e)}")

    df_valid = pd.DataFrame(validated_rows)
    df_transformed = preprocess_features_for_inference(df_valid, preprocessor)

    # Predict
    probabilities = model.predict_proba(df_transformed)
    failure_probs = probabilities[:, 1]
    healthy_probs = probabilities[:, 0]
    predictions = (failure_probs >= threshold).astype(int)

    risk_levels = [determine_risk_level(p, threshold) for p in failure_probs]
    status_labels = ["Machine Failure" if p == 1 else "Healthy" for p in predictions]

    results_df = df.copy()
    results_df["Predicted_Class"] = predictions
    results_df["Status"] = status_labels
    results_df["Failure_Probability"] = np.round(failure_probs, 4)
    results_df["Healthy_Probability"] = np.round(healthy_probs, 4)
    results_df["Risk_Level"] = risk_levels

    return results_df


if __name__ == "__main__":
    print("Testing Prediction Engine...")
    sample_normal = {
        "Type": "M",
        "Air temperature [K]": 298.2,
        "Process temperature [K]": 308.7,
        "Rotational speed [rpm]": 1408.0,
        "Torque [Nm]": 46.3,
        "Tool wear [min]": 3.0
    }
    pred_res = predict_single(sample_normal)
    print("\n--- Single Prediction Result ---")
    for k, v in pred_res.items():
        print(f"{k}: {v}")
