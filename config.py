"""
Configuration settings for the Manufacturing Analytics Predictive Maintenance System.
"""
from pathlib import Path

# Base project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "ai4i2020.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
METRICS_DIR = OUTPUTS_DIR / "metrics"
REPORTS_DIR = OUTPUTS_DIR / "reports"
DRIFT_DIR = OUTPUTS_DIR / "drift"

# Random Seed for Reproducibility
RANDOM_STATE = 42

# Column Definitions
ID_COLS = ["UDI", "Product ID"]
TARGET_COL = "Machine failure"
FAILURE_TYPE_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
CATEGORICAL_COLS = ["Type"]
NUMERICAL_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]
FEATURE_COLS = CATEGORICAL_COLS + NUMERICAL_COLS

# Preprocessor artifact path
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"

