# AI-Based Predictive Maintenance System for CNC Machines in Automotive Manufacturing Using Industrial Sensor Data

## Domain
**Manufacturing Analytics / Industry 4.0 / Machine Health Prognostics**

## Project Team
- **Pragadeshwaran** – 231801126
- **Sanjey Ram Kumar** – 231801153
- **Santhosh G** – 231801154

*Department of Artificial Intelligence and Data Science*  
*Rajalakshmi Engineering College*

---

## 1. Project Overview & Objective
This project establishes a comprehensive, leakage-free Machine Learning Predictive Maintenance pipeline for industrial Computer Numerical Control (CNC) machines. By monitoring operating parameters—including ambient and process temperatures, rotational speed, torque, and cumulative tool wear—the system predicts impending machine failure before catastrophic breakdown occurs.

The system is trained and evaluated on the **AI4I 2020 Predictive Maintenance Dataset** (a 10,000-observation benchmark).

---

## 2. Dataset & Feature Engineering

### Target Variable
- `Machine failure` (Binary: `0` = Healthy, `1` = Machine Failure)
- **Class Distribution**: 9,661 Healthy (96.61%) vs. 339 Failures (3.39%) — Imbalance Ratio: **28.50 : 1**

### Predictive Operating Features (6 Raw Inputs ➔ 8 Encoded Features)
1. `Type`: Machine Quality Variant (`L` = Low 60%, `M` = Medium 30%, `H` = High 10%) ➔ One-Hot Encoded (`Type_H`, `Type_L`, `Type_M`)
2. `Air temperature [K]`: Ambient temperature (passthrough unscaled physical units)
3. `Process temperature [K]`: Cutting zone temperature (passthrough unscaled physical units)
4. `Rotational speed [rpm]`: Spindle motor speed (passthrough unscaled physical units)
5. `Torque [Nm]`: Spindle mechanical torque (passthrough unscaled physical units)
6. `Tool wear [min]`: Cumulative cutter usage time (passthrough unscaled physical units)

### Excluded Features (Data Leakage Prevention)
- **Identifiers**: `UDI`, `Product ID` (non-predictive metadata)
- **Failure Mode Flags**: `TWF`, `HDF`, `PWF`, `OSF`, `RNF` (direct failure consequence flags excluded from primary machine failure prediction)

---

## 3. Preprocessing & Resampling Pipeline
- **Stratified Train/Test Split**: 80% Train (8,000 rows, 271 failures) / 20% Test (2,000 rows, 68 failures) with fixed `random_state=42`.
- **Leakage-Safe Preprocessor**: Scikit-learn `ColumnTransformer` fitted **strictly** on `X_train` and serialized to `models/preprocessor.joblib`.
- **Isolated SMOTE**: Applied strictly to the preprocessed training set, generating a balanced training set ($N=15,458$, 7,729 healthy / 7,729 failures). The test set remains untouched at its natural 3.40% failure prevalence.

---

## 4. Empirical Model Benchmarking & Results

All models were evaluated on the **untouched test set** ($N=2,000$, 68 actual failures, 1,932 healthy):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | TN | FP | FN | TP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree** | 0.9515 | 0.3782 | 0.6618 | 0.4813 | 0.8292 | 0.3874 | 1,858 | 74 | 23 | 45 |
| **Random Forest** | 0.9535 | 0.3913 | 0.6618 | 0.4918 | 0.9596 | 0.5810 | 1,862 | 70 | 23 | 45 |
| **XGBoost** | 0.9675 | 0.5143 | 0.7941 | 0.6243 | 0.9723 | 0.7751 | 1,881 | 51 | 14 | 54 |
| **LightGBM (Champion)** | **0.9695** | **0.5340** | **0.8088** | **0.6433** | **0.9726** | **0.7963** | **1,884** | **48** | **13** | **55** |
| **Stacking Ensemble** | **0.9695** | **0.5361** | 0.7647 | 0.6303 | 0.9712 | 0.7247 | **1,887** | **45** | 16 | 52 |

### Final Model Selection Rationale
- **LightGBM** is selected as the **Primary Production Model**:
  - Highest **Recall** (**80.88%** — capturing 55 of 68 test failures).
  - Lowest **False Negatives** (**13** missed failures), minimizing unpredicted downtime risk.
  - Highest **F1-Score** (**0.6433**) and **PR-AUC** (**0.7963**).
- **Stacking Ensemble**: Combined the 4 base learners with a Logistic Regression meta-learner using 5-fold Stratified CV out-of-fold probabilities. While Stacking achieved slightly higher Precision (53.61% vs 53.40%), it missed 3 additional failures (16 FN vs 13 FN).

---

## 5. Decision Support & Risk Tiers

The prediction engine (`src/prediction.py` and `src/recommendations.py`) maps model posterior failure probabilities $P(\text{Failure})$ to operational risk tiers:

- **Low Risk ($P < 0.20$)**: Nominal operating conditions; continue scheduled maintenance.
- **Moderate Risk ($0.20 \le P < 0.50$)**: Sub-threshold drift; monitor telemetry on next shift.
- **High Risk ($0.50 \le P < 0.80$)**: Machine at risk; schedule cutter inspection and torque load check.
- **Critical Risk ($P \ge 0.80$)**: Immediate breakdown risk; priority maintenance inspection alert.

---

## 6. Project Directory Structure
```text
Manufacturing_Analytics/
├── app/
│   └── app.py                       # Streamlit dashboard application
├── config.py                        # Project configurations, paths, and constants
├── data/
│   ├── raw/
│   │   └── ai4i2020.csv             # Verified raw dataset
│   └── processed/
│       ├── X_train.csv              # Transformed training features (8,000 x 8)
│       ├── X_train_smote.csv        # SMOTE-balanced training features (15,458 x 8)
│       ├── X_test.csv               # Untouched test features (2,000 x 8)
│       ├── y_train.csv              # Training targets (8,000)
│       ├── y_train_smote.csv        # SMOTE training targets (15,458)
│       └── y_test.csv               # Untouched test targets (2,000)
├── models/
│   ├── preprocessor.joblib          # Fitted ColumnTransformer
│   ├── decision_tree.joblib         # Serialized Decision Tree
│   ├── random_forest.joblib         # Serialized Random Forest
│   ├── xgboost.joblib               # Serialized XGBoost
│   ├── lightgbm.joblib              # Serialized LightGBM (Primary Model)
│   └── stacking_ensemble.joblib     # Serialized Stacking Ensemble
├── notebooks/
│   ├── 01_dataset_inspection.ipynb  # Phase 1 dataset quality inspection
│   ├── 02_eda.ipynb                 # Phase 3 exploratory data analysis
│   ├── 03_preprocessing.ipynb       # Phase 2 preprocessing & train/test split
│   ├── 04_model_training.ipynb      # Phase 4 baseline model development
│   └── 06_stacking_ensemble.ipynb   # Phase 5 stacking ensemble & evaluation
├── outputs/
│   ├── figures/                     # 12 Publication-quality 300 DPI figures
│   ├── tables/                      # Summary CSV tables
│   ├── metrics/                     # JSON metrics and evaluation results
│   └── reports/                     # Markdown validation reports
├── scripts/
│   └── predict_single.py            # CLI single-record prediction tool
├── src/
│   ├── __init__.py
│   ├── data_loader.py               # Ingestion and schema integrity module
│   ├── preprocessing.py             # Preprocessing & SMOTE pipeline module
│   ├── eda.py                       # Exploratory analytics and plotting module
│   ├── train_models.py              # ML model training and evaluation module
│   ├── ensemble.py                  # Stacking ensemble and CV module
│   ├── prediction.py                # Reusable inference engine & batch API
│   └── recommendations.py           # Domain heuristic maintenance guidance
├── tests/
│   ├── run_all_tests.py             # Unified test runner (43/43 checks)
│   ├── test_preprocessing.py        # Phase 2 test suite
│   ├── test_models.py               # Phase 4 test suite
│   ├── test_ensemble.py             # Phase 5 test suite
│   ├── test_prediction.py           # Phase 6 test suite
│   └── test_app.py                  # Phase 7 test suite
├── requirements.txt                 # Clean Python dependencies
└── README.md                        # Complete project documentation
```

---

## 7. Setup & Execution

### 1. Environment Setup
```bash
# Clone or navigate to the project directory
cd Manufacturing_Analytics

# Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Test Suite (43 automated checks)
```bash
python tests/run_all_tests.py
```

### 3. Single-Record CLI Inference
```bash
python scripts/predict_single.py --type M --air_temp 298.2 --process_temp 308.7 --speed 1408 --torque 46.3 --tool_wear 3.0
```

### 4. Launch Streamlit Analytics Dashboard
```bash
streamlit run app/app.py
```

---

## 8. Limitations & Research Disclaimer
1. **Synthetic Benchmark Nature**: The AI4I 2020 dataset is a realistic simulated benchmark dataset reflecting industrial machine telemetry. It is not raw sensor telemetry harvested from a live automotive assembly plant.
2. **Prototype Boundary**: This application is an analytical decision-support prototype. It does not interface with physical SCADA/MES industrial control systems, PLC controllers, or hardware emergency stop relays.
3. **Operational Risk Tiers**: Risk tier thresholds (Low, Moderate, High, Critical) are heuristic interpretation bands based on model probability distribution and have not been certified under industrial machinery standards (e.g., ISO 13849 / IEC 61508).
4. **Generalization Scope**: Model performance on this synthetic dataset does not guarantee equivalent precision/recall on different industrial machine tools with different vibration, cutting fluid, or material characteristics without domain re-calibration.
