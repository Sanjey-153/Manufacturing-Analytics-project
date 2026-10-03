# Final System Integration & Validation Report

**Project Title:** AI-Based Predictive Maintenance System for CNC Machines in Automotive Manufacturing Using Industrial Sensor Data  
**Team:** Pragadeshwaran (231801126), Sanjey Ram Kumar (231801153), Santhosh G (231801154)  
**Department:** Artificial Intelligence and Data Science, Rajalakshmi Engineering College  
**System Status:** **PASS (Production & Research Ready)**  
**Date:** September 2026  

---

## 1. Dataset & Integrity Audit
- **Source Dataset:** `data/raw/ai4i2020.csv` (AI4I 2020 Predictive Maintenance Dataset)
- **Total Population:** 10,000 instances, 14 original columns
- **Data Quality:** 0 missing values, 0 duplicate rows, verified schema
- **Target Variable:** `Machine failure` (9,661 Healthy / 339 Failures — 28.50:1 imbalance ratio)
- **Excluded Features:** `UDI`, `Product ID` (non-predictive metadata), `TWF`, `HDF`, `PWF`, `OSF`, `RNF` (target failure consequence flags strictly excluded to prevent data leakage)
- **Operating Predictive Features (6 raw ➔ 8 encoded):**
  - `Type` (One-Hot Encoded ➔ `Type_H`, `Type_L`, `Type_M`)
  - `Air temperature [K]`
  - `Process temperature [K]`
  - `Rotational speed [rpm]`
  - `Torque [Nm]`
  - `Tool wear [min]`

---

## 2. Preprocessing & Train/Test Pipeline
- **Stratified Split:** 80% Train ($N=8,000$, 271 failures, 3.39%) / 20% Test ($N=2,000$, 68 failures, 3.40%) with `random_state=42`.
- **Fitted Pipeline:** `models/preprocessor.joblib` fitted **strictly** on `X_train`.
- **Resampling Strategy:** SMOTE applied **strictly to training set** ($N=15,458$, 7,729 healthy / 7,729 failures).
- **Test Set Isolation:** Test set remained untouched at 2,000 samples and natural 3.40% failure prevalence.

---

## 3. Empirical Model Evaluation (Untouched Test Set $N=2,000$)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | TN | FP | FN | TP | Rank / Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Decision Tree** | 0.9515 | 0.3782 | 0.6618 | 0.4813 | 0.8292 | 0.3874 | 1,858 | 74 | 23 | 45 | Baseline |
| **Random Forest** | 0.9535 | 0.3913 | 0.6618 | 0.4918 | 0.9596 | 0.5810 | 1,862 | 70 | 23 | 45 | Bagging Baseline |
| **XGBoost** | 0.9675 | 0.5143 | 0.7941 | 0.6243 | 0.9723 | 0.7751 | 1,881 | 51 | 14 | 54 | 2nd Best Individual |
| **LightGBM** | **0.9695** | **0.5340** | **0.8088** | **0.6433** | **0.9726** | **0.7963** | **1,884** | **48** | **13** | **55** | **CHAMPION / PRIMARY** |
| **Stacking Ensemble** | **0.9695** | **0.5361** | 0.7647 | 0.6303 | 0.9712 | 0.7247 | **1,887** | **45** | 16 | 52 | Verified Benchmark |

---

## 4. Model Selection & Stacking Ensemble Conclusion
1. **Primary Deployment Model (LightGBM)**:
   - Highest Recall (**80.88%** — capturing 55 out of 68 actual test failures).
   - Lowest False Negatives (**13 FN**), minimizing catastrophic breakdown risk.
   - Highest F1-Score (**0.6433**) and PR-AUC (**0.7963**).
2. **Stacking Ensemble Evaluation**:
   - Stacking (DT + RF + XGB + LGBM ➔ Logistic Regression meta-learner with 5-fold Stratified CV) yielded slightly higher Precision (+0.21%, 45 vs 48 FP) but lower Recall (-4.41%, 52 vs 55 TP) and 3 additional False Negatives (16 vs 13 FN).
   - **Conclusion**: LightGBM was selected as the final production model because predictive maintenance heavily prioritizes catching failures (Recall/FN) over small false-alarm differences.

---

## 5. Prediction Engine & Maintenance Decision Support
- **Inference Module (`src/prediction.py`)**: End-to-end single-instance and batch prediction with input validation, non-numeric checks, sanity range checks, and probability extraction.
- **Decision Support Module (`src/recommendations.py`)**: Operational risk classification:
  - **Low Risk**: $P < 0.20$
  - **Moderate Risk**: $0.20 \le P < 0.50$
  - **High Risk**: $0.50 \le P < 0.80$
  - **Critical Risk**: $P \ge 0.80$
- **Heuristic Diagnostic Flags**: High Tool Wear ($\ge 200\text{ min}$), High Torque ($\ge 60\text{ Nm}$), Thermal Gradient Anomaly ($\Delta T < 8.6\text{ K}$), Abnormal Speed.

---

## 6. Streamlit Dashboard Verification
- **Entry Point:** `app/app.py` (`streamlit run app/app.py`)
- **Functional Pages:**
  - Overview KPI dashboard
  - Interactive Single-Machine Telemetry Inference
  - Batch CSV Telemetry Processing
  - Manufacturing Analytics (EDA)
  - Model Performance & Comparison
  - Feature Importance & Explainability
  - Project Architecture & Disclaimers
- **Performance:** Model and preprocessor loaded via `@st.cache_resource`. Zero runtime retraining. Instant sub-second inference latency.
- **Browser Subagent Test:** Fully verified via autonomous browser execution.

---

## 7. Complete Test Suite Execution Summary
- **Unified Test Runner:** `tests/run_all_tests.py`
- **Total Test Checks:** **43**
- **Passed:** **43 (100%)**
- **Failed:** **0**
- **Execution Time:** ~4.2 seconds
- **Breakdown:**
  - Phase 2 Preprocessing & Leakage Checks: 10/10 PASS
  - Phase 4 ML Model Serialization & Metric Replication: 3/3 PASS
  - Phase 5 Stacking Ensemble & Out-of-Fold CV: 10/10 PASS
  - Phase 6 Prediction Engine & Validation: 10/10 PASS
  - Phase 7 Streamlit Application & Component Integrity: 10/10 PASS

---

## 8. Known Limitations & Research Boundaries
1. **Synthetic Telemetry Benchmark**: Trained and evaluated on the AI4I 2020 simulated dataset. Not harvested from physical automotive production lines.
2. **Decision-Support Scope**: The software is an analytical recommendation tool and does not interface with PLC controllers or SCADA industrial hardware.
3. **Prototype Risk Boundaries**: Risk tiers (Low, Moderate, High, Critical) are application-level interpretation heuristics rather than certified ISO/IEC machinery standards.

---

## 9. Reproducibility Instructions
```bash
# 1. Setup Environment
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 2. Run Preprocessing Pipeline
python src/preprocessing.py

# 3. Run Exploratory Data Analysis
python src/eda.py

# 4. Train & Benchmark ML Models
python src/train_models.py

# 5. Train & Evaluate Stacking Ensemble
python src/ensemble.py

# 6. Execute Unified Test Suite
python tests/run_all_tests.py

# 7. Launch Interactive Dashboard
streamlit run app/app.py
```
