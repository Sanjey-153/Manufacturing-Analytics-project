"""
Unified Test Runner for the Manufacturing Analytics Predictive Maintenance System.
Executes all 5 test suites across Preprocessing, Baseline Models, Stacking Ensemble,
Prediction Engine, and Streamlit Application components.
"""
import sys
import time
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import tests.test_preprocessing as t_prep
import tests.test_models as t_mod
import tests.test_ensemble as t_ens
import tests.test_prediction as t_pred
import tests.test_app as t_app
import tests.test_telemetry_simulator as t_sim
import tests.test_explainability as t_exp
import tests.test_drift_monitor as t_drift
import tests.test_threshold_analysis as t_thresh
import tests.test_work_orders as t_wo
import tests.test_time_series_pipeline as t_ts
import tests.test_industrial_validation as t_ind
import tests.test_f8_regression as t_f8


def run_complete_project_test_suite():
    print("=" * 80)
    print("MANUFACTURING ANALYTICS PREDICTIVE MAINTENANCE: COMPLETE SYSTEM TEST SUITE")
    print("=" * 80)

    start_time = time.time()
    suite_results = {}

    suites = [
        ("Phase 2: Data Preprocessing & Leakage Checks (10 checks)", t_prep.run_all_validation_checks, 10),
        ("Phase 4: ML Models & Metric Replication (3 checks)", t_mod.test_models_and_evaluation, 3),
        ("Phase 5: Stacking Ensemble & CV Out-of-Fold (10 checks)", t_ens.run_ensemble_validation, 10),
        ("Phase 6: Prediction Engine & Input Validation (10 checks)", t_pred.run_all_prediction_tests, 10),
        ("Phase 7: Streamlit Dashboard & Application (10 checks)", t_app.run_app_tests, 10),
        ("Future Work F1: Simulated Telemetry & Hardened Streaming Engine (12 checks)", t_sim.run_telemetry_simulator_tests, 12),
        ("Future Work F2: SHAP Explainability Engine & Feature Attributions (14 checks)", t_exp.run_explainability_tests, 14),
        ("Future Work F3: Drift Monitoring Engine & Population Stability (22 checks)", t_drift.run_drift_monitor_tests, 22),
        ("Future Work F4: Cost-Based Threshold Framework & Sensitivity (22 checks)", t_thresh.run_threshold_analysis_tests, 22),
        ("Future Work F5: Maintenance Work-Order Prototype (15 checks)", t_wo.run_work_orders_tests, 15),
        ("Future Work F6: Advanced Time-Series Architecture (18 checks)", t_ts.run_time_series_pipeline_tests, 18),
        ("Future Work F7: Real Industrial Validation Readiness (20 checks)", t_ind.run_industrial_validation_tests, 20),
        ("F8 Regression: Post-Integration Functional Fixes (2 checks)", t_f8.run_f8_regression_tests, 2)
    ]

    total_tests = sum(count for _, _, count in suites)
    total_passed = 0
    total_failed = 0

    for name, runner, count in suites:
        print(f"\n>>> Executing {name}...")
        try:
            runner()
            suite_results[name] = {"status": "PASSED", "checks": count}
            total_passed += count
        except Exception as e:
            suite_results[name] = {"status": "FAILED", "error": str(e), "checks": count}
            total_failed += count
            print(f"[FAIL] Error in {name}: {e}")

    elapsed = time.time() - start_time

    print("\n" + "=" * 80)
    print("FINAL TEST EXECUTION SUMMARY")
    print("=" * 80)
    for name, res in suite_results.items():
        status = res["status"]
        checks = res["checks"]
        icon = "[PASS]" if status == "PASSED" else "[FAIL]"
        print(f"{icon} {name:<60}: {status} ({checks}/{checks} checks)")

    print("-" * 80)
    print(f"Total Test Checks: {total_tests}")
    print(f"Passed:            {total_passed}")
    print(f"Failed:            {total_failed}")
    print(f"Execution Time:    {elapsed:.2f} seconds")
    print("=" * 80)

    return {
        "total_tests": total_tests,
        "total_passed": total_passed,
        "total_failed": total_failed,
        "execution_time_sec": round(elapsed, 2),
        "status": "ALL_PASSED" if total_failed == 0 else "FAILURES_DETECTED"
    }


if __name__ == "__main__":
    summary = run_complete_project_test_suite()
    if summary["total_failed"] > 0:
        sys.exit(1)
