"""
F8 Regression Test Suite: Verifies Bug 1 (TelemetrySimulator get_history) and Bug 2 (Threshold & Cost Sweep parameter binding).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

import joblib

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.telemetry_simulator import TelemetrySimulator, MODE_REPLAY, MODE_SCENARIO
from src.threshold_analysis import (
    generate_threshold_sweep,
    find_cost_minimizing_threshold,
    calculate_threshold_metrics,
    compare_thresholds
)


def run_f8_regression_tests():
    print("=" * 70)
    print("RUNNING F8 POST-INTEGRATION FUNCTIONAL REGRESSION TEST SUITE")
    print("=" * 70)

    # 1. Load trained models
    model = joblib.load(PROJECT_ROOT / "models" / "lightgbm.joblib")
    preprocessor = joblib.load(PROJECT_ROOT / "models" / "preprocessor.joblib")

    # --- BUG 1 REGRESSION: TelemetrySimulator get_history API ---
    sim = TelemetrySimulator(mode=MODE_REPLAY, max_history=50, model=model, preprocessor=preprocessor)
    
    # Verify initial empty history
    assert sim.get_history() == []
    assert len(sim.get_history_df()) == 0

    # Step simulator multiple times
    for _ in range(15):
        sim.step()

    # Test full history accessor
    hist_all = sim.get_history()
    assert isinstance(hist_all, list)
    assert len(hist_all) == 15
    assert hist_all[-1]["step"] == 15

    # Test limit slice accessor
    hist_limited = sim.get_history(limit=5)
    assert len(hist_limited) == 5
    assert hist_limited[-1]["step"] == 15
    assert hist_limited[0]["step"] == 11

    # Verify get_history_df still returns matching DataFrame
    df_hist = sim.get_history_df(limit=5)
    assert len(df_hist) == 5
    assert df_hist.iloc[-1]["step"] == 15

    print("[PASS] Regression Check 1: TelemetrySimulator.get_history() and get_history_df() verified.")

    # --- BUG 2 REGRESSION: Threshold Analysis API & Cost Parameter Passing ---
    np.random.seed(42)
    y_true = np.random.choice([0, 1], size=200, p=[0.95, 0.05])
    y_prob = np.random.uniform(0.0, 1.0, size=200)

    # Verify generate_threshold_sweep works with direct cost parameters
    df_sweep = generate_threshold_sweep(
        y_true=y_true,
        y_prob=y_prob,
        c_tn=0.0,
        c_fp=15.0,
        c_fn=120.0,
        c_tp=5.0,
        step=0.01
    )
    assert isinstance(df_sweep, pd.DataFrame)
    assert len(df_sweep) == 101
    assert "Expected_Cost" in df_sweep.columns
    assert "Total_Cost" in df_sweep.columns
    assert "Threshold" in df_sweep.columns

    # Verify find_cost_minimizing_threshold works with df_sweep
    best_opt = find_cost_minimizing_threshold(df_sweep)
    assert isinstance(best_opt, dict)
    assert "Threshold" in best_opt
    assert "Expected_Cost" in best_opt
    assert 0.0 <= float(best_opt["Threshold"]) <= 1.0

    # Verify calculate_threshold_metrics baseline at 0.50
    m_050 = calculate_threshold_metrics(y_true, y_prob, threshold=0.50, c_tn=0.0, c_fp=15.0, c_fn=120.0, c_tp=5.0)
    assert m_050["Threshold"] == 0.50
    assert "Recall" in m_050
    assert "Precision" in m_050

    print("[PASS] Regression Check 2: generate_threshold_sweep & find_cost_minimizing_threshold signatures verified.")

    print("\n" + "=" * 70)
    print("ALL F8 REGRESSION TESTS PASSED (2/2)")
    print("=" * 70)


if __name__ == "__main__":
    run_f8_regression_tests()
