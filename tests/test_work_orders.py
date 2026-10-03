"""
Test Suite for F5 Maintenance Work-Order Prototype.
Validates work-order creation, lifecycle transitions, priority mapping,
human-in-the-loop review enforcement, persistence, and CSV export.
"""

import os
import sys
import shutil
from pathlib import Path
import pandas as pd
import numpy as np

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.work_orders import (
    create_work_order_draft,
    update_work_order_status,
    get_all_work_orders,
    get_work_order_by_id,
    export_work_orders_to_csv,
    generate_work_order_id,
    get_priority_info,
    VALID_STATUSES,
    ALLOWED_TRANSITIONS,
    WORK_ORDERS_DIR,
    WORK_ORDERS_FILE
)
from src.prediction import predict_single


def run_work_orders_tests():
    print("=" * 70)
    print("TEST SUITE: F5 MAINTENANCE WORK-ORDER PROTOTYPE")
    print("=" * 70)

    # Setup isolated test store
    backup_file = None
    if WORK_ORDERS_FILE.exists():
        backup_file = WORK_ORDERS_DIR / "work_orders_backup.json"
        shutil.copyfile(WORK_ORDERS_FILE, backup_file)

    try:
        # Reset store for clean deterministic testing
        if WORK_ORDERS_FILE.exists():
            WORK_ORDERS_FILE.unlink()

        # 1. Work-order creation
        sample_pred = {
            "failure_probability": 0.8500,
            "predicted_class": 1,
            "risk_level": "Critical",
            "recommended_actions": ["Halt cutting cycle and inspect tool wear land."],
            "diagnostic_flags": ["High Spindle Torque (68.0 Nm)"],
            "summary": "CRITICAL ALERT: High failure risk detected."
        }
        wo = create_work_order_draft(
            prediction_result=sample_pred,
            machine_reference="CNC-M01",
            source="02 Machine Prediction",
            simulation_or_data_source="BENCHMARK",
            notes="Test draft creation."
        )
        assert wo is not None, "Failed to create work order draft"
        print("[PASS] Test 1: Work-order draft creation verified.")

        # 2. Unique ID generation
        id1 = generate_work_order_id(0)
        id2 = generate_work_order_id(1)
        assert id1 != id2, "Generated IDs must be unique"
        assert id1.startswith("WO-"), f"ID must follow format WO-YYYYMMDD-XXXX, got {id1}"
        print(f"[PASS] Test 2: Structured unique ID generation verified ({id1}, {id2}).")

        # 3. Required fields presence
        required_keys = [
            "work_order_id", "created_timestamp", "source", "machine_reference",
            "simulation_or_data_source", "prediction_probability", "predicted_class",
            "risk_tier", "priority", "recommended_action", "reason", "current_status",
            "assigned_role", "due_window", "notes", "created_by", "human_review_required"
        ]
        for k in required_keys:
            assert k in wo, f"Missing required field '{k}' in work order"
        print("[PASS] Test 3: All required work-order fields verified.")

        # 4. Valid risk mapping
        assert wo["risk_tier"] == "Critical", f"Expected Critical risk, got {wo['risk_tier']}"
        print("[PASS] Test 4: Risk tier propagation verified.")

        # 5. Valid priority mapping
        p_info_crit = get_priority_info("Critical")
        p_info_low = get_priority_info("Low")
        assert p_info_crit["priority"] == "Urgent Review"
        assert p_info_low["priority"] == "Routine Monitoring"
        assert wo["priority"] == "Urgent Review"
        print(f"[PASS] Test 5: Priority mapping verified (Critical -> {wo['priority']}).")

        # 6. Recommendation reuse
        assert "Halt cutting cycle" in wo["recommended_action"][0]
        print("[PASS] Test 6: Recommendation engine reuse verified.")

        # 7. Valid status transition
        wo_id = wo["work_order_id"]
        success, msg, wo_updated = update_work_order_status(wo_id, "REVIEW_PENDING", comment="Lead review")
        assert success is True, f"Valid transition to REVIEW_PENDING failed: {msg}"
        assert wo_updated["current_status"] == "REVIEW_PENDING"

        success2, msg2, wo_updated2 = update_work_order_status(wo_id, "APPROVED", comment="Lead approved")
        assert success2 is True, f"Valid transition to APPROVED failed: {msg2}"
        assert wo_updated2["current_status"] == "APPROVED"
        print("[PASS] Test 7: Valid lifecycle transitions (DRAFT -> REVIEW_PENDING -> APPROVED) verified.")

        # 8. Invalid status transition rejection
        # From APPROVED, cannot jump directly to COMPLETED (must go via SCHEDULED -> IN_PROGRESS)
        invalid_succ, invalid_msg, _ = update_work_order_status(wo_id, "COMPLETED")
        assert invalid_succ is False, "Invalid transition directly from APPROVED to COMPLETED should be rejected"
        print(f"[PASS] Test 8: Invalid status transition correctly rejected ({invalid_msg}).")

        # 9. Persistence
        assert WORK_ORDERS_FILE.exists(), "Work orders store file must exist on disk"
        print("[PASS] Test 9: Work-order disk persistence verified.")

        # 10. Reload verification
        reloaded = get_work_order_by_id(wo_id)
        assert reloaded is not None, "Failed to reload work order from disk"
        assert reloaded["current_status"] == "APPROVED", f"Expected reloaded status APPROVED, got {reloaded['current_status']}"
        print(f"[PASS] Test 10: Work-order reload and state preservation verified (Status: {reloaded['current_status']}).")

        # 11. CSV export
        export_path = export_work_orders_to_csv()
        assert export_path.exists(), f"Export CSV not found at {export_path}"
        df_exp = pd.read_csv(export_path)
        assert len(df_exp) >= 1, "Exported CSV must have at least 1 record"
        assert "work_order_id" in df_exp.columns
        print(f"[PASS] Test 11: Work-order CSV export verified ({len(df_exp)} rows).")

        # 12. Duplicate ID prevention
        wo2 = create_work_order_draft(
            prediction_result=sample_pred,
            machine_reference="CNC-Lathe-02",
            source="08 Live Telemetry"
        )
        assert wo2["work_order_id"] != wo["work_order_id"], "Subsequent work orders must have distinct IDs"
        print(f"[PASS] Test 12: Distinct ID incrementation verified ({wo2['work_order_id']}).")

        # 13. Human review requirement flag
        assert wo["human_review_required"] is True, "human_review_required must always be True"
        print("[PASS] Test 13: Human-in-the-loop requirement flag verified.")

        # 14. F1 event telemetry compatibility
        f1_telemetry_event = {
            "failure_probability": 0.7200,
            "predicted_class": 1,
            "risk_level": "High",
            "recommended_actions": ["Inspect workpiece feed rate and spindle mechanical loading."],
            "diagnostic_flags": ["Spindle Overload Event"],
            "summary": "MAINTENANCE ALERT: Machine failure risk."
        }
        wo_f1 = create_work_order_draft(
            prediction_result=f1_telemetry_event,
            machine_reference="CNC-Mill-01",
            source="08 Live Telemetry",
            simulation_or_data_source="SIMULATED"
        )
        assert wo_f1["risk_tier"] == "High"
        assert wo_f1["priority"] == "Priority Inspection"
        print("[PASS] Test 14: F1 live telemetry event compatibility verified.")

        # 15. Single prediction engine compatibility
        sample_input = {
            "Type": "M",
            "Air temperature [K]": 302.0,
            "Process temperature [K]": 310.5,
            "Rotational speed [rpm]": 1350.0,
            "Torque [Nm]": 62.0,
            "Tool wear [min]": 215.0
        }
        live_pred = predict_single(sample_input)
        wo_pred = create_work_order_draft(
            prediction_result=live_pred,
            machine_reference="CNC-M01",
            source="02 Machine Prediction"
        )
        assert wo_pred["prediction_probability"] == live_pred["failure_probability"]
        print("[PASS] Test 15: Prediction engine (predict_single) compatibility verified.")

        print("\n" + "=" * 70)
        print("ALL 15 MAINTENANCE WORK-ORDER TESTS PASSED (15/15)")
        print("=" * 70)

    finally:
        # Restore or clean up
        if backup_file and backup_file.exists():
            shutil.copyfile(backup_file, WORK_ORDERS_FILE)
            backup_file.unlink()


if __name__ == "__main__":
    run_work_orders_tests()
