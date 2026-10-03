"""
Maintenance Work-Order Prototype Module for CNC Manufacturing Analytics (F5).
Provides structured work-order drafting, priority mapping, human-in-the-loop review,
lifecycle status transitions, local persistence, and CSV export capabilities.

RESEARCH & OPERATIONAL DISCLAIMER:
This is a prototype decision-support workflow and is NOT connected to a real enterprise
CMMS/EAM system, physical maintenance dispatch, or automated machine actuation infrastructure.
All generated work orders require certified human engineering review before execution.
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd

# Setup project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.recommendations import generate_maintenance_recommendation, determine_risk_level

WORK_ORDERS_DIR = PROJECT_ROOT / "outputs" / "work_orders"
WORK_ORDERS_FILE = WORK_ORDERS_DIR / "work_orders.json"

# Valid Prototype Statuses
VALID_STATUSES = [
    "DRAFT",
    "REVIEW_PENDING",
    "APPROVED",
    "SCHEDULED",
    "IN_PROGRESS",
    "COMPLETED",
    "CANCELLED"
]

# Prototype Priority Mapping based on validated risk tiers
RISK_TO_PRIORITY_MAP = {
    "Low": {
        "priority": "Routine Monitoring",
        "assigned_role": "Plant Reliability Technician",
        "due_window": "Next Scheduled PM Window (30 Days)",
        "urgency_level": "P4 - Normal"
    },
    "Moderate": {
        "priority": "Planned Review",
        "assigned_role": "Shift Maintenance Specialist",
        "due_window": "Within 48 Hours / Next Shift",
        "urgency_level": "P3 - Moderate"
    },
    "High": {
        "priority": "Priority Inspection",
        "assigned_role": "Mechanical Maintenance Lead",
        "due_window": "Within 12 Hours / Pre-Shift",
        "urgency_level": "P2 - High"
    },
    "Critical": {
        "priority": "Urgent Review",
        "assigned_role": "Senior Maintenance Engineer",
        "due_window": "Immediate Review (< 2 Hours)",
        "urgency_level": "P1 - Critical"
    }
}

# Strict Valid Status Transitions
ALLOWED_TRANSITIONS = {
    "DRAFT": ["REVIEW_PENDING", "CANCELLED"],
    "REVIEW_PENDING": ["APPROVED", "DRAFT", "CANCELLED"],
    "APPROVED": ["SCHEDULED", "CANCELLED"],
    "SCHEDULED": ["IN_PROGRESS", "CANCELLED"],
    "IN_PROGRESS": ["COMPLETED", "CANCELLED"],
    "COMPLETED": [],
    "CANCELLED": []
}


def get_priority_info(risk_tier: str) -> Dict[str, str]:
    """Returns deterministic prototype priority, role, and due window mapping for a given risk tier."""
    return RISK_TO_PRIORITY_MAP.get(risk_tier, RISK_TO_PRIORITY_MAP["Low"])


def generate_work_order_id(existing_count: int = 0) -> str:
    """Generates a structured prototype work order ID (e.g. WO-20261003-0001)."""
    date_str = datetime.now().strftime("%Y%m%d")
    return f"WO-{date_str}-{existing_count + 1:04d}"


def load_work_orders_store() -> List[Dict[str, Any]]:
    """Loads stored work orders from the local JSON store."""
    WORK_ORDERS_DIR.mkdir(parents=True, exist_ok=True)
    if not WORK_ORDERS_FILE.exists():
        return []
    try:
        with open(WORK_ORDERS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return []
    except Exception:
        return []


def save_work_orders_store(work_orders: List[Dict[str, Any]]) -> None:
    """Saves work orders to the local JSON store."""
    WORK_ORDERS_DIR.mkdir(parents=True, exist_ok=True)
    with open(WORK_ORDERS_FILE, "w", encoding="utf-8") as f:
        json.dump(work_orders, f, indent=2, ensure_ascii=False)


def create_work_order_draft(
    prediction_result: Dict[str, Any],
    machine_reference: str = "CNC-M01",
    source: str = "02 Machine Prediction",
    simulation_or_data_source: str = "BENCHMARK",
    notes: Optional[str] = None,
    created_by: str = "Human Operator (Decision Support)"
) -> Dict[str, Any]:
    """
    Creates and persists a new prototype maintenance work order draft from a prediction result.
    Reuses existing recommendation engine outputs from src/recommendations.py.
    """
    existing_orders = load_work_orders_store()
    wo_id = generate_work_order_id(len(existing_orders))

    # Extract recommendation info
    prob = float(prediction_result.get("failure_probability", 0.0))
    pred_class = int(prediction_result.get("predicted_class", 0))
    risk_tier = prediction_result.get("risk_level", determine_risk_level(prob))
    rec_actions = prediction_result.get("recommended_actions", ["Perform standard sensor check."])
    flags = prediction_result.get("diagnostic_flags", [])
    summary = prediction_result.get("summary", "Automated model recommendation.")
    
    priority_info = get_priority_info(risk_tier)

    now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    work_order = {
        "work_order_id": wo_id,
        "created_timestamp": now_iso,
        "source": str(source),
        "machine_reference": str(machine_reference),
        "simulation_or_data_source": str(simulation_or_data_source),
        "prediction_probability": round(prob, 4),
        "predicted_class": pred_class,
        "risk_tier": risk_tier,
        "priority": priority_info["priority"],
        "urgency_level": priority_info["urgency_level"],
        "assigned_role": priority_info["assigned_role"],
        "due_window": priority_info["due_window"],
        "recommended_action": rec_actions,
        "reason": "; ".join(flags) if flags else summary,
        "current_status": "DRAFT",
        "human_review_required": True,
        "notes": notes or "Initial prototype work-order draft generated from model prediction.",
        "created_by": str(created_by),
        "status_history": [
            {
                "from_status": None,
                "to_status": "DRAFT",
                "timestamp": now_iso,
                "updated_by": str(created_by),
                "comment": "Work order created in DRAFT state."
            }
        ],
        "disclaimer": "Prototype work order — requires certified human engineering review before action."
    }

    existing_orders.append(work_order)
    save_work_orders_store(existing_orders)
    return work_order


def update_work_order_status(
    work_order_id: str,
    new_status: str,
    updated_by: str = "Maintenance Supervisor",
    comment: Optional[str] = None
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Updates the lifecycle status of a work order following strict allowed transition rules.
    Returns: (success_bool, message, updated_work_order)
    """
    new_status = new_status.upper().strip()
    if new_status not in VALID_STATUSES:
        return False, f"Invalid status '{new_status}'. Valid statuses: {VALID_STATUSES}", None

    orders = load_work_orders_store()
    target_idx = None
    for idx, wo in enumerate(orders):
        if wo.get("work_order_id") == work_order_id:
            target_idx = idx
            break

    if target_idx is None:
        return False, f"Work order '{work_order_id}' not found.", None

    curr_wo = orders[target_idx]
    curr_status = curr_wo.get("current_status", "DRAFT")

    if curr_status == new_status:
        return True, f"Work order is already in '{new_status}' state.", curr_wo

    allowed_next = ALLOWED_TRANSITIONS.get(curr_status, [])
    if new_status not in allowed_next:
        return (
            False,
            f"Invalid transition from '{curr_status}' to '{new_status}'. Allowed transitions: {allowed_next}",
            curr_wo
        )

    now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    curr_wo["current_status"] = new_status
    if "status_history" not in curr_wo:
        curr_wo["status_history"] = []

    curr_wo["status_history"].append({
        "from_status": curr_status,
        "to_status": new_status,
        "timestamp": now_iso,
        "updated_by": str(updated_by),
        "comment": comment or f"Status transitioned to {new_status}."
    })

    orders[target_idx] = curr_wo
    save_work_orders_store(orders)
    return True, f"Successfully updated work order status to '{new_status}'.", curr_wo


def get_all_work_orders(sort_newest_first: bool = True) -> List[Dict[str, Any]]:
    """Retrieves all stored work orders with optional chronological sorting."""
    orders = load_work_orders_store()
    if sort_newest_first:
        orders = list(reversed(orders))
    return orders


def get_work_order_by_id(work_order_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves a single work order record by ID."""
    orders = load_work_orders_store()
    for wo in orders:
        if wo.get("work_order_id") == work_order_id:
            return wo
    return None


def export_work_orders_to_csv(output_path: Optional[Path] = None) -> Path:
    """Exports all stored work orders to CSV with required audit columns."""
    orders = load_work_orders_store()
    if output_path is None:
        WORK_ORDERS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = WORK_ORDERS_DIR / "work_orders_export.csv"

    records = []
    for wo in orders:
        actions = wo.get("recommended_action", [])
        if isinstance(actions, list):
            actions_str = " | ".join(actions)
        else:
            actions_str = str(actions)

        records.append({
            "work_order_id": wo.get("work_order_id"),
            "created_timestamp": wo.get("created_timestamp"),
            "machine_reference": wo.get("machine_reference"),
            "source": wo.get("source"),
            "prediction_probability": wo.get("prediction_probability"),
            "predicted_class": wo.get("predicted_class"),
            "risk_tier": wo.get("risk_tier"),
            "priority": wo.get("priority"),
            "recommended_action": actions_str,
            "status": wo.get("current_status"),
            "due_window": wo.get("due_window"),
            "assigned_role": wo.get("assigned_role"),
            "human_review_required": wo.get("human_review_required", True),
            "created_by": wo.get("created_by")
        })

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    return output_path


def seed_prototype_work_orders_if_empty() -> None:
    """Initializes a few representative prototype drafts if store is empty for instant demonstration."""
    existing = load_work_orders_store()
    if existing:
        return

    # Prototype Seed 1: High Risk Spindle Overload
    sample_res_high = {
        "failure_probability": 0.7420,
        "predicted_class": 1,
        "risk_level": "High",
        "recommended_actions": [
            "Inspect workpiece feed rate, depth of cut, and spindle mechanical loading.",
            "Schedule tool insert replacement / inspect cutter wear land."
        ],
        "diagnostic_flags": ["High Spindle Torque (65.2 Nm >= 60.0 Nm limit)", "Elevated Tool Wear (210.0 min >= 200 min)"],
        "summary": "MAINTENANCE ALERT: Machine is classified as at risk of failure."
    }
    create_work_order_draft(
        prediction_result=sample_res_high,
        machine_reference="CNC-Mill-01",
        source="08 Live Telemetry",
        simulation_or_data_source="SIMULATED",
        notes="Automated draft from elevated torque telemetry event during Spindle Overload scenario."
    )

    # Prototype Seed 2: Critical Thermal Dissipation
    sample_res_crit = {
        "failure_probability": 0.8950,
        "predicted_class": 1,
        "risk_level": "Critical",
        "recommended_actions": [
            "Inspect coolant flow rate, heat exchanger, and thermal dissipation paths.",
            "Halt non-essential cutting cycles and conduct priority maintenance inspection."
        ],
        "diagnostic_flags": ["Low Thermal Dissipation Gradient (Delta_T = 6.4 K < 8.6 K)"],
        "summary": "CRITICAL ALERT: Machine is classified as at immediate failure risk (>80% probability)."
    }
    wo_crit = create_work_order_draft(
        prediction_result=sample_res_crit,
        machine_reference="CNC-Lathe-04",
        source="02 Machine Prediction",
        simulation_or_data_source="BENCHMARK",
        notes="Operator manual prediction on elevated thermal stress input."
    )
    # Move to REVIEW_PENDING
    update_work_order_status(wo_crit["work_order_id"], "REVIEW_PENDING", comment="Shift lead queued for engineering review.")


if __name__ == "__main__":
    print("Testing Work Orders Module...")
    seed_prototype_work_orders_if_empty()
    orders = get_all_work_orders()
    print(f"Loaded {len(orders)} prototype work orders.")
    for o in orders:
        print(f" - {o['work_order_id']}: {o['machine_reference']} | Risk: {o['risk_tier']} | Status: {o['current_status']}")
