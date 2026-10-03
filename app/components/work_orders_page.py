"""
Page 11: Maintenance Work Orders Component for Streamlit Predictive Maintenance Dashboard (F5).
Lightweight Kanban-style work-order workspace inspired by modern SaaS project dashboards.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import streamlit as st

# Setup paths
APP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.work_orders import (
    get_all_work_orders,
    get_work_order_by_id,
    create_work_order_draft,
    update_work_order_status,
    export_work_orders_to_csv,
    seed_prototype_work_orders_if_empty,
    VALID_STATUSES,
    ALLOWED_TRANSITIONS
)


def render_work_orders_page():
    """Renders the dedicated 11 Maintenance Work Orders dashboard page with Kanban board."""
    seed_prototype_work_orders_if_empty()

    # 1. Page Header & Research Notice
    st.markdown("""
    <div class="content-card">
        <div style="font-size: 1.25rem; font-weight: 700; color: #1F1F1F; margin-bottom: 4px;">
            📋 Maintenance Work-Order Management
        </div>
        <div style="font-size: 0.85rem; color: #6B6B63; line-height: 1.5;">
            Track actionable maintenance work-order drafts generated from machine health predictions and diagnostic alerts.
            Enforces a <strong>Human-in-the-Loop (HITL)</strong> review lifecycle before any maintenance scheduling.
        </div>
        <div class="disclaimer-box" style="margin-top: 12px; margin-bottom: 0px; font-size: 0.78rem;">
            <strong>⚠️ Prototype Notice:</strong> Decision-support workflow only. Not connected to live factory CMMS/EAM systems. All work orders require certified human engineering review.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Load orders
    all_orders = get_all_work_orders(sort_newest_first=True)

    # 2. Status Summary Counters (KPI Strip)
    counts = {s: 0 for s in VALID_STATUSES}
    for wo in all_orders:
        s = wo.get("current_status", "DRAFT")
        if s in counts:
            counts[s] += 1

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">DRAFTS</div>
            <div class="kpi-value" style="font-size: 1.6rem;">{counts['DRAFT']}</div>
            <div class="kpi-subtitle">Unsubmitted</div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">REVIEW PENDING</div>
            <div class="kpi-value kpi-watch" style="font-size: 1.6rem;">{counts['REVIEW_PENDING']}</div>
            <div class="kpi-subtitle">Awaiting Lead Review</div>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">APPROVED</div>
            <div class="kpi-value kpi-healthy" style="font-size: 1.6rem;">{counts['APPROVED']}</div>
            <div class="kpi-subtitle">Ready to Schedule</div>
        </div>
        """, unsafe_allow_html=True)

    with k4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">SCHEDULED</div>
            <div class="kpi-value kpi-analytical" style="font-size: 1.6rem;">{counts['SCHEDULED']}</div>
            <div class="kpi-subtitle">Assigned to Shift</div>
        </div>
        """, unsafe_allow_html=True)

    with k5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">IN PROGRESS</div>
            <div class="kpi-value kpi-warning" style="font-size: 1.6rem;">{counts['IN_PROGRESS']}</div>
            <div class="kpi-subtitle">Active Inspection</div>
        </div>
        """, unsafe_allow_html=True)

    with k6:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">COMPLETED</div>
            <div class="kpi-value kpi-healthy" style="font-size: 1.6rem;">{counts['COMPLETED']}</div>
            <div class="kpi-subtitle">Verified & Closed</div>
        </div>
        """, unsafe_allow_html=True)

    # 3. KANBAN-STYLE WORK ORDER BOARD (Inspired by Reference UI)
    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size: 1.05rem; font-weight: 700; color: #1F1F1F; margin-bottom: 12px;">
        📌 Work-Order Lifecycle Board
    </div>
    """, unsafe_allow_html=True)

    kanban_statuses = ["REVIEW_PENDING", "APPROVED", "SCHEDULED", "IN_PROGRESS", "COMPLETED"]
    kanban_labels = ["Review", "Approved", "Scheduled", "In Progress", "Completed"]

    kb_cols = st.columns(5)
    for col, st_key, st_label in zip(kb_cols, kanban_statuses, kanban_labels):
        col_orders = [o for o in all_orders if o.get("current_status") == st_key]
        with col:
            st.markdown(f"""
            <div class="kanban-col-header">
                <span>{st_label}</span>
                <span class="badge-neutral">{len(col_orders)}</span>
            </div>
            """, unsafe_allow_html=True)

            if not col_orders:
                st.markdown("""
                <div style="background: #FFFFFF; border: 1px dashed #E4E4DD; border-radius: 10px; padding: 20px 10px; text-align: center; color: #8E8E85; font-size: 0.74rem;">
                    No orders
                </div>
                """, unsafe_allow_html=True)
            else:
                for wo in col_orders[:4]:  # Show top 4 per column
                    risk_t = wo.get("risk_tier", "Low")
                    prob = wo.get("prediction_probability", 0.0)
                    actions = wo.get("recommended_action", [])
                    action_summary = actions[0] if actions else "Routine check"
                    
                    # Highlight critical / urgent orders with high contrast card
                    is_priority = (risk_t in ["High", "Critical"] and st_key in ["REVIEW_PENDING", "IN_PROGRESS"])
                    card_cls = "kanban-card kanban-card-priority" if is_priority else "kanban-card"
                    
                    badge_cls = "badge-risk-crit" if risk_t == "Critical" else ("badge-risk-high" if risk_t == "High" else ("badge-risk-mod" if risk_t == "Moderate" else "badge-risk-low"))

                    st.markdown(f"""
                    <div class="{card_cls}">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span class="kanban-card-id">{wo.get('work_order_id')}</span>
                            <span class="{badge_cls}">{risk_t}</span>
                        </div>
                        <div style="font-weight: 600; font-size: 0.8rem; margin-top: 6px;">{wo.get('machine_reference')}</div>
                        <div class="kanban-card-text">{action_summary}</div>
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 8px; font-size: 0.7rem; color: {'#E4E4DD' if is_priority else '#8E8E85'};">
                            <span>Prob: {prob*100:.1f}%</span>
                            <span>{wo.get('due_window', '48 Hours')}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    # 4. Filters, Search & Registry Table
    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
    f_col1, f_col2, f_col3, f_col4 = st.columns([1.5, 1.5, 2, 1.5])
    with f_col1:
        status_filter = st.selectbox("Filter Status", ["All Statuses"] + VALID_STATUSES, index=0)
    with f_col2:
        risk_filter = st.selectbox("Filter Risk Tier", ["All Risks", "Low", "Moderate", "High", "Critical"], index=0)
    with f_col3:
        search_query = st.text_input("Search ID / Machine / Reason", placeholder="e.g. WO-20261003 or CNC-Mill")
    with f_col4:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        export_csv_path = export_work_orders_to_csv()
        with open(export_csv_path, "r", encoding="utf-8") as f:
            csv_data = f.read()
        st.download_button(
            label="📥 Export Orders CSV",
            data=csv_data,
            file_name="work_orders_export.csv",
            mime="text/csv",
            use_container_width=True
        )

    # Apply filters
    filtered_orders = all_orders
    if status_filter != "All Statuses":
        filtered_orders = [o for o in filtered_orders if o.get("current_status") == status_filter]
    if risk_filter != "All Risks":
        filtered_orders = [o for o in filtered_orders if o.get("risk_tier") == risk_filter]
    if search_query.strip():
        q = search_query.strip().lower()
        filtered_orders = [
            o for o in filtered_orders
            if q in o.get("work_order_id", "").lower()
            or q in o.get("machine_reference", "").lower()
            or q in o.get("reason", "").lower()
            or q in o.get("source", "").lower()
        ]

    # 5. Work Order Detail & Transition Form
    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
    col_left, col_right = st.columns([1.3, 1.0])

    with col_left:
        st.markdown(f"""
        <div style="font-size: 1rem; font-weight: 700; color: #1F1F1F; margin-bottom: 8px;">
            📑 Work-Order Registry ({len(filtered_orders)} Records)
        </div>
        """, unsafe_allow_html=True)

        if not filtered_orders:
            st.info("No work orders match the selected filters.")
        else:
            table_rows = []
            for wo in filtered_orders:
                prob = wo.get("prediction_probability", 0.0)
                table_rows.append({
                    "WO ID": wo.get("work_order_id"),
                    "Created": wo.get("created_timestamp"),
                    "Machine": wo.get("machine_reference"),
                    "Risk": wo.get("risk_tier"),
                    "Prob": f"{prob * 100:.1f}%",
                    "Priority": wo.get("priority"),
                    "Status": wo.get("current_status"),
                    "Source": wo.get("source")
                })
            
            df_table = pd.DataFrame(table_rows)
            st.dataframe(df_table, use_container_width=True, height=320)

    with col_right:
        st.markdown("""
        <div style="font-size: 1rem; font-weight: 700; color: #1F1F1F; margin-bottom: 8px;">
            🔍 Work-Order Detail & Lifecycle Transition
        </div>
        """, unsafe_allow_html=True)

        if not all_orders:
            st.info("No work orders currently available.")
        else:
            order_ids = [wo["work_order_id"] for wo in all_orders]
            selected_wo_id = st.selectbox("Select Work Order to Inspect", order_ids, index=0)
            selected_wo = get_work_order_by_id(selected_wo_id)

            if selected_wo:
                curr_st = selected_wo.get("current_status", "DRAFT")
                allowed_next = ALLOWED_TRANSITIONS.get(curr_st, [])
                risk_t = selected_wo.get("risk_tier", "Low")
                risk_color = "#8B3F3F" if risk_t in ["High", "Critical"] else ("#9A7A32" if risk_t == "Moderate" else "#477A55")

                st.markdown(f"""
                <div class="content-card" style="font-size: 0.84rem; line-height: 1.6; padding: 18px 20px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #ECECE6; padding-bottom: 8px; margin-bottom: 8px;">
                        <span style="font-size: 1.05rem; font-weight: 700; color: #1F1F1F; font-family: 'JetBrains Mono', monospace;">{selected_wo['work_order_id']}</span>
                        <span class="badge-neutral">{curr_st}</span>
                    </div>
                    <div><strong>Machine:</strong> {selected_wo.get('machine_reference')} ({selected_wo.get('simulation_or_data_source')})</div>
                    <div><strong>Source:</strong> {selected_wo.get('source')}</div>
                    <div><strong>Failure Probability:</strong> {selected_wo.get('prediction_probability')*100:.2f}% | <span style="color: {risk_color}; font-weight: 700;">{risk_t} Risk</span></div>
                    <div><strong>Priority:</strong> {selected_wo.get('priority')} ({selected_wo.get('urgency_level')})</div>
                    <div><strong>Assigned Role:</strong> {selected_wo.get('assigned_role')}</div>
                    <div><strong>Due Window:</strong> {selected_wo.get('due_window')}</div>
                    <div style="margin-top: 6px;"><strong>Diagnostic Reason:</strong> <span style="color: #6B6B63;">{selected_wo.get('reason')}</span></div>
                    <div style="margin-top: 6px;"><strong>Recommended Actions:</strong></div>
                    <ul style="margin: 4px 0 8px 16px; padding: 0; color: #6B6B63;">
                        {"".join([f"<li>{act}</li>" for act in selected_wo.get('recommended_action', [])])}
                    </ul>
                </div>
                """, unsafe_allow_html=True)

                if allowed_next:
                    st.markdown("<div style='font-size: 0.84rem; font-weight: 600; color: #1F1F1F; margin-top: 10px;'>Advance Lifecycle Status:</div>", unsafe_allow_html=True)
                    t_col1, t_col2 = st.columns([1.5, 1.0])
                    with t_col1:
                        target_status = st.selectbox("Next Status", allowed_next, key=f"target_st_{selected_wo_id}")
                    with t_col2:
                        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
                        if st.button("Apply Status", use_container_width=True, key=f"btn_st_{selected_wo_id}"):
                            success, msg, _ = update_work_order_status(
                                work_order_id=selected_wo_id,
                                new_status=target_status,
                                updated_by="Shift Engineer (HITL Review)",
                                comment=f"Advanced to {target_status} via dashboard review."
                            )
                            if success:
                                st.success(msg)
                                st.rerun()
                            else:
                                st.error(msg)
                else:
                    st.info(f"Work order is in terminal state '{curr_st}'. No further transitions permitted.")

    # 6. Manual Draft Creation Accordion
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    with st.expander("➕ Draft New Maintenance Work Order (Manual Entry)", expanded=False):
        st.markdown("""
        <div style="font-size: 0.84rem; color: #6B6B63; margin-bottom: 10px;">
            Create a prototype maintenance work order draft manually with customized machine telemetry context.
        </div>
        """, unsafe_allow_html=True)

        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            m_mach = st.text_input("Machine Reference ID", value="CNC-Mill-08")
            m_prob = st.slider("Observed / Simulated Failure Risk", min_value=0.0, max_value=1.0, value=0.65, step=0.01)
        with m_col2:
            m_source = st.selectbox("Telemetry Source Page", ["02 Machine Prediction", "08 Live Telemetry", "03 Batch Telemetry", "Plant Floor Alert"], index=0)
            m_tier = "Critical" if m_prob >= 0.80 else ("High" if m_prob >= 0.50 else ("Moderate" if m_prob >= 0.20 else "Low"))
            st.markdown(f"<div style='margin-top: 28px;'>Calculated Risk Tier: <strong>{m_tier}</strong></div>", unsafe_allow_html=True)
        with m_col3:
            m_notes = st.text_area("Audit Notes / Operator Comment", value="Operator requested priority spindle inspection due to vibration.")

        if st.button("Generate Prototype Work Order Draft", type="primary"):
            custom_pred = {
                "failure_probability": m_prob,
                "predicted_class": int(m_prob >= 0.50),
                "risk_level": m_tier,
                "recommended_actions": [
                    "Perform comprehensive spindle vibration and bearing temperature diagnostic.",
                    "Inspect cutting tool wear land and replace inserts if required."
                ],
                "diagnostic_flags": [f"Operator Flagged Risk ({m_prob*100:.1f}%)"],
                "summary": f"Manual work-order draft created for {m_mach}."
            }
            new_wo = create_work_order_draft(
                prediction_result=custom_pred,
                machine_reference=m_mach,
                source=m_source,
                simulation_or_data_source="MANUAL_INPUT",
                notes=m_notes,
                created_by="Shift Maintenance Engineer"
            )
            st.success(f"Work Order {new_wo['work_order_id']} created successfully!")
            st.rerun()
