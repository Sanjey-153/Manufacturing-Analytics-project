"""
Page 13: Industrial Validation Readiness Component for Streamlit Predictive Maintenance Dashboard (F7).
Redesigned with clean light SaaS aesthetics and muted status badges.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import streamlit as st

# Setup paths
APP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.industrial_validation import (
    get_industrial_readiness_status,
    validate_industrial_schema,
    evaluate_industrial_dataset,
    generate_industrial_validation_report,
    create_mock_industrial_fixture,
    READINESS_CHECKLIST_ITEMS
)


def render_industrial_validation_page(model: Optional[Any] = None, preprocessor: Optional[Any] = None):
    """Renders the dedicated 13 Industrial Validation dashboard page."""
    status_info = get_industrial_readiness_status()

    # 1. PAGE HEADER & MANDATORY STATUS NOTICE
    st.markdown("""
    <div class="content-card">
        <div style="font-size: 1.25rem; font-weight: 700; color: #1F1F1F; margin-bottom: 4px;">
            🏭 Industrial Validation Readiness
        </div>
        <div style="font-size: 0.85rem; color: #6B6B63; line-height: 1.5;">
            Establishes rigorous validation gates, schema translation, engineering unit verification, and ground-truth alignment for external factory telemetry.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Prominent Primary Status Banner
    st.markdown("""
    <div style="background: #FAF5E8; border: 1px solid #E6D8AD; padding: 16px 20px; border-radius: 12px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center;">
        <div>
            <div style="font-size: 0.72rem; font-weight: 700; color: #9A7A32; letter-spacing: 0.05em; text-transform: uppercase;">CURRENT VALIDATION STATUS</div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #1F1F1F; margin-top: 2px;">
                ⏸️ AWAITING VERIFIED INDUSTRIAL TELEMETRY
            </div>
            <div style="font-size: 0.8rem; color: #6B6B63; margin-top: 4px;">
                Active Data Source: <strong>BENCHMARK (AI4I 2020 CNC Milling Dataset)</strong>. Empirical validation gate remains locked until verified SCADA/MES telemetry is ingested.
            </div>
        </div>
        <div style="text-align: right;">
            <span class="badge-risk-high">
                GATE LOCKED
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. READINESS CHECKLIST (12-POINT AUDIT)
    st.markdown("""
    <div style="font-size: 1rem; font-weight: 700; color: #1F1F1F; margin-bottom: 10px;">
        📋 12-Point Industrial Deployment Readiness Checklist
    </div>
    """, unsafe_allow_html=True)

    chk_col1, chk_col2 = st.columns(2)

    half = len(READINESS_CHECKLIST_ITEMS) // 2
    items_left = READINESS_CHECKLIST_ITEMS[:half]
    items_right = READINESS_CHECKLIST_ITEMS[half:]

    with chk_col1:
        for it in items_left:
            st.markdown(f"""
            <div style="background: #FFFFFF; border: 1px solid #E4E4DD; padding: 12px 16px; border-radius: 10px; margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 600; color: #1F1F1F; font-size: 0.84rem;">[ ] {it['title']}</span>
                    <span class="badge-neutral" style="font-size: 0.7rem;">PENDING DATA</span>
                </div>
                <div style="font-size: 0.76rem; color: #6B6B63; margin-top: 3px;">{it['description']}</div>
            </div>
            """, unsafe_allow_html=True)

    with chk_col2:
        for it in items_right:
            st.markdown(f"""
            <div style="background: #FFFFFF; border: 1px solid #E4E4DD; padding: 12px 16px; border-radius: 10px; margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 600; color: #1F1F1F; font-size: 0.84rem;">[ ] {it['title']}</span>
                    <span class="badge-neutral" style="font-size: 0.7rem;">PENDING DATA</span>
                </div>
                <div style="font-size: 0.76rem; color: #6B6B63; margin-top: 3px;">{it['description']}</div>
            </div>
            """, unsafe_allow_html=True)

    # 3. INDUSTRIAL INGESTION GATEWAY & VALIDATION RUNNER
    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size: 1rem; font-weight: 700; color: #1F1F1F; margin-bottom: 10px;">
        🚪 External Telemetry Validation Gateway
    </div>
    """, unsafe_allow_html=True)

    c_g1, c_g2 = st.columns([1.5, 1.0])

    with c_g1:
        st.markdown("""
        <div class="content-card" style="padding: 18px 20px;">
            <div style="font-size: 0.92rem; font-weight: 700; color: #1F1F1F; margin-bottom: 4px;">
                Upload External CNC Telemetry File (CSV)
            </div>
            <div style="font-size: 0.8rem; color: #6B6B63; margin-bottom: 12px;">
                Upload SCADA / MES telemetry exports. Required: Machine ID, Timestamps, 5 Sensor Telemetries, and Failure Outcome Labels.
            </div>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Upload Industrial CSV", type=["csv"], help="Upload external SCADA telemetry CSV for validation.")
        use_mock = st.checkbox("🧪 Load Synthetic Unit-Test Fixture for Schema Verification Demo", value=False)

    with c_g2:
        st.markdown("""
        <div class="content-card" style="height: 100%; padding: 18px 20px;">
            <div style="font-size: 0.92rem; font-weight: 700; color: #1F1F1F; margin-bottom: 6px;">
                Validation Policy & Standards
            </div>
            <div style="font-size: 0.8rem; color: #6B6B63; line-height: 1.5;">
                <p><strong>Zero Fabrication Rule:</strong> Metrics are only calculated if ground-truth failure labels are provided in the upload.</p>
                <p><strong>Baseline Preservation:</strong> The authoritative LightGBM model is evaluated strictly without retraining or weight alteration.</p>
                <p><strong>Label Alignment:</strong> Failure labels must represent documented breakdown events rather than planned maintenance stops.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 4. EXECUTION & RESULTS
    target_df = None
    source_tag = "INDUSTRIAL"

    if uploaded_file is not None:
        try:
            target_df = pd.read_csv(uploaded_file)
            st.info(f"Loaded external telemetry dataset with {len(target_df)} records.")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
    elif use_mock:
        target_df = create_mock_industrial_fixture(100)
        source_tag = "INDUSTRIAL"
        st.warning("Using synthetic test fixture (TEST FIXTURE ONLY - Not real industrial data).")

    if target_df is not None:
        st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
        schema_report = validate_industrial_schema(target_df, source_tag=source_tag)

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown(f"""
            <div class="content-card" style="padding: 18px 20px;">
                <div style="font-size: 0.92rem; font-weight: 700; color: #1F1F1F; margin-bottom: 6px;">
                    Schema & Unit Validation Report
                </div>
                <div style="font-size: 0.84rem;">Status: <strong style="color: {'#477A55' if schema_report['is_valid'] else '#8B3F3F'};">{'VALID' if schema_report['is_valid'] else 'INVALID'}</strong></div>
                <div style="font-size: 0.84rem;">Mapped Columns: <strong>{len(schema_report['mapped_columns'])}</strong></div>
                <div style="font-size: 0.84rem;">Ground Truth Labels: <strong>{'PRESENT' if schema_report['has_ground_truth_labels'] else 'MISSING'}</strong></div>
            </div>
            """, unsafe_allow_html=True)

        with col_s2:
            st.markdown("""
            <div class="content-card" style="padding: 18px 20px;">
                <div style="font-size: 0.92rem; font-weight: 700; color: #1F1F1F; margin-bottom: 6px;">
                    Validation Evaluation Action
                </div>
            """, unsafe_allow_html=True)
            if schema_report["is_valid"] and schema_report["has_ground_truth_labels"]:
                if st.button("🚀 Run Empirical Industrial Validation", type="primary"):
                    eval_res = evaluate_industrial_dataset(
                        target_df,
                        source_tag=source_tag,
                        model=model,
                        preprocessor=preprocessor
                    )
                    st.session_state["ind_eval_results"] = eval_res
                    j_p, c_p = generate_industrial_validation_report(eval_res)
                    st.success(f"Validation completed. Reports generated at {j_p.name} and {c_p.name}.")
            else:
                st.warning("Validation Gate Locked: Uploaded data must contain all 5 sensor fields and verified failure_label.")
            st.markdown("</div>", unsafe_allow_html=True)

        # Show evaluation metrics if available
        if "ind_eval_results" in st.session_state and st.session_state["ind_eval_results"].get("metrics"):
            res = st.session_state["ind_eval_results"]
            m = res["metrics"]
            st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-size: 1rem; font-weight: 700; color: #1F1F1F; margin-bottom: 10px;">
                📊 External Benchmark Performance Metrics (N={res['total_records']})
            </div>
            """, unsafe_allow_html=True)

            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.metric("Accuracy", f"{m['Accuracy']*100:.2f}%")
            with m2:
                st.metric("Recall", f"{m['Recall']*100:.2f}%")
            with m3:
                st.metric("Precision", f"{m['Precision']*100:.2f}%")
            with m4:
                st.metric("F1-Score", f"{m['F1']*100:.2f}%")
            with m5:
                st.metric("Alert Rate", f"{m['Alert_Rate']*100:.2f}%")

    # 5. RESEARCH INTEGRITY FOOTER
    st.markdown("""
    <div class="disclaimer-box" style="margin-top: 24px; font-size: 0.78rem;">
        <strong>Academic & Industrial Disclaimer:</strong> No claim of real industrial model performance is made because verified industrial telemetry and longitudinal ground-truth outcomes are not currently available. The system is architected for seamless external benchmarking when industrial partners provide verified CNC sensory streams.
    </div>
    """, unsafe_allow_html=True)
