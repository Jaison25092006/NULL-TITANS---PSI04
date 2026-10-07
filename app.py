"""Fraud Intelligence — Explainable Real-Time Financial Fraud Intelligence Platform.
HackNex 2026 · HNX26PSI04.

Run:  streamlit run app.py
(results are precomputed in outputs/; if missing they are built on first start)
"""

import pandas as pd
import streamlit as st

from fraud.action_engine import DISCLAIMER
from fraud.config import MAX_UPLOAD_ROWS, RISK_WEIGHTS
from ui import nav
from ui.components import inject_css
from ui.pages import (account, alerts, behavior, executive, explain, guide, live, network,
                      performance, rings, transaction)

st.set_page_config(page_title="Fraud Intelligence", page_icon="🛡️", layout="wide",
                   initial_sidebar_state="expanded")
inject_css()

PAGES = {
    "executive": st.Page(executive.render, title="Executive Dashboard", icon="📊",
                         url_path="executive", default=True),
    "live": st.Page(live.render, title="Live Transaction Monitor", icon="📡", url_path="live"),
    "transaction": st.Page(transaction.render, title="Transaction Investigation", icon="🔎",
                           url_path="transaction"),
    "account": st.Page(account.render, title="Account Intelligence", icon="👤",
                       url_path="account"),
    "rings": st.Page(rings.render, title="Fraud Ring Detection", icon="🕸️", url_path="rings"),
    "network": st.Page(network.render, title="Network Graph", icon="🌐", url_path="network"),
    "behavior": st.Page(behavior.render, title="Behavioral Analysis", icon="🧬",
                        url_path="behavior"),
    "explain": st.Page(explain.render, title="Explainable AI", icon="💡", url_path="explain"),
    "alerts": st.Page(alerts.render, title="Alerts & Actions", icon="🚨", url_path="alerts"),
    "performance": st.Page(performance.render, title="Model Performance", icon="📈",
                           url_path="performance"),
    "guide": st.Page(guide.render, title="Demo Guide", icon="🎯", url_path="guide"),
}
nav.PAGES.update(PAGES)


def sidebar():
    with st.sidebar:
        st.markdown("## 🛡️ Fraud Intelligence")
        st.caption("HackNex 2026 · HNX26PSI04 · Real-Time Financial Fraud Intelligence")

        with st.expander("⚖️ Risk weights (configurable)", expanded=False):
            st.caption("Final = wT·Transaction + wB·Behavioral + wN·Network "
                       "(normalised to sum to 1), plus the critical-evidence guardrail.")
            cur = st.session_state.get("weights", dict(RISK_WEIGHTS))
            wt = st.slider("Transaction weight", 0.0, 1.0, float(cur["transaction"]), 0.05)
            wb = st.slider("Behavioral weight", 0.0, 1.0, float(cur["behavioral"]), 0.05)
            wn = st.slider("Network / Ring weight", 0.0, 1.0, float(cur["network"]), 0.05)
            total = (wt + wb + wn) or 1.0
            st.session_state["weights"] = {"transaction": round(wt / total, 4),
                                           "behavioral": round(wb / total, 4),
                                           "network": round(wn / total, 4)}
            w = st.session_state["weights"]
            st.code(f"Final = {w['transaction']:.2f}·T + {w['behavioral']:.2f}·B + "
                    f"{w['network']:.2f}·N", language=None)
            if st.button("Reset to default (0.45 / 0.25 / 0.30)"):
                st.session_state["weights"] = dict(RISK_WEIGHTS)
                st.rerun()

        with st.expander("📂 Data source", expanded=False):
            src = st.session_state.get("upload_name")
            st.caption(f"Active: **{src or 'Built-in synthetic dataset (41k transactions)'}**")
            up = st.file_uploader("Upload a CSV (PaySim / IEEE-CIS / any transactions file)",
                                  type=["csv"])
            if up is not None and st.button("Analyse uploaded file", type="primary"):
                analyse_upload(up)
            if src and st.button("Back to built-in dataset"):
                for k in ("upload_data", "upload_name", "upload_key"):
                    st.session_state.pop(k, None)
                st.rerun()
        st.caption(DISCLAIMER)


def analyse_upload(up):
    from fraud.data_loader import adapt_external
    from fraud.pipeline import run_frames
    from ui.state import prepare
    try:
        with st.spinner("Mapping columns and running every intelligence layer…"):
            df = pd.read_csv(up, nrows=MAX_UPLOAD_ROWS, low_memory=False)
            tx, accounts, merchants, labels = adapt_external(df)
            res = run_frames(tx, accounts, merchants)
            st.session_state["upload_data"] = prepare(res, labels, None, source=up.name)
            st.session_state["upload_name"] = f"{up.name} ({len(tx):,} rows)"
            st.session_state["upload_key"] = f"{up.name}-{len(tx)}-{pd.Timestamp.now().value}"
        st.success(f"Analysed {len(tx):,} transactions.")
        st.rerun()
    except Exception as exc:  # show, don't crash
        st.error(f"Could not analyse this file: {exc}")


sidebar()
st.navigation(list(PAGES.values()), position="sidebar").run()
