"""Demo Guide: the 3-5 minute judging walkthrough, architecture and value proposition."""

import streamlit as st

from ui import nav, state
from ui.components import alert, hero

ARCH = """
digraph G {
  rankdir=TB; bgcolor="transparent"; nodesep=0.25; ranksep=0.32;
  node [shape=box, style="rounded,filled", fillcolor="#13223a", color="#3b82f6",
        fontcolor="#e2e8f0", fontname="Helvetica", fontsize=11];
  edge [color="#64748b"];
  data [label="TRANSACTION DATA\\n(synthetic · PaySim · IEEE-CIS · CSV)"];
  pre [label="DATA PREPROCESSING\\nclean · enrich · unseen categories"];
  tf [label="TRANSACTION FEATURES\\nreal-time rules (past-only)"];
  bf [label="BEHAVIOR FEATURES\\nper-account profile deviation"];
  an [label="ANOMALY DETECTION\\nIsolation Forest · LOF"];
  ml [label="FRAUD ML MODEL\\nRF / GB / LR, trained on other seeds"];
  gc [label="GRAPH CONSTRUCTION\\naccounts · devices · merchants · beneficiaries · locations"];
  fr [label="FRAUD RING DETECTOR\\n+ temporal windows + Fraud DNA"];
  re [label="RISK ENGINE", fillcolor="#3b1d1d", color="#ef4444"];
  t [label="Transaction Risk"]; b [label="Behavioral / Account Risk"]; n [label="Ring / Network Risk"];
  xai [label="EXPLAINABLE AI\\nchecklist · narrative · contributions"];
  act [label="ACTION RECOMMENDER\\nALLOW · MONITOR · STEP-UP · BLOCK"];
  ui [label="STREAMLIT DASHBOARD", fillcolor="#0f2d1f", color="#22c55e"];
  data -> pre; pre -> tf; pre -> bf; tf -> an; bf -> an; an -> ml; ml -> gc; gc -> fr; fr -> re;
  re -> t; re -> b; re -> n; t -> xai; b -> xai; n -> xai; xai -> act; act -> ui;
}
"""

STEPS = [
    ("executive", "Executive Dashboard", "Transactions, alerts, fraud rings and model metrics at a glance."),
    ("live", "Live Transaction Monitor", "Choose “⚡ FR-001 attack” and press ▶ Stream live: normal "
     "transactions flow, then the ring's transactions fire alerts."),
    ("transaction", "Transaction Investigation", "Open a red alert: three risk scores, the formula, "
     "the WHY checklist and connected entities."),
    ("account", "Account Intelligence", "Follow the account: profile, timeline, risk evolution, linked accounts."),
    ("network", "Network Graph", "See the devices, merchants, beneficiary and other accounts it connects to. Click nodes."),
    ("rings", "Fraud Ring Detection", "The coordinated cluster: ring risk, evidence breakdown, graph."),
    ("rings", "Temporal sequence", "On the ring page: who did what, when — the 50-minute coordinated window."),
    ("explain", "Explainable AI", "Exactly why: evidence weights, ML feature contributions, the formula."),
    ("alerts", "Recommended Action", "BLOCK + INVESTIGATE with the ring playbook (decision support)."),
    ("live", "False-positive control", "Scenario Lab → “✅ Legit ₹5,00,000”: LOW risk, because it matches "
     "the account's normal behaviour. Then “🚨 ₹48,000 account takeover”: CRITICAL."),
]


def render():
    hero("Demo Guide", "A 3–5 minute walkthrough that maps every requirement of HNX26PSI04 to the product")
    alert("<b>Traditional fraud detection asks:</b> “Is this transaction fraudulent?”<br>"
          "<b>We ask:</b> Is it unusual? Why? Who is it connected to? Is it part of a coordinated "
          "fraud ring? What evidence supports the decision? And what should the analyst do next?",
          "info")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown("### Demo flow")
        rings = state.rings()
        first = rings[0]["ring_id"] if rings else None
        for i, (page, title, text) in enumerate(STEPS, 1):
            a, b = st.columns([4, 1.2])
            a.markdown(f"**{i}. {title}** — {text}")
            with b:
                extra = {"sel_ring": first} if page == "rings" and first else {}
                if page in ("transaction", "account") and first:
                    r = state.ring(first)
                    m = [x for x in r["members"] if x not in r["mules"]][0]
                    ev = state.txn()
                    ev = ev[ev["txn_id"].isin(set(r["evidence_txns"])) & (ev["account_id"] == m)]
                    extra = {"sel_account": m}
                    if len(ev):
                        extra["sel_txn"] = ev.sort_values("final_risk").iloc[-1]["txn_id"]
                nav.button("Go →", page, f"guide_{i}", **extra)
    with c2:
        st.markdown("### Architecture")
        try:
            st.graphviz_chart(ARCH, width="stretch")
        except Exception:
            st.code("DATA → PREPROCESSING → TRANSACTION + BEHAVIOR FEATURES → ANOMALY → ML → "
                    "GRAPH → FRAUD RINGS → RISK ENGINE → XAI → ACTIONS → DASHBOARD")
    st.markdown("### What each requirement maps to")
    st.markdown("""
| Requirement | Where |
|---|---|
| Learn normal behaviour, behavioural profiling | `fraud/behavioral_analysis.py` · Behavioral Analysis, Account Intelligence |
| Transaction risk 0–100 with LOW/MEDIUM/HIGH/CRITICAL | `fraud/stream.py`, `fraud/risk_engine.py` · Transaction Investigation |
| Account risk, ring risk, final weighted risk (configurable) | `fraud/pipeline.py`, `fraud/fraud_ring.py` · sidebar weights |
| Heterogeneous graph, hidden relationships | `fraud/graph_analysis.py`, `fraud/network.py` · Network Graph |
| Fraud-ring detection with evidence | `fraud/fraud_ring.py` · Fraud Ring Detection |
| Temporal windows (5 min … 24 h) | `fraud/temporal_analysis.py` · Behavioral Analysis → Temporal |
| Fraud DNA / behavioural similarity | `fraud/fraud_dna.py` · Behavioral Analysis → Fraud DNA |
| Explainability (checklist, narrative, feature contributions) | `fraud/explainability.py`, `fraud/fraud_model.py` · Explainable AI |
| Action recommendation | `fraud/action_engine.py` · Alerts & Actions |
| False-positive protection | context-aware scoring · Scenario Lab · Model Performance look-alikes |
| Real-time simulation | `fraud/live.py` · Live Transaction Monitor |
| Precision / recall / F1 / FPR / ROC-AUC | `eval/evaluate.py` · Model Performance |
""")
