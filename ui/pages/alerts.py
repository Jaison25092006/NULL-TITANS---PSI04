"""Alerts & Recommended Actions: the analyst's work queue."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fraud.action_engine import ACCOUNT_ACTIONS, DISCLAIMER, TXN_ACTIONS
from fraud.risk_engine import BAND_COLORS
from ui import nav, state
from ui.components import STATUS_ICON, alert, badge, esc, hero, kpi_row, style_fig

DISPOSITIONS = ["Open", "Confirmed fraud", "False positive", "Escalated"]


def render():
    t, a, rings = state.txn(), state.acc(), state.rings()
    hero("Alerts & Recommended Actions",
         "Prioritised queue: fraud rings first, then accounts, then transactions — each with "
         "its recommended response")
    alerts_ = t[t["final_risk"] >= 61]
    kpi_row([("Ring alerts", len(rings), "coordinated fraud", "#ef4444" if rings else None),
             ("Account alerts", int((a["account_risk"] >= 61).sum()), "step-up / freeze", "#f97316"),
             ("Transaction alerts", len(alerts_), "HIGH + CRITICAL"),
             ("BLOCK + INVESTIGATE", int((t["band"] == "CRITICAL").sum()), "critical", "#ef4444"),
             ("STEP-UP", int((t["band"] == "HIGH").sum()), "OTP / biometric", "#f97316")])
    st.write("")
    c1, c2 = st.columns([1, 1.4])
    with c1:
        counts = t["action"].value_counts().reindex([v[0] for v in TXN_ACTIONS.values()]).fillna(0)
        fig = go.Figure(go.Bar(x=counts.values, y=counts.index, orientation="h",
                               marker_color=[BAND_COLORS[b] for b in TXN_ACTIONS],
                               text=[f"{int(v):,}" for v in counts.values], textposition="outside"))
        fig.update_xaxes(type="log", title="transactions (log)")
        st.plotly_chart(style_fig(fig, height=260, legend=False).update_layout(
            title="Recommended action per transaction"), width="stretch")
    with c2:
        st.markdown("**Action policy**")
        st.dataframe(pd.DataFrame([{"Risk": r, "Band": b, "Action": v[0], "What it means": v[1]}
                                   for (b, v), r in zip(TXN_ACTIONS.items(),
                                                        ["0–30", "31–60", "61–80", "81–100"])]),
                     hide_index=True, width="stretch")
        st.caption(DISCLAIMER)

    st.markdown("### 🕸️ Ring alerts")
    if not rings:
        st.info("No ring alerts.")
    for r in rings:
        with st.expander(f"{r['ring_id']} · ring risk {r['ring_risk']:.0f} · {r['size']} accounts · "
                         f"{r['recommended_action']}", expanded=r is rings[0]):
            st.markdown(f"{badge(r['band'])} {esc(r['pattern'])}", unsafe_allow_html=True)
            for i, s in enumerate(r["actions"], 1):
                st.markdown(f"**{i}. {s['step']}** — {esc(s['detail'])}")
            nav.button(f"Open {r['ring_id']} →", "rings", f"al_ring_{r['ring_id']}",
                       sel_ring=r["ring_id"])

    st.markdown("### 👤 Account alerts")
    acc_alerts = a[a["account_risk"] >= 61]
    if acc_alerts.empty:
        st.info("No account alerts.")
    else:
        st.dataframe(pd.DataFrame({
            "Account": acc_alerts["account_id"], "Risk": acc_alerts["account_risk"],
            "Band": acc_alerts["band"], "Action": acc_alerts["action"],
            "Ring": acc_alerts["ring_id"], "Flagged txns": acc_alerts["n_flagged"],
            "Why": acc_alerts["explanation"].str.slice(0, 140)}),
            hide_index=True, width="stretch", height=280,
            column_config={"Risk": st.column_config.ProgressColumn("Risk", min_value=0,
                                                                   max_value=100, format="%d")})

    st.markdown("### 💳 Transaction alert queue")
    f1, f2, f3 = st.columns(3)
    bands_ = f1.multiselect("Band", ["CRITICAL", "HIGH", "MEDIUM"], default=["CRITICAL", "HIGH"])
    ring_ids = ["(any)"] + [r["ring_id"] for r in rings]
    ring_f = f2.selectbox("Ring", ring_ids)
    sort = f3.selectbox("Sort by", ["Risk", "Most recent", "Amount"])
    q = t[t["band"].isin(bands_)]
    if ring_f != "(any)":
        q = q[q["ring_id"] == ring_f]
    q = q.sort_values({"Risk": "final_risk", "Most recent": "timestamp", "Amount": "amount_inr"}[sort],
                      ascending=False)
    disp = st.session_state.setdefault("dispositions", {})
    view = pd.DataFrame({
        "Transaction": q["txn_id"], "Time": q["timestamp"].dt.strftime("%d %b %H:%M"),
        "Account": q["account_id"], "Amount (₹)": q["amount_inr"].round(0),
        "Risk": q["final_risk"], "Status": q["status"].map(STATUS_ICON), "Action": q["action"],
        "Ring": q["ring_id"], "Disposition": [disp.get(x, "Open") for x in q["txn_id"]],
        "Why": q["explanation"].str.slice(0, 160)})
    edited = st.data_editor(view.head(300), hide_index=True, width="stretch", height=380,
                            disabled=[c for c in view.columns if c != "Disposition"],
                            column_config={"Disposition": st.column_config.SelectboxColumn(
                                options=DISPOSITIONS),
                                "Risk": st.column_config.ProgressColumn("Risk", min_value=0,
                                                                        max_value=100, format="%d")},
                            key="alert_editor")
    for tid, dv in zip(edited["Transaction"], edited["Disposition"]):
        if dv != "Open":
            disp[tid] = dv
    if disp:
        vc = pd.Series(list(disp.values())).value_counts()
        st.caption("Analyst dispositions this session: " +
                   ", ".join(f"{k} {v}" for k, v in vc.items()) +
                   " (kept in this browser session only; feed them back as labels to retrain).")
    c1, c2 = st.columns([1, 3])
    c1.download_button("⬇ Download alert queue (CSV)", view.to_csv(index=False).encode("utf-8"),
                       "alert_queue.csv", "text/csv")
    if len(q):
        with c2:
            pick = st.selectbox("Investigate", q["txn_id"].head(300), label_visibility="collapsed")
            nav.button("Open investigation →", "transaction", "al_tx", sel_txn=pick)
    st.markdown("### Account action policy")
    st.dataframe(pd.DataFrame([{"Band": b, "Action": v[0], "What it means": v[1]}
                               for b, v in ACCOUNT_ACTIONS.items()]), hide_index=True, width="stretch")
    alert("ℹ️ " + DISCLAIMER, "info")
