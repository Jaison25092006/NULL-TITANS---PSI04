"""Transaction Investigation: what happened, the three risk scores, why, who it is
connected to, and what to do."""

import plotly.graph_objects as go
import streamlit as st

from fraud.action_engine import DISCLAIMER, txn_rationale
from fraud.behavioral_analysis import DIMENSIONS
from fraud.explainability import checklist as build_checklist
from fraud.explainability import fp_protection, parse_reasons, waterfall
from fraud.fraud_model import contributions
from fraud.graph_analysis import neighbourhood, nid
from fraud.risk_engine import formula_text
from ui import nav, state
from ui.components import (alert, badge, checklist, chips, esc, hero, network_figure,
                           score_card, style_fig)

LAYER_COLOR = {"rule": "#60a5fa", "model": "#a78bfa", "anomaly": "#fbbf24", "network": "#ef4444"}


def render():
    t = state.txn()
    hero("Transaction Investigation",
         "Transaction → three risk scores → WHY it was flagged → connected entities → action")
    flagged = t[t["final_risk"] >= 61].sort_values(["final_risk", "timestamp"], ascending=False)
    c1, c2 = st.columns([3, 2])
    with c1:
        pool = list(flagged["txn_id"].head(400))
        cur = st.session_state.get("sel_txn")
        if cur and cur not in pool:
            pool = [cur] + pool
        if not pool:
            pool = list(t.sort_values("final_risk", ascending=False)["txn_id"].head(50))
        risk = t.set_index("txn_id")["final_risk"]
        tid = nav.pick(pool, "sel_txn", "Flagged transactions (highest risk first)",
                       fmt=lambda x: f"{x} · risk {risk.get(x, 0):.0f}")
    with c2:
        typed = st.text_input("…or look up any transaction ID", placeholder="e.g. T030393")
        if typed and typed.strip() in set(t["txn_id"]):
            tid = typed.strip()
            st.session_state["sel_txn"] = tid
        elif typed:
            st.warning("Transaction not found.")
    if tid is None:
        st.info("No transactions available.")
        return
    row = state.txn_row(tid)
    d = state.data()
    reasons = parse_reasons(row["reasons"])
    ring = state.ring(row["ring_id"]) if row["ring_id"] else None

    st.markdown(
        f"<div class='fi-card'><b style='font-size:1.15rem'>{row['txn_id']}</b> &nbsp; "
        f"{badge(row['band'])} {badge(row['action'])}<br>"
        f"<span class='fi-muted'>Account</span> <b>{row['account_id']}</b> · "
        f"<span class='fi-muted'>Amount</span> <b>₹{row['amount_inr']:,.2f}</b> · "
        f"<span class='fi-muted'>Time</span> {row['timestamp']:%d %b %Y %H:%M:%S} · "
        f"<span class='fi-muted'>{'Merchant' if row['merchant_id'] else 'Beneficiary'}</span> "
        f"<b>{esc(row['merchant_name'] or row['payee_account_id'])}</b> · "
        f"<span class='fi-muted'>Device</span> {esc(row['device_id'] or 'card-present')} · "
        f"<span class='fi-muted'>Location</span> {esc(row['city'])}, {row['country']} · "
        f"<span class='fi-muted'>Channel</span> {row['channel']}</div>", unsafe_allow_html=True)
    st.write("")
    c = st.columns(4)
    score_card(c[0], "Transaction risk", row["transaction_risk"], "rules · anomaly · ML")
    score_card(c[1], "Behavioral risk", row["behavioral_risk"], "vs this account's profile")
    score_card(c[2], "Network / ring risk", row["network_risk"],
               f"ring {row['ring_id']}" if row["ring_id"] else "graph evidence")
    score_card(c[3], "Final risk", row["final_risk"], row["action"])
    acc_row = state.account_row(row["account_id"])
    if acc_row is not None:
        st.caption(f"Account {row['account_id']} risk: **{acc_row['account_risk']:.0f}/100** "
                   f"({acc_row['band']})" + (f" · Ring {ring['ring_id']} risk: "
                                              f"**{ring['ring_risk']:.0f}/100**" if ring else ""))
    lines, _ = formula_text(row["transaction_risk"], row["behavioral_risk"], row["network_risk"],
                            row["network_evidence"], state.weights())
    st.code("\n".join(lines), language=None)

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("### WHY WAS THIS FLAGGED?")
        checklist(build_checklist(row, reasons, ring))
        note = fp_protection(row)
        if note:
            alert(f"🛡️ {esc(note)}", "ok")
    with right:
        st.markdown("### Explanation")
        st.markdown(f"<div class='fi-card'>{esc(row['explanation'])}</div>",
                    unsafe_allow_html=True)
        st.markdown("### Recommended action")
        kind = {"CRITICAL": "crit", "HIGH": "high", "MEDIUM": "info", "LOW": "ok"}[row["band"]]
        alert(f"<b>{row['action']}</b> — {txn_rationale(row['band'])}", kind)
        st.caption(DISCLAIMER)

    st.markdown("### Ranked evidence")
    wf = waterfall([r for r in reasons if r.get("layer") != "network"])
    net = [r for r in reasons if r.get("layer") == "network"]
    c1, c2 = st.columns([1.3, 1])
    with c1:
        if wf:
            fig = go.Figure(go.Waterfall(
                orientation="h", y=[w[0] for w in wf] + ["TRANSACTION RISK"],
                x=[w[3] for w in wf] + [0], measure=["relative"] * len(wf) + ["total"],
                text=[f"+{w[3]:.0f}" for w in wf] + [f"{row['transaction_risk']:.0f}"],
                connector=dict(line=dict(color="#334155")),
                increasing=dict(marker=dict(color="#ef4444")),
                totals=dict(marker=dict(color="#f97316")),
                hovertext=[w[4] for w in wf] + ["noisy-OR of all transaction-level evidence"],
                hoverinfo="text"))
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(style_fig(fig, height=80 + 34 * len(wf), legend=False).update_layout(
                title="How each reason raised the Transaction Risk (noisy-OR)"), width="stretch")
        else:
            st.success("No transaction-level warning signs.")
        for code, layer, w, inc, ev in wf:
            st.markdown(f"<span style='color:{LAYER_COLOR.get(layer, '#94a3b8')}'>●</span> "
                        f"**{code}** · weight {w:.2f} · +{inc:.0f} pts — {esc(ev)}",
                        unsafe_allow_html=True)
        for r in net:
            st.markdown(f"<span style='color:#ef4444'>●</span> **{r['code']}** (network) — "
                        f"{esc(r['evidence'])}", unsafe_allow_html=True)
    with c2:
        dims = list(DIMENSIONS)
        vals = [float(row.get(f"b_{k}", 0)) for k in dims]
        fig = go.Figure(go.Bar(x=vals, y=[DIMENSIONS[k][1] for k in dims], orientation="h",
                               marker_color=["#ef4444" if v >= 0.5 else "#f97316" if v > 0 else
                                             "#334155" for v in vals],
                               text=[f"{v:.0%}" for v in vals], textposition="outside"))
        fig.update_xaxes(range=[0, 1.15], tickformat=".0%")
        st.plotly_chart(style_fig(fig, height=300, legend=False).update_layout(
            title=f"Behavioral deviation by dimension (score {row['behavioral_risk']:.0f})"),
            width="stretch")
        contrib = contributions(d["model"], row)
        if contrib:
            top = [c for c in contrib if abs(c[3]) > 0.002][:8]
            if top:
                fig = go.Figure(go.Bar(
                    x=[c[3] * 100 for c in top][::-1], y=[c[1] for c in top][::-1],
                    orientation="h",
                    marker_color=["#ef4444" if c[3] > 0 else "#22c55e" for c in top][::-1],
                    hovertext=[f"value {c[2]:.2f}" for c in top][::-1], hoverinfo="text+x"))
                fig.update_xaxes(title="contribution to fraud probability (pts)")
                st.plotly_chart(style_fig(fig, height=300, legend=False).update_layout(
                    title=f"ML model drivers · p(fraud) = {row['ml_probability']:.0%}"),
                    width="stretch")

    st.markdown("### Connected entities")
    G, _ = state.graphs()
    c1, c2 = st.columns([1, 1.6])
    with c1:
        same_dev = t[(t["device_id"] == row["device_id"]) & (row["device_id"] != "")]["account_id"].nunique()
        st.markdown(f"**Account** {chips([row['account_id']])}", unsafe_allow_html=True)
        if row["device_id"]:
            st.markdown(f"**Device** {chips([row['device_id']])} used by {same_dev} account(s)",
                        unsafe_allow_html=True)
        if row["merchant_id"]:
            st.markdown(f"**Merchant** {chips([row['merchant_name']])} ({row['merchant_category']})",
                        unsafe_allow_html=True)
        if row["payee_account_id"]:
            senders = t[t["payee_account_id"] == row["payee_account_id"]]["account_id"].nunique()
            is_mule = any(row["payee_account_id"] in r["mules"] for r in state.rings())
            st.markdown(f"**Beneficiary** {chips([row['payee_account_id']])} receives from "
                        f"{senders} account(s){' · ⚠️ mule account' if is_mule else ''}",
                        unsafe_allow_html=True)
        st.markdown(f"**Location** {chips([row['city'] + ', ' + row['country']])}",
                    unsafe_allow_html=True)
        if ring:
            st.markdown(f"**Fraud ring** {chips([ring['ring_id']], '#ef4444')} {ring['size']} "
                        f"accounts · risk {ring['ring_risk']:.0f}", unsafe_allow_html=True)
        if acc_row is not None and acc_row["connected_accounts"]:
            st.markdown("**Linked suspicious accounts** " +
                        chips(acc_row["connected_accounts"].split(";")[:10]),
                        unsafe_allow_html=True)
        st.write("")
        nav.button(f"Open account {row['account_id']} →", "account", "tx_acc",
                   sel_account=row["account_id"])
        if ring:
            nav.button(f"Open fraud ring {ring['ring_id']} →", "rings", "tx_ring",
                       sel_ring=ring["ring_id"])
        nav.button("Show in network graph →", "network", "tx_net",
                   sel_node=nid("account", row["account_id"]))
    with c2:
        H = neighbourhood(G, nid("account", row["account_id"]), radius=2, max_nodes=70,
                          types=("account", "device", "beneficiary", "merchant", "location"))
        focus = [nid("account", row["account_id"])]
        for k, v in (("device", row["device_id"]), ("merchant", row["merchant_id"]),
                     ("beneficiary", row["payee_account_id"])):
            if v:
                focus.append(nid(k, v))
        fig, _ = network_figure(H, height=430, highlight=focus)
        st.plotly_chart(fig, width="stretch", key="tx_graph")
