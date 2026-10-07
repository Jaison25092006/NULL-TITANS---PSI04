"""Explainable AI: global drivers of the system and local explanations of any decision."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fraud.explainability import checklist as build_checklist
from fraud.explainability import parse_reasons, waterfall
from fraud.fraud_model import FEATURES, HAS_SHAP, contributions
from fraud.risk_engine import formula_text, normalise
from ui import nav, state
from ui.components import checklist, esc, hero, style_fig


@st.cache_data(show_spinner=False)
def _reason_stats(key, wkey):
    t = state.txn()
    f = t[t["final_risk"] >= 61]
    rows = []
    for rs in f["reasons"]:
        for r in parse_reasons(rs):
            rows.append((r["code"], r.get("layer", "rule")))
    return pd.DataFrame(rows, columns=["code", "layer"])


def render():
    hero("Explainable AI",
         "Never “Risk = 92” without the why: global drivers, per-decision evidence, model "
         "feature contributions and the exact formula")
    d, t = state.data(), state.txn()
    bundle = d["model"]
    tabs = st.tabs(["🌍 Global explanations", "🔍 Explain one decision"])
    with tabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            imp = (bundle or {}).get("importance") or state.model_card().get("importance", [])
            if imp:
                top = imp[:12][::-1]
                fig = go.Figure(go.Bar(x=[v for _, v in top], y=[FEATURES.get(k, k) for k, _ in top],
                                       orientation="h", marker_color="#a78bfa"))
                fig.update_xaxes(title="drop in average precision when shuffled")
                st.plotly_chart(style_fig(fig, height=420, legend=False).update_layout(
                    title=f"ML model ({(bundle or {}).get('name', 'model')}) — permutation "
                          f"feature importance"), width="stretch")
        with c2:
            rs = _reason_stats(st.session_state.get("upload_key", "default"),
                               tuple(sorted(state.weights().items())))
            if len(rs):
                vc = rs["code"].value_counts().head(14)[::-1]
                colors = {"rule": "#60a5fa", "model": "#a78bfa", "anomaly": "#fbbf24",
                          "network": "#ef4444"}
                layer = rs.drop_duplicates("code").set_index("code")["layer"]
                fig = go.Figure(go.Bar(x=vc.values, y=vc.index, orientation="h",
                                       marker_color=[colors.get(layer.get(c), "#94a3b8")
                                                     for c in vc.index]))
                st.plotly_chart(style_fig(fig, height=420, legend=False).update_layout(
                    title="Evidence behind flagged transactions (blue rule · purple ML · "
                          "yellow anomaly · red network)"), width="stretch")
        f = t[t["final_risk"] >= 61]
        if len(f):
            w = normalise(state.weights())
            parts = pd.DataFrame({"Transaction": w["transaction"] * f["transaction_risk"],
                                  "Behavioral": w["behavioral"] * f["behavioral_risk"],
                                  "Network": w["network"] * f["network_risk"]}).mean()
            c1, c2 = st.columns(2)
            with c1:
                fig = go.Figure(go.Bar(x=parts.index, y=parts.values,
                                       marker_color=["#60a5fa", "#34d399", "#ef4444"],
                                       text=[f"{v:.1f}" for v in parts.values], textposition="outside"))
                st.plotly_chart(style_fig(fig, height=300, legend=False).update_layout(
                    title="Average weighted contribution to the blend (flagged transactions)"),
                    width="stretch")
            with c2:
                g = f["guardrail"].replace("", "weighted blend").value_counts()
                fig = go.Figure(go.Pie(labels=g.index, values=g.values, hole=0.55,
                                       marker_colors=["#f97316", "#ef4444", "#60a5fa"]))
                st.plotly_chart(style_fig(fig, height=300).update_layout(
                    title="What decided the final score"), width="stretch")
        st.markdown("""
**How every score is built**

| Score | Built from | Formula |
|---|---|---|
| Transaction Risk | real-time rules, Isolation Forest anomaly, supervised ML | `100 × (1 − ∏(1 − wᵢ))` over named reasons |
| Behavioral Risk | 7 deviations from the account's own profile | `Σ weight × deviation` (amount .22, device .18, location .14, time .14, velocity .12, merchant .10, beneficiary .10) |
| Network Risk | ring evidence, controlled accounts, links to risky accounts | ring risk, or capped share of it |
| Final Risk | the three above | `wT·T + wB·B + wN·N`, never below a CRITICAL evidence-backed T or N |
| Ring Risk | kinds of ring evidence | `100 × (1 − ∏(1 − w_k))` |
""")
    with tabs[1]:
        flagged = t[t["final_risk"] >= 31].sort_values("final_risk", ascending=False)
        pool = list(flagged["txn_id"].head(300))
        if not pool:
            st.info("Nothing above LOW risk to explain.")
            return
        risk = t.set_index("txn_id")["final_risk"]
        tid = nav.pick(pool, "sel_txn", "Transaction", fmt=lambda x: f"{x} · risk {risk.get(x, 0):.0f}",
                       key="xai_tx")
        row = state.txn_row(tid)
        reasons = parse_reasons(row["reasons"])
        ring = state.ring(row["ring_id"]) if row["ring_id"] else None
        st.markdown(f"<div class='fi-card'>{esc(row['explanation'])}</div>", unsafe_allow_html=True)
        lines, _ = formula_text(row["transaction_risk"], row["behavioral_risk"], row["network_risk"],
                                row["network_evidence"], state.weights())
        st.code("\n".join(lines), language=None)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### WHY WAS THIS FLAGGED?")
            checklist(build_checklist(row, reasons, ring))
        with c2:
            wf = waterfall(reasons)
            if wf:
                fig = go.Figure(go.Bar(x=[w[2] for w in wf][::-1], y=[w[0] for w in wf][::-1],
                                       orientation="h", marker_color="#ef4444",
                                       hovertext=[w[4] for w in wf][::-1], hoverinfo="text+x"))
                fig.update_xaxes(range=[0, 1], title="evidence weight")
                st.plotly_chart(style_fig(fig, height=300, legend=False).update_layout(
                    title="Rule / model / network evidence weights"), width="stretch")
            contrib = contributions(bundle, row)
            top = [c for c in contrib if abs(c[3]) > 0.002][:10]
            if top:
                fig = go.Figure(go.Bar(
                    x=[c[3] * 100 for c in top][::-1], y=[c[1] for c in top][::-1], orientation="h",
                    marker_color=["#ef4444" if c[3] > 0 else "#22c55e" for c in top][::-1]))
                fig.update_xaxes(title="pts of fraud probability")
                method = "SHAP" if HAS_SHAP else "baseline substitution (exact, model-agnostic)"
                st.plotly_chart(style_fig(fig, height=340, legend=False).update_layout(
                    title=f"Top ML features · p = {row['ml_probability']:.0%} · {method}"),
                    width="stretch")
        nav.button("Open full investigation →", "transaction", "xai_go", sel_txn=tid)
