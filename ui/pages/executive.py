import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from fraud.config import FLAG_THRESHOLD
from fraud.risk_engine import BAND_COLORS
from ui import nav, state
from ui.components import alert, badge, esc, hero, kpi_row, money, style_fig

CITY_COORDS = {
    "Chennai": (13.08, 80.27), "Coimbatore": (11.02, 76.96), "Bengaluru": (12.97, 77.59),
    "Mumbai": (19.08, 72.88), "Delhi": (28.61, 77.21), "New Delhi": (28.61, 77.21),
    "Hyderabad": (17.39, 78.49), "Kochi": (9.93, 76.27), "Pune": (18.52, 73.86),
    "Singapore": (1.35, 103.82), "Dubai": (25.20, 55.27), "London": (51.51, -0.13),
    "Frankfurt": (50.11, 8.68), "Bangkok": (13.76, 100.50),
}
INNOVATIONS = [
    ("Fraud Ring Intelligence", "Graph links accounts through devices, beneficiaries, merchants and sequences"),
    ("Fraud DNA", "Behavioural fingerprints reveal accounts following the same playbook"),
    ("Temporal Pattern Detection", "Coordinated activity found in 5-min to 24-h windows"),
    ("Explainable Risk Scoring", "Every score is a sum of named, human-readable evidence"),
    ("Multi-Level Risk", "Transaction · Behavioral · Network → Final, plus account & ring risk"),
    ("Context-Aware FP Prevention", "Judged against the account's own normal: big ≠ fraud"),
    ("Real-Time Simulation", "Causal scoring: each event uses only the past, in milliseconds"),
    ("Recommended Response Engine", "ALLOW · MONITOR · STEP-UP · BLOCK and ring playbooks"),
]


def metrics(t, labels):
    if labels is None:
        return None
    d = t[["txn_id", "final_risk"]].merge(labels[["txn_id", "is_fraud"]], on="txn_id")
    if d.empty:
        return None
    f, y = d["final_risk"] >= FLAG_THRESHOLD, d["is_fraud"]
    tp, fp = int((f & y).sum()), int((f & ~y).sum())
    fn, tn = int((~f & y).sum()), int((~f & ~y).sum())
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0,
            "fpr": fp / (fp + tn) if fp + tn else 0}


def render():
    d = state.data()
    t, a, rings = state.txn(), state.acc(), state.rings()
    period = f"{t['timestamp'].min():%d %b %Y} → {t['timestamp'].max():%d %b %Y}"
    hero("Fraud Intelligence Command Center",
         f"{esc(d['source'])} · {period} · every score explained · decision support for analysts")

    flagged = t[t["final_risk"] >= FLAG_THRESHOLD]
    m = metrics(t, d["labels"])
    pct = (lambda x: f"{100 * x:.1f}%") if m else (lambda x: "n/a")
    kpi_row([
        ("Transactions analyzed", f"{len(t):,}", money(t["amount_inr"].sum()) + " volume"),
        ("Frauds detected", f"{len(flagged):,}", f"{money(flagged['amount_inr'].sum())} at risk",
         "#ef4444"),
        ("Fraud rings", f"{len(rings)}", f"{sum(r['size'] for r in rings)} accounts involved",
         "#f97316" if rings else None),
        ("Accounts monitored", f"{len(a):,}", f"{int((a['account_risk'] >= 61).sum())} high-risk"),
        ("Critical alerts", f"{int((t['band'] == 'CRITICAL').sum()):,}", "BLOCK + INVESTIGATE",
         "#ef4444"),
    ])
    st.write("")
    kpi_row([
        ("High-risk transactions", f"{int((t['band'] == 'HIGH').sum()):,}", "STEP-UP authentication",
         "#f97316"),
        ("Model precision", pct(m["precision"]) if m else "n/a", "of flagged are real fraud",
         "#22c55e" if m else None),
        ("Model recall", pct(m["recall"]) if m else "n/a", "of fraud is caught"),
        ("F1 score", pct(m["f1"]) if m else "n/a", "balance of both"),
        ("False positive rate", f"{100 * m['fpr']:.3f}%" if m else "n/a",
         "legit transactions flagged", "#22c55e" if m else None),
    ])
    if m is None:
        st.caption("Accuracy metrics need fraud labels; this dataset has none.")

    st.write("")
    if rings:
        top = rings[0]
        c1, c2 = st.columns([5, 1])
        with c1:
            alert(f"🚨 <b>FRAUD RING DETECTED — {top['ring_id']}</b> · {top['size']} accounts · "
                  f"Ring Risk <b>{top['ring_risk']:.0f}/100</b> · {esc(top['pattern'][:110])} "
                  f"&nbsp; {badge(top['recommended_action'].split(' (')[0], '#ef4444')}", "crit")
        with c2:
            nav.button("Investigate ring →", "rings", "exec_ring", sel_ring=top["ring_id"])
    big_ok = t[(t["amount_inr"] >= 100000) & (t["final_risk"] < FLAG_THRESHOLD)]
    alert(f"🛡️ <b>False-positive protection:</b> {len(big_ok):,} transactions above ₹1 lakh "
          f"({money(big_ok['amount_inr'].sum())}) were <b>allowed</b> because they match their "
          f"account's normal behaviour. <i>High amount alone is insufficient evidence.</i>", "ok")

    c1, c2 = st.columns(2)
    with c1:
        bins = pd.cut(t["final_risk"], [-0.1, 30.5, 60.5, 80.5, 100],
                      labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"])
        counts = bins.value_counts().reindex(["LOW", "MEDIUM", "HIGH", "CRITICAL"]).fillna(0)
        fig = go.Figure(go.Bar(x=counts.index, y=counts.values, text=[f"{int(v):,}" for v in counts],
                               textposition="outside",
                               marker_color=[BAND_COLORS[b] for b in counts.index]))
        fig.update_yaxes(type="log", title="transactions (log scale)")
        st.plotly_chart(style_fig(fig, legend=False).update_layout(
            title="Transaction risk distribution"), width="stretch")
    with c2:
        if d["labels"] is not None:
            x = t[["txn_id", "final_risk"]].merge(d["labels"][["txn_id", "is_fraud"]], on="txn_id")
            lab = (x["final_risk"] >= FLAG_THRESHOLD).map({True: "Flagged", False: "Allowed"}) + \
                " · " + x["is_fraud"].map({True: "fraud", False: "legitimate"})
            vc = lab.value_counts()
        else:
            vc = (t["final_risk"] >= FLAG_THRESHOLD).map({True: "Flagged", False: "Allowed"}).value_counts()
        colors = {"Allowed · legitimate": "#22c55e", "Flagged · fraud": "#ef4444",
                  "Flagged · legitimate": "#eab308", "Allowed · fraud": "#a855f7",
                  "Allowed": "#22c55e", "Flagged": "#ef4444"}
        fig = go.Figure(go.Pie(labels=vc.index, values=vc.values, hole=0.6,
                               marker_colors=[colors.get(k, "#64748b") for k in vc.index],
                               textinfo="value"))
        st.plotly_chart(style_fig(fig).update_layout(title="Fraud vs legitimate transactions"),
                        width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        day = t.assign(day=t["timestamp"].dt.floor("D"))
        daily = day.groupby(["day", "band"]).size().unstack(fill_value=0)
        fig = go.Figure()
        for b in ("HIGH", "CRITICAL"):
            if b in daily:
                fig.add_trace(go.Bar(x=daily.index, y=daily[b], name=b, marker_color=BAND_COLORS[b]))
        mean = day.groupby("day")["final_risk"].mean()
        fig.add_trace(go.Scatter(x=mean.index, y=mean.values, name="mean risk", yaxis="y2",
                                 line=dict(color="#60a5fa", width=2)))
        fig.update_layout(barmode="stack", yaxis2=dict(overlaying="y", side="right",
                                                       showgrid=False, title="mean risk"))
        st.plotly_chart(style_fig(fig).update_layout(title="Risk over time"), width="stretch")
    with c2:
        if rings:
            fig = go.Figure(go.Bar(
                x=[r["ring_id"] for r in rings], y=[r["size"] for r in rings],
                text=[f"risk {r['ring_risk']:.0f}" for r in rings], textposition="outside",
                marker_color=[BAND_COLORS[r["band"]] for r in rings],
                hovertext=[r["pattern"] for r in rings]))
            fig.update_yaxes(title="accounts in ring")
            st.plotly_chart(style_fig(fig, legend=False).update_layout(
                title=f"Fraud rings ({len(rings)}) by size and risk"), width="stretch")
        else:
            st.info("No fraud rings detected in this dataset.")

    c1, c2 = st.columns(2)
    with c1:
        g = t.groupby("city").agg(n=("txn_id", "size"),
                                  flagged=("final_risk", lambda s: int((s >= FLAG_THRESHOLD).sum())))
        g = g[g.index.isin(CITY_COORDS)]
        if len(g):
            g = g.assign(lat=[CITY_COORDS[c][0] for c in g.index],
                         lon=[CITY_COORDS[c][1] for c in g.index]).reset_index()
            fig = px.scatter_geo(g, lat="lat", lon="lon", size="n", color="flagged",
                                 hover_name="city", color_continuous_scale="OrRd", size_max=32,
                                 projection="natural earth")
            fig.update_geos(bgcolor="rgba(0,0,0,0)", showcountries=True, countrycolor="#334155",
                            showland=True, landcolor="#0f1b2d", showocean=False,
                            lataxis_range=[-5, 58], lonaxis_range=[-10, 115])
            st.plotly_chart(style_fig(fig, legend=False).update_layout(
                title="Geographic distribution (bubble = volume, colour = flagged)"),
                width="stretch")
        else:
            top = t["city"].value_counts().head(12)
            st.plotly_chart(style_fig(go.Figure(go.Bar(x=top.index, y=top.values))).update_layout(
                title="Transactions by location"), width="stretch")
    with c2:
        ch = t.groupby(["channel", "band"]).size().unstack(fill_value=0)
        fig = go.Figure()
        for b in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
            if b in ch:
                fig.add_trace(go.Bar(y=ch.index, x=ch[b], name=b, orientation="h",
                                     marker_color=BAND_COLORS[b]))
        fig.update_layout(barmode="stack")
        fig.update_xaxes(type="log", title="transactions (log)")
        st.plotly_chart(style_fig(fig).update_layout(title="Payment channel distribution"),
                        width="stretch")

    st.subheader("Core innovations")
    for row in (INNOVATIONS[:4], INNOVATIONS[4:]):
        cols = st.columns(4)
        for col, (name, desc) in zip(cols, row):
            col.markdown(f"<div class='fi-innov'><b>{esc(name)}</b><div>{esc(desc)}</div></div>",
                         unsafe_allow_html=True)
        st.write("")
    with st.expander("Why this is not a simple fraud classifier"):
        st.markdown(
            "Traditional fraud detection asks: **Is this transaction fraudulent?**\n\n"
            "This platform asks: **Is it unusual? Why? Who is it connected to? Is it part of a "
            "coordinated fraud ring? What evidence supports the decision? What should the "
            "analyst do next?**")
