"""Fraud Ring Detection: ring risk, evidence, graph, timeline, shared behaviour, playbook."""

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fraud.action_engine import DISCLAIMER
from fraud.fraud_dna import similarity
from fraud.graph_analysis import nid
from fraud.risk_engine import BAND_COLORS
from fraud.temporal_analysis import coordination_bursts, coordinated_sequences
from ui import nav, state
from ui.components import alert, badge, esc, hero, kpi_row, network_figure, style_fig


def ring_graph(G, ring):
    """Ring members plus the entities that appear in the ring's evidence."""
    ev = state.txn()
    ev = ev[ev["txn_id"].isin(set(ring["evidence_txns"]))]
    keep = {nid("account", m) for m in ring["members"]}
    for col, kind in (("device_id", "device"), ("merchant_id", "merchant"),
                      ("payee_account_id", "beneficiary")):
        keep |= {nid(kind, v) for v in set(ev[col]) - {""}}
    keep |= {nid("location", c) for c in ring.get("locations", [])}
    return G.subgraph([n for n in keep if n in G]).copy()


def render():
    rings = state.rings()
    hero("Fraud Ring Detection",
         "Groups of accounts acting together: shared devices, beneficiaries, merchants, "
         "sequences and timing — where no single transaction looks bad")
    if not rings:
        alert("✅ No fraud rings detected in this dataset. Individual transaction and account "
              "risk scoring is still active.", "ok")
        return
    cols = st.columns(min(len(rings), 5))
    for col, r in zip(cols, rings[:5]):
        col.markdown(f"<div class='fi-kpi' style='border-color:{BAND_COLORS[r['band']]}'>"
                     f"<div class='lbl'>{r['ring_id']} · {r['size']} accounts</div>"
                     f"<div class='val' style='color:{BAND_COLORS[r['band']]}'>"
                     f"{r['ring_risk']:.0f}<span style='font-size:.9rem;color:#64748b'>/100</span></div>"
                     f"<div class='sub'>{esc(r['pattern'].split(':')[0])}</div></div>",
                     unsafe_allow_html=True)
    st.write("")
    ids = [r["ring_id"] for r in rings]
    rid = nav.pick(ids, "sel_ring", "Select ring",
                   fmt=lambda x: f"{x} · {state.ring(x)['size']} accounts · risk "
                                 f"{state.ring(x)['ring_risk']:.0f}")
    ring = state.ring(rid)
    t = state.txn()
    ev = t[t["txn_id"].isin(set(ring["evidence_txns"]))].sort_values("timestamp")

    alert(f"🚨 <b>FRAUD RING DETECTED — {ring['ring_id']}</b> · Ring Risk "
          f"<b>{ring['ring_risk']:.0f}/100</b> · {esc(ring['pattern'])} &nbsp; "
          f"{badge(ring['recommended_action'].split(' (')[0], BAND_COLORS[ring['band']])}",
          "crit" if ring["band"] == "CRITICAL" else "high")
    kpi_row([("Ring risk", f"{ring['ring_risk']:.0f}", ring["band"], BAND_COLORS[ring["band"]]),
             ("Accounts", ring["size"], f"{len(ring['mules'])} mule(s)"),
             ("Devices", len(ring["all_devices"]), f"{len(ring['devices'])} shared"),
             ("Merchants", len(ring["merchants"]), ""),
             ("Beneficiaries", len(ring["beneficiaries"]), ", ".join(ring["beneficiaries"][:2])),
             ("Locations", len(ring["locations"]), ""),
             ("Money involved", f"₹{ring['amount_inr'] / 1e5:,.1f} L", "")])

    st.markdown("### Evidence")
    c1, c2 = st.columns([1.25, 1])
    with c1:
        bd = ring["evidence_breakdown"]
        st.dataframe(pd.DataFrame({"Evidence": [b["label"] for b in bd],
                                   "Weight": [b["weight"] for b in bd],
                                   "Detail": [b["detail"] for b in bd]}),
                     hide_index=True, width="stretch")
        prod = " × ".join(f"(1 − {b['weight']:.2f})" for b in bd)
        st.code(f"Ring Risk = 100 × [1 − {prod}] = {ring['ring_risk']:.1f}", language=None)
    with c2:
        st.markdown("**Evidence summary**")
        pw = ring.get("peak_window") or {}
        items = [f"{ring['size']} accounts connected"]
        if ring["devices"]:
            items.append(f"{len(ring['devices'])} shared / linked device(s): "
                         f"{', '.join(ring['devices'][:4])}")
        if ring["beneficiaries"]:
            items.append(f"{len(ring['beneficiaries'])} common beneficiar"
                         f"{'y' if len(ring['beneficiaries']) == 1 else 'ies'}: "
                         f"{', '.join(ring['beneficiaries'][:3])}")
        if "same_sequence" in ring["evidence_kinds"]:
            items.append("similar transaction sequences")
        if pw.get("members_active", 0) >= 3:
            items.append(f"{pw['members_active']} accounts active within {pw['span_min']:.0f} minutes")
        if ring.get("dna_similarity"):
            items.append(f"behavioural similarity {ring['dna_similarity']:.0%} (Fraud DNA)")
        if len(ring["locations"]) >= 3:
            items.append(f"unusual geographic spread: {len(ring['locations'])} cities")
        st.markdown("\n".join(f"- {i}" for i in items))
        st.caption(f"Active {ring['first_seen'][:16]} → {ring['last_seen'][:16]}")

    G, _ = state.graphs()
    c1, c2 = st.columns([1.3, 1])
    with c1:
        H = ring_graph(G, ring)
        fig, _ = network_figure(H, height=520, highlight=[nid("account", m) for m in ring["mules"]])
        st.plotly_chart(fig, width="stretch", key="ring_graph")
    with c2:
        members = pd.DataFrame({"Account": ring["members"]})
        a = state.acc().set_index("account_id")
        members["Risk"] = [a["account_risk"].get(m, 0) for m in ring["members"]]
        members["City"] = [a["home_city"].get(m, "") for m in ring["members"]]
        members["Role"] = ["mule / beneficiary" if m in ring["mules"] else
                           "synthetic identity" if m in ring.get("synthetic_identities", []) else
                           "member" for m in ring["members"]]
        st.dataframe(members.sort_values("Risk", ascending=False), hide_index=True,
                     width="stretch", height=300)
        pick = st.selectbox("Open member account", ring["members"])
        nav.button("Open account →", "account", "ring_acc", sel_account=pick)

    st.markdown("### Temporal sequence")
    if ev.empty:
        st.info("No evidence transactions.")
    else:
        pw = ring.get("peak_window") or {}
        actors = [m for m in ring["members"] if m not in ring["mules"]]
        focus = ev
        if pw.get("start"):
            s = pd.Timestamp(pw["start"]) - pd.Timedelta(hours=1)
            e = pd.Timestamp(pw["end"]) + pd.Timedelta(hours=8)
            win = ev[(ev["timestamp"] >= s) & (ev["timestamp"] <= e)]
            if len(win) >= 3:
                focus = win
        focus = focus.assign(step=np.where(focus["payee_account_id"] != "",
                                           "→ " + focus["payee_account_id"], focus["merchant_name"]))
        fig = go.Figure()
        for step, g in focus.groupby("step"):
            fig.add_trace(go.Scatter(
                x=g["timestamp"], y=g["account_id"], mode="markers", name=step,
                marker=dict(size=13, line=dict(width=1, color="#0b1220")),
                hovertext=[f"{r.timestamp:%H:%M} · {r.account_id} · {step} · ₹{r.amount_inr:,.0f}"
                           for r in g.itertuples()], hoverinfo="text"))
        fig.update_yaxes(categoryorder="array", categoryarray=sorted(focus["account_id"].unique()))
        st.plotly_chart(style_fig(fig, height=420).update_layout(
            title="Who did what, when (zoomed to the coordinated window)"), width="stretch")
        bursts = coordination_bursts(focus, state.data()["merchants"], state.data()["accounts"],
                                     window_min=15)
        seqs = coordinated_sequences(bursts, window_min=15)
        for s in seqs.itertuples():
            if len(set(s.accounts) & set(ring["members"])) >= 3:
                alert(f"⏱️ <b>{esc(s.narrative)}</b> ({' → '.join(s.steps)})", "high")
        log = focus.head(40)
        st.dataframe(pd.DataFrame({"Time": log["timestamp"].dt.strftime("%d %b %H:%M"),
                                   "Account": log["account_id"], "Step": log["step"],
                                   "Amount (₹)": log["amount_inr"].round(0),
                                   "Device": log["device_id"], "City": log["city"],
                                   "Risk": log["final_risk"]}),
                     hide_index=True, width="stretch", height=260)

    st.markdown("### Common behavioural patterns")
    d = state.data()
    actors = [m for m in ring["members"] if m not in ring["mules"]]
    sim = similarity(d["dna_z"], actors)
    c1, c2 = st.columns(2)
    with c1:
        if len(sim) >= 2:
            fig = go.Figure(go.Heatmap(z=sim.values * 100, x=sim.columns, y=sim.index,
                                       colorscale="Reds", zmin=0, zmax=100,
                                       colorbar=dict(title="%")))
            st.plotly_chart(style_fig(fig, height=380, legend=False).update_layout(
                title="Fraud DNA similarity between members"), width="stretch")
    with c2:
        radar = d["dna_radar"]
        cats = list(radar.columns)
        inside = radar.loc[[m for m in actors if m in radar.index]].mean()
        outside = radar.drop(index=[m for m in ring["members"] if m in radar.index]).mean()
        fig = go.Figure()
        for vals, name, col in ((inside, "ring members", "#ef4444"), (outside, "everyone else", "#60a5fa")):
            fig.add_trace(go.Scatterpolar(r=list(vals) + [vals.iloc[0]], theta=cats + [cats[0]],
                                          fill="toself", name=name, line_color=col))
        fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)",
                                     radialaxis=dict(range=[0, 1], gridcolor="#1e3352")))
        st.plotly_chart(style_fig(fig, height=380).update_layout(title="Average behaviour profile"),
                        width="stretch")

    st.markdown("### Recommended action")
    alert(f"<b>{ring['recommended_action']}</b>", "crit" if ring["band"] == "CRITICAL" else "high")
    for i, a_ in enumerate(ring["actions"], 1):
        st.markdown(f"**{i}. {a_['step']}** — {esc(a_['detail'])}")
    st.caption(DISCLAIMER)
