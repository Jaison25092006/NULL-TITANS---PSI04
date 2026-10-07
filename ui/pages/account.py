"""Account Intelligence: risk, behaviour profile, timeline, network and ring membership."""

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fraud.action_engine import ACCOUNT_ACTIONS
from fraud.fraud_dna import most_similar
from fraud.graph_analysis import neighbourhood, nid
from fraud.risk_engine import BAND_COLORS
from ui import nav, state
from ui.components import (STATUS_ICON, alert, badge, chips, esc, hero, kpi_row, network_figure,
                           score_card, style_fig)


def render():
    a, t = state.acc(), state.txn()
    hero("Account Intelligence",
         "Who is this customer normally, what changed, and who are they connected to?")
    risk = a.set_index("account_id")["account_risk"]
    c1, c2 = st.columns([3, 2])
    with c1:
        acct = nav.pick(list(a["account_id"]), "sel_account",
                        "Account (ranked by risk)", fmt=lambda x: f"{x} · risk {risk.get(x, 0):.0f}")
    with c2:
        only = st.radio("Show", ["All", "High-risk only", "Ring members"], horizontal=True)
        if only != "All":
            st.caption("Filter applies to the ranked table below.")
    row = state.account_row(acct)
    if row is None:
        st.info("Account not found.")
        return
    at = t[t["account_id"] == acct].sort_values("timestamp")
    ring = state.ring(row["ring_id"]) if row["ring_id"] else None

    c = st.columns([1.1, 3])
    score_card(c[0], "Account risk", row["account_risk"], ACCOUNT_ACTIONS[row["band"]][0])
    with c[1]:
        st.markdown(
            f"<div class='fi-card'><b style='font-size:1.2rem'>{acct}</b> &nbsp; "
            f"{badge(row['band'])} {badge(row['action'])}"
            f"{' ' + badge('MULE', '#ef4444') if row.get('is_mule') else ''}<br>"
            f"<span class='fi-muted'>Customer</span> {esc(row['customer_id'])} · "
            f"<span class='fi-muted'>Type</span> {row['account_type']} · "
            f"<span class='fi-muted'>Home</span> {esc(row['home_city'])} · "
            f"<span class='fi-muted'>Opened</span> {row['signup_date']}<br>"
            f"<span class='fi-muted'>Recommended:</span> {ACCOUNT_ACTIONS[row['band']][1]}</div>",
            unsafe_allow_html=True)
        st.write("")
        kpi_row([("Transactions", f"{len(at):,}", f"{row['n_flagged']} flagged"),
                 ("Devices", f"{int(row.get('n_devices', 0))}", ""),
                 ("Locations", f"{int(row.get('n_locations', 0))}", ""),
                 ("Merchants", f"{int(row.get('n_merchants', 0))}", ""),
                 ("Beneficiaries", f"{int(row.get('n_beneficiaries', 0))}", ""),
                 ("Behavior deviation", row["behavior_level"], f"max {row['max_behavioral']:.0f}/100",
                  BAND_COLORS.get({"HIGH": "CRITICAL", "MEDIUM": "MEDIUM", "LOW": "LOW"}
                                  [row["behavior_level"]]))])
    linked = [x for x in row["connected_accounts"].split(";") if x]
    if linked:
        st.markdown("**Connected suspicious accounts** " +
                    chips([f"{x} · {risk.get(x, 0):.0f}" for x in linked[:15]], "#ef4444"),
                    unsafe_allow_html=True)
    reasons = json.loads(row["reasons"]) if row["reasons"] else []
    if reasons:
        with st.expander("Why this account risk?", expanded=row["account_risk"] >= 61):
            for r in sorted(reasons, key=lambda r: -r["weight"]):
                st.markdown(f"- **{r['code']}** · weight {r['weight']:.2f} — {esc(r['evidence'])}")
            st.caption("Account risk = 1 − ∏(1 − weight) over these reasons.")
    else:
        alert("✅ No risk factors: this account behaves consistently with its profile.", "ok")
    if ring:
        c1, c2 = st.columns([4, 1])
        with c1:
            alert(f"🕸️ Member of <b>{ring['ring_id']}</b> · {ring['size']} accounts · ring risk "
                  f"<b>{ring['ring_risk']:.0f}</b> · {esc(ring['pattern'][:120])}", "crit")
        with c2:
            nav.button("Open ring →", "rings", "acc_ring", sel_ring=ring["ring_id"])

    tabs = st.tabs(["📈 Timeline & risk evolution", "🧾 Transaction history",
                    "🧬 Behavioral profile", "🌐 Network connections", "🔬 Fraud DNA"])
    with tabs[0]:
        if at.empty:
            st.info("No transactions.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=at["timestamp"], y=at["final_risk"], mode="markers", name="transaction",
                marker=dict(color=[BAND_COLORS[b] for b in at["band"]],
                            size=(at["amount_inr"].clip(lower=1) ** 0.33).clip(5, 30)),
                hovertext=[f"{r.txn_id} · ₹{r.amount_inr:,.0f} · {r.merchant_name or '→ ' + r.payee_account_id}"
                           f" · risk {r.final_risk:.0f}" for r in at.itertuples()], hoverinfo="text"))
            evo = at.set_index("timestamp")["final_risk"].rolling("7D").max()
            fig.add_trace(go.Scatter(x=evo.index, y=evo.values, name="risk evolution (7-day max)",
                                     line=dict(color="#60a5fa", width=2, shape="hv")))
            for y, col in ((61, "#f97316"), (81, "#ef4444")):
                fig.add_hline(y=y, line=dict(color=col, dash="dot", width=1))
            fig.update_yaxes(range=[-3, 103], title="final risk")
            st.plotly_chart(style_fig(fig, height=360).update_layout(
                title="Account timeline (bubble size = amount)"), width="stretch")
    with tabs[1]:
        view = at.sort_values("timestamp", ascending=False)
        st.dataframe(pd.DataFrame({
            "Transaction": view["txn_id"], "Time": view["timestamp"].dt.strftime("%d %b %H:%M"),
            "Type": view["txn_type"], "Merchant / Beneficiary": view["merchant_name"].where(
                view["merchant_name"] != "", "→ " + view["payee_account_id"]),
            "Amount (₹)": view["amount_inr"].round(0), "Device": view["device_id"],
            "Location": view["city"] + ", " + view["country"],
            "T": view["transaction_risk"], "B": view["behavioral_risk"], "N": view["network_risk"],
            "Final": view["final_risk"], "Status": view["status"].map(STATUS_ICON)}),
            hide_index=True, width="stretch", height=380,
            column_config={"Final": st.column_config.ProgressColumn("Final", min_value=0,
                                                                    max_value=100, format="%d")})
        risky = view[view["final_risk"] >= 61]
        if len(risky):
            pick = st.selectbox("Investigate a flagged transaction", risky["txn_id"])
            nav.button("Open transaction →", "transaction", "acc_tx", sel_txn=pick)
    with tabs[2]:
        prof = [("Normal amount range", f"₹{row.get('amount_p10', 0):,.0f} – ₹{row.get('amount_p90', 0):,.0f}"),
                ("Average amount", f"₹{row.get('avg_amount', 0):,.0f}"),
                ("Average daily transactions", f"{row.get('avg_daily_txns', 0):.1f}"),
                ("Typical time", row.get("typical_hours", "")),
                ("Typical locations", row.get("typical_locations", "")),
                ("Typical devices", row.get("typical_devices", "") or "card-present only"),
                ("Typical merchants", row.get("typical_merchants", "")),
                ("Typical categories", row.get("typical_categories", "")),
                ("Payment channels", row.get("typical_channels", "")),
                ("Normal beneficiaries", row.get("normal_beneficiaries", "") or "—")]
        c1, c2 = st.columns([1, 1.2])
        with c1:
            st.markdown("".join(f"<div class='fi-check miss' style='color:#cbd5e1'><b>{k}</b>"
                                f"<br><small>{esc(v)}</small></div>" for k, v in prof),
                        unsafe_allow_html=True)
        with c2:
            hours = at["timestamp"].dt.hour.value_counts().reindex(range(24), fill_value=0)
            fig = go.Figure(go.Bar(x=hours.index, y=hours.values, marker_color="#60a5fa"))
            flag_h = at[at["final_risk"] >= 61]["timestamp"].dt.hour.value_counts()
            if len(flag_h):
                fig.add_trace(go.Bar(x=flag_h.index, y=flag_h.values, marker_color="#ef4444",
                                     name="flagged"))
            st.plotly_chart(style_fig(fig, height=230, legend=False).update_layout(
                title="Hour-of-day profile (red = flagged)", barmode="overlay"), width="stretch")
            mix = at["merchant_category"].value_counts().head(8)
            fig = go.Figure(go.Bar(x=mix.values, y=mix.index, orientation="h",
                                   marker_color="#34d399"))
            st.plotly_chart(style_fig(fig, height=230, legend=False).update_layout(
                title="Spending mix"), width="stretch")
    with tabs[3]:
        G, _ = state.graphs()
        H = neighbourhood(G, nid("account", acct), radius=2, max_nodes=90,
                          types=("account", "device", "beneficiary", "merchant", "location",
                                 "transaction"))
        fig, _ = network_figure(H, height=520, highlight=[nid("account", acct)])
        st.plotly_chart(fig, width="stretch", key="acc_graph")
        nav.button("Explore in the full network graph →", "network", "acc_net",
                   sel_node=nid("account", acct))
    with tabs[4]:
        d = state.data()
        z, radar = d["dna_z"], d["dna_radar"]
        if acct not in z.index:
            st.info("No fingerprint for this account.")
        else:
            sims = most_similar(z, acct, k=8)
            c1, c2 = st.columns([1, 1])
            with c1:
                fig = go.Figure()
                cats = list(radar.columns)
                fig.add_trace(go.Scatterpolar(r=list(radar.loc[acct]) + [radar.loc[acct].iloc[0]],
                                              theta=cats + [cats[0]], fill="toself", name=acct,
                                              line_color="#ef4444"))
                if len(sims):
                    b = sims.index[0]
                    fig.add_trace(go.Scatterpolar(r=list(radar.loc[b]) + [radar.loc[b].iloc[0]],
                                                  theta=cats + [cats[0]], fill="toself",
                                                  name=f"{b} ({sims.iloc[0]:.0%} similar)",
                                                  line_color="#60a5fa", opacity=0.6))
                fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)",
                                             radialaxis=dict(range=[0, 1], gridcolor="#1e3352")))
                st.plotly_chart(style_fig(fig, height=380).update_layout(title="Fraud DNA"),
                                width="stretch")
            with c2:
                st.markdown("**Most similar behavioural fingerprints**")
                st.dataframe(pd.DataFrame({
                    "Account": sims.index, "Similarity": (sims.values * 100).round(1),
                    "Account risk": [risk.get(x, 0) for x in sims.index],
                    "Same ring": ["✓" if ring and x in ring["members"] else "" for x in sims.index]}),
                    hide_index=True, width="stretch",
                    column_config={"Similarity": st.column_config.ProgressColumn(
                        "Similarity %", min_value=0, max_value=100, format="%.0f%%")})
                st.caption("Cosine similarity of standardised fingerprints (amount, time, device, "
                           "location, merchant, velocity, network, sequence). ≥ 80% = same playbook.")
