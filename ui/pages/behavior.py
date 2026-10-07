"""Behavioral Analysis: profiles and deviation, Fraud DNA, and temporal patterns."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fraud.behavioral_analysis import DIMENSIONS, deviation_findings
from fraud.config import TIME_WINDOWS
from fraud.fraud_dna import most_similar, similarity
from fraud.temporal_analysis import coordination_bursts, coordinated_sequences, hour_day_heatmap
from ui import nav, state
from ui.components import alert, esc, hero, style_fig

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


@st.cache_data(show_spinner="Scanning for coordinated activity…")
def _temporal(key, window, min_accounts):
    d = state.data()
    bursts = coordination_bursts(d["txn"], d["merchants"], d["accounts"], window, min_accounts)
    return bursts, coordinated_sequences(bursts, window, min_shared=min_accounts)


def render():
    hero("Behavioral Analysis",
         "Each account's normal behaviour · how far a transaction deviates · behavioural "
         "fingerprints (Fraud DNA) · coordinated timing")
    tabs = st.tabs(["🧭 Profile & deviation", "🧬 Fraud DNA", "⏱️ Temporal patterns"])
    with tabs[0]:
        profile_tab()
    with tabs[1]:
        dna_tab()
    with tabs[2]:
        temporal_tab()


def profile_tab():
    a, t = state.acc(), state.txn()
    risk = a.set_index("account_id")["account_risk"]
    acct = nav.pick(list(a["account_id"]), "sel_account", "Account",
                    fmt=lambda x: f"{x} · risk {risk.get(x, 0):.0f}", key="beh_acc")
    row = state.account_row(acct)
    at = t[t["account_id"] == acct]
    if row is None or at.empty:
        st.info("No data for this account.")
        return
    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.markdown(f"#### Normal behaviour of {acct}")
        st.markdown(
            f"<div class='fi-card'>"
            f"<b>Average amount:</b> ₹{row.get('avg_amount', 0):,.0f} "
            f"(normal range ₹{row.get('amount_p10', 0):,.0f} – ₹{row.get('amount_p90', 0):,.0f})<br>"
            f"<b>Typical location:</b> {esc(row.get('typical_locations', ''))}<br>"
            f"<b>Typical time:</b> {row.get('typical_hours', '')}<br>"
            f"<b>Typical device:</b> {esc(row.get('typical_devices', '') or 'card-present only')}<br>"
            f"<b>Typical merchants:</b> {esc(row.get('typical_merchants', ''))}<br>"
            f"<b>Payment channels:</b> {esc(row.get('typical_channels', ''))}<br>"
            f"<b>Average daily transactions:</b> {row.get('avg_daily_txns', 0):.1f}<br>"
            f"<b>Normal beneficiaries:</b> {esc(row.get('normal_beneficiaries', '') or '—')}"
            f"</div>", unsafe_allow_html=True)
    with c2:
        top = at.sort_values("behavioral_risk", ascending=False)
        tid = st.selectbox("Compare a transaction (most deviating first)", top["txn_id"].head(60),
                           format_func=lambda x: f"{x} · deviation "
                           f"{top.set_index('txn_id').loc[x, 'behavioral_risk']:.0f}")
        r = top.set_index("txn_id").loc[tid]
        st.markdown(f"#### New transaction {tid}")
        st.markdown(
            f"<div class='fi-card'><b>Amount:</b> ₹{r['amount_inr']:,.0f} · "
            f"<b>Time:</b> {r['timestamp']:%H:%M} · <b>Location:</b> {esc(r['city'])}, "
            f"{r['country']} · <b>Device:</b> {esc(r['device_id'] or 'card-present')} · "
            f"<b>Merchant/Beneficiary:</b> {esc(r['merchant_name'] or r['payee_account_id'])}"
            f"</div>", unsafe_allow_html=True)
        findings = deviation_findings(r)
        st.markdown(f"**Behavioral deviation score: {r['behavioral_risk']:.0f}/100**")
        if findings:
            st.markdown("\n".join(f"- ⚠️ {text}" for _, text in findings))
        else:
            st.success("Consistent with this account's normal behaviour on every dimension.")
    dims = list(DIMENSIONS)
    fig = go.Figure(go.Bar(x=[DIMENSIONS[k][1] for k in dims], y=[float(r[f"b_{k}"]) for k in dims],
                           marker_color=["#ef4444" if float(r[f"b_{k}"]) >= 0.5 else
                                         "#f97316" if float(r[f"b_{k}"]) > 0 else "#334155"
                                         for k in dims],
                           text=[f"w={DIMENSIONS[k][0]:.2f}" for k in dims], textposition="outside"))
    fig.update_yaxes(range=[0, 1.2], tickformat=".0%", title="deviation")
    st.plotly_chart(style_fig(fig, height=300, legend=False).update_layout(
        title="Deviation per dimension (Behavioral Risk = Σ weight × deviation)"), width="stretch")
    st.caption("A single unusual dimension can contribute at most 22 points, so a large amount "
               "on a known device, at a normal time, from a normal place stays LOW. "
               "High amount alone is insufficient evidence.")


def dna_tab():
    d = state.data()
    z, radar = d["dna_z"], d["dna_radar"]
    rings = state.rings()
    risk = state.acc().set_index("account_id")["account_risk"]
    default = [m for m in rings[0]["members"] if m not in rings[0]["mules"]][0] if rings else z.index[0]
    accounts = list(risk.index)
    c1, c2 = st.columns([1, 2])
    acct = c1.selectbox("Suspicious account", accounts,
                        index=accounts.index(default) if default in accounts else 0, key="dna_acc")
    sims = most_similar(z, acct, k=10)
    with c1:
        st.markdown("**Most similar fingerprints**")
        for b, s in sims.head(6).items():
            col = "#ef4444" if s >= 0.8 else "#94a3b8"
            st.markdown(f"{acct} ↔ {b} &nbsp; <b style='color:{col}'>{s:.0%}</b> "
                        f"<span class='fi-muted'>(risk {risk.get(b, 0):.0f})</span>",
                        unsafe_allow_html=True)
        high = sims[sims >= 0.8]
        if len(high) >= 2:
            alert(f"🧬 {len(high)} accounts share ≥80% of {acct}'s behavioural fingerprint "
                  f"despite different identities — network risk raised.", "crit")
    with c2:
        group = [acct] + list(sims.head(7).index)
        sim = similarity(z, group)
        fig = go.Figure(go.Heatmap(z=sim.values * 100, x=sim.columns, y=sim.index,
                                   colorscale="Reds", zmin=0, zmax=100, colorbar=dict(title="%"),
                                   text=(sim.values * 100).round(0), texttemplate="%{text}"))
        st.plotly_chart(style_fig(fig, height=380, legend=False).update_layout(
            title="Behaviour similarity matrix"), width="stretch")
    cats = list(radar.columns)
    fig = go.Figure()
    for x, col in zip(group[:4], ["#ef4444", "#f97316", "#eab308", "#60a5fa"]):
        if x in radar.index:
            v = list(radar.loc[x])
            fig.add_trace(go.Scatterpolar(r=v + [v[0]], theta=cats + [cats[0]], fill="toself",
                                          name=x, line_color=col, opacity=0.7))
    fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)",
                                 radialaxis=dict(range=[0, 1], gridcolor="#1e3352")))
    st.plotly_chart(style_fig(fig, height=400).update_layout(
        title="Fraud DNA: amount · time · device · location · merchant · velocity · network · sequence"),
        width="stretch")
    st.caption("Fingerprints are built from each account's incident window (ring evidence and "
               "high-risk activity, or its latest activity), standardised against the population "
               "and compared with cosine similarity.")


def temporal_tab():
    c1, c2 = st.columns([2, 1])
    window_label = c1.radio("Time window", list(TIME_WINDOWS), index=1, horizontal=True)
    min_acc = c2.slider("Min. accounts", 3, 8, 3)
    window = TIME_WINDOWS[window_label]
    key = st.session_state.get("upload_key", "default")
    bursts, seqs = _temporal(key, window, min_acc)
    sus = bursts[~bursts["benign_hub"].astype(bool)] if len(bursts) else bursts
    c = st.columns(3)
    c[0].metric("Coordinated sequences", len(seqs))
    c[1].metric("Coordinated bursts", len(sus))
    c[2].metric("Benign hub bursts (ignored)", int(bursts["benign_hub"].astype(bool).sum()) if len(bursts) else 0,
                help="Many people paying a business (e.g. rent to a landlord) at the same time")
    if len(seqs):
        st.markdown("#### Coordinated multi-step sequences")
        for s in seqs.sort_values("n_accounts", ascending=False).head(8).itertuples():
            alert(f"⏱️ <b>{esc(s.narrative)}</b><br><span class='fi-muted'>"
                  f"{s.start:%d %b %H:%M} → {s.end:%H:%M} · steps: {' → '.join(map(str, s.steps))} · "
                  f"accounts: {', '.join(s.accounts[:10])}</span>", "crit")
    else:
        st.info(f"No multi-step coordination found within {window_label}.")
    if len(sus):
        st.markdown("#### Bursts: ≥ %d accounts hitting the same target within %s" % (min_acc, window_label))
        st.dataframe(pd.DataFrame({
            "Start": sus["start"].dt.strftime("%d %b %H:%M"), "Target": sus["target_name"],
            "Type": sus["target_type"], "Accounts": sus["n_accounts"],
            "Span (min)": sus["span_min"], "Transactions": sus["n_txns"],
            "Amount (₹)": sus["amount_inr"].round(0),
            "Members": sus["accounts"].map(lambda x: ", ".join(x[:8]))}).head(30),
            hide_index=True, width="stretch")
    t = state.txn()
    hm = hour_day_heatmap(t, t["final_risk"] >= 61)
    fig = go.Figure(go.Heatmap(z=hm.values, x=[f"{h:02d}" for h in hm.columns], y=DAYS,
                               colorscale="OrRd"))
    st.plotly_chart(style_fig(fig, height=300, legend=False).update_layout(
        title="When flagged transactions happen (day × hour)"), width="stretch")
