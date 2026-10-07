"""Live Transaction Monitor: replay the stream as if it were arriving now, plus a
Scenario Lab that scores brand-new transactions in real time."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fraud.risk_engine import BAND_COLORS, formula_text
from ui import nav, state
from ui.components import (STATUS_ICON, alert, badge, checklist, esc, hero, kpi_row, money,
                           score_card, style_fig)

CITIES = ["Chennai", "Coimbatore", "Bengaluru", "Mumbai", "New Delhi", "Hyderabad", "Kochi",
          "Pune", "Dubai", "Singapore", "London", "Bangkok"]
COUNTRY = {"Dubai": "AE", "Singapore": "SG", "London": "GB", "Bangkok": "TH"}


def _start_points(t, rings):
    pts = {"Start of data": t["timestamp"].min()}
    for r in rings[:4]:
        if r.get("peak_window", {}).get("start"):
            pts[f"⚡ {r['ring_id']} attack ({r['size']} accounts)"] = \
                pd.Timestamp(r["peak_window"]["start"]) - pd.Timedelta(minutes=20)
        elif r.get("first_seen"):
            pts[f"⚡ {r['ring_id']} first activity"] = pd.Timestamp(r["first_seen"]) - \
                pd.Timedelta(minutes=20)
    crit = t[t["band"] == "CRITICAL"]
    if len(crit):
        pts["Latest critical alert"] = crit["timestamp"].max() - pd.Timedelta(minutes=30)
    return pts


def _feed_frame(rows):
    if rows.empty:
        return rows
    return pd.DataFrame({
        "Transaction": rows["txn_id"], "Time": rows["timestamp"].dt.strftime("%d %b %H:%M:%S"),
        "Account": rows["account_id"], "Amount (₹)": rows["amount_inr"].round(0),
        "Merchant / Beneficiary": rows["merchant_name"].where(rows["merchant_name"] != "",
                                                              "→ " + rows["payee_account_id"]),
        "Device": rows["device_id"].replace("", "card-present"),
        "Risk": rows["final_risk"].round(0), "Status": rows["status"].map(STATUS_ICON),
    })


def render():
    t, rings = state.txn(), state.rings()
    hero("Live Transaction Monitor",
         "Historical stream replayed as live · each event is scored only from what came "
         "before it · alerts fire the moment risk crosses a band")
    tab_stream, tab_lab = st.tabs(["📡 Live stream", "🧪 Scenario Lab — score a new transaction"])

    with tab_stream:
        pts = _start_points(t, rings)
        c1, c2, c3, c4 = st.columns([2.4, 1, 1, 1.2])
        start = c1.selectbox("Replay from", list(pts), index=1 if len(pts) > 1 else 0)
        per_tick = c2.select_slider("Events / tick", [1, 2, 5, 10, 20, 50], value=5)
        every = c3.select_slider("Tick (s)", [1.0, 1.5, 2.0, 3.0], value=1.5)
        playing = c4.toggle("▶ Stream live", value=False, help="Auto-refresh the stream")
        if st.session_state.get("live_start") != start:
            st.session_state["live_start"] = start
            st.session_state["live_cursor"] = int(t["timestamp"].searchsorted(pts[start]))
        b1, b2, _ = st.columns([1, 1, 4])
        if b1.button("Step ▸", width="stretch"):
            st.session_state["live_cursor"] = min(len(t), st.session_state["live_cursor"] + per_tick)
        if b2.button("Reset ⟲", width="stretch"):
            st.session_state["live_cursor"] = int(t["timestamp"].searchsorted(pts[start]))
        origin = int(t["timestamp"].searchsorted(pts[start]))

        @st.fragment(run_every=every if playing else None)
        def stream():
            if playing:
                st.session_state["live_cursor"] = min(len(t), st.session_state["live_cursor"] + per_tick)
            cur = st.session_state["live_cursor"]
            seen = t.iloc[origin:cur]
            injected = st.session_state.get("lab_events", [])
            kpi_row([
                ("Stream clock", f"{t['timestamp'].iloc[max(cur - 1, 0)]:%d %b %H:%M}", "replay time"),
                ("Processed", f"{len(seen):,}", money(seen["amount_inr"].sum())),
                ("Alerts", f"{int((seen['final_risk'] >= 61).sum())}", "HIGH + CRITICAL", "#f97316"),
                ("Blocked", f"{int((seen['band'] == 'CRITICAL').sum())}", "BLOCK + INVESTIGATE",
                 "#ef4444"),
                ("Step-up", f"{int((seen['band'] == 'HIGH').sum())}", "OTP / biometric"),
            ])
            recent_alerts = seen[seen["final_risk"] >= 61].tail(3).iloc[::-1]
            for r in recent_alerts.itertuples():
                kind = "crit" if r.band == "CRITICAL" else "high"
                target = r.merchant_name or f"→ {r.payee_account_id}"
                alert(f"{'🚨' if kind == 'crit' else '⚠️'} <b>{r.status}</b> · {r.txn_id} · "
                      f"{r.account_id} · ₹{r.amount_inr:,.0f} · {esc(target)} · risk "
                      f"<b>{r.final_risk:.0f}</b> {('· ring ' + r.ring_id) if r.ring_id else ''}",
                      kind)
            if seen.empty:
                st.info("Press **▶ Stream live** or **Step ▸** to start the stream.")
                return
            left, right = st.columns([3, 2])
            with left:
                feed = _feed_frame(seen.tail(25).iloc[::-1])
                st.dataframe(feed, hide_index=True, width="stretch", height=420,
                             column_config={"Risk": st.column_config.ProgressColumn(
                                 "Risk", min_value=0, max_value=100, format="%d"),
                                 "Amount (₹)": st.column_config.NumberColumn(format="₹%d")})
            with right:
                w = seen.tail(300)
                fig = go.Figure(go.Scatter(
                    x=w["timestamp"], y=w["final_risk"], mode="markers",
                    marker=dict(color=[BAND_COLORS[b] for b in w["band"]],
                                size=[14 if b in ("HIGH", "CRITICAL") else 6 for b in w["band"]]),
                    hovertext=w["txn_id"] + " " + w["account_id"], hoverinfo="text"))
                for y, c in ((61, "#f97316"), (81, "#ef4444")):
                    fig.add_hline(y=y, line=dict(color=c, dash="dot", width=1))
                fig.update_yaxes(range=[-3, 103], title="final risk")
                st.plotly_chart(style_fig(fig, height=420, legend=False).update_layout(
                    title="Live risk"), width="stretch")
            if injected:
                st.caption(f"{len(injected)} transaction(s) injected from the Scenario Lab:")
                st.dataframe(pd.DataFrame(injected), hide_index=True, width="stretch")

        stream()
        flagged = t.iloc[origin:st.session_state["live_cursor"]]
        flagged = flagged[flagged["final_risk"] >= 61]
        if len(flagged):
            c1, c2 = st.columns([3, 1])
            pick = c1.selectbox("Investigate an alert", flagged["txn_id"].iloc[::-1],
                                format_func=lambda x: f"{x} · risk "
                                f"{flagged.set_index('txn_id').loc[x, 'final_risk']:.0f}")
            with c2:
                st.write("")
                nav.button("Open investigation →", "transaction", "live_go", sel_txn=pick)

    with tab_lab:
        lab()


def _hni_account(t):
    tr = t[(t["txn_type"] == "transfer") & (t["amount_inr"] >= 200000) & (t["final_risk"] < 61)]
    per = tr.groupby("account_id").size()
    acc = state.acc().set_index("account_id")
    per = per[[acc.loc[a, "account_type"] == "personal" if a in acc.index else False
               for a in per.index]]
    return per.idxmax() if len(per) else None


def _small_account(t):
    acc = state.acc()
    cand = acc[(acc["account_type"] == "personal") & (acc["account_risk"] < 31) &
               (acc["avg_amount"] < 2000) & (acc["n_transactions"] > 40)]
    return cand["account_id"].iloc[0] if len(cand) else acc["account_id"].iloc[-1]


def lab():
    t, a, rings = state.txn(), state.acc(), state.rings()
    st.markdown("Score a **new** transaction in real time against the full history "
                "(real-time rules, behavioural profile, anomaly model, ML model and network "
                "context). Nothing is written to the history.")
    hni, small = _hni_account(t), _small_account(t)
    mule = rings[0]["mules"][0] if rings and rings[0]["mules"] else None
    c1, c2, c3 = st.columns(3)
    if c1.button("✅ Legit ₹5,00,000 (premium customer)", width="stretch", disabled=hni is None):
        h = t[t["account_id"] == hni]
        payee = h[h["amount_inr"] >= 200000]["payee_account_id"].mode().iloc[0]
        dev = h[h["device_id"] != ""]["device_id"].mode().iloc[0]
        st.session_state["lab"] = dict(account=hni, kind="transfer", payee=payee, amount=500000.0,
                                       device=dev, city=a.set_index("account_id").loc[hni, "home_city"],
                                       hour=14, minute=10)
    if c2.button("🚨 ₹48,000 account takeover", width="stretch"):
        st.session_state["lab"] = dict(account=small, kind="transfer", payee=mule or "NEW-BEN-01",
                                       amount=48000.0, device="D-9981 (new)", city="New Delhi",
                                       hour=2, minute=37)
    if c3.button("🧾 Normal ₹1,200 grocery", width="stretch"):
        h = t[t["account_id"] == small]
        st.session_state["lab"] = dict(account=small, kind="purchase", merchant="FreshBasket",
                                       amount=1200.0, device=(h[h["device_id"] != ""]["device_id"]
                                                              .mode().iloc[0] if (h["device_id"] != "").any() else ""),
                                       city=a.set_index("account_id").loc[small, "home_city"],
                                       hour=18, minute=5)
    preset = st.session_state.get("lab", dict(account=small, kind="purchase", merchant="FreshBasket",
                                              amount=1200.0, device="", city="Chennai", hour=18,
                                              minute=5))
    merchants = state.data()["merchants"]
    ids = list(a["account_id"])
    with st.form("lab_form"):
        f1, f2, f3 = st.columns(3)
        acct = f1.selectbox("Account", ids, index=ids.index(preset["account"])
                            if preset["account"] in ids else 0)
        hist = t[t["account_id"] == acct]
        kind = f1.radio("Type", ["purchase", "transfer"], horizontal=True,
                        index=0 if preset["kind"] == "purchase" else 1)
        names = list(merchants["merchant_name"])
        merchant = f2.selectbox("Merchant (purchases)", names,
                                index=names.index(preset.get("merchant", names[0]))
                                if preset.get("merchant") in names else 0)
        payees = sorted(set(a["account_id"]) - {acct}) + ["NEW-BEN-01"]
        payee_default = preset.get("payee", payees[0])
        payee = f2.selectbox("Beneficiary (transfers)", payees,
                             index=payees.index(payee_default) if payee_default in payees else 0)
        amount = f2.number_input("Amount (₹)", min_value=1.0, value=float(preset["amount"]),
                                 step=1000.0)
        devices = sorted(d for d in hist["device_id"].unique() if d)
        dev_opts = devices + ["D-9981 (new)"]
        device = f3.selectbox("Device", dev_opts, index=dev_opts.index(preset["device"])
                              if preset["device"] in dev_opts else 0)
        city = f3.selectbox("Location", CITIES, index=CITIES.index(preset["city"])
                            if preset["city"] in CITIES else 0)
        hh = f3.slider("Hour", 0, 23, int(preset["hour"]))
        go_btn = st.form_submit_button("⚡ Score in real time", type="primary")
    if hist.shape[0]:
        prof = a.set_index("account_id").loc[acct]
        st.caption(f"{acct} normally: avg ₹{prof.get('avg_amount', 0):,.0f} "
                   f"(₹{prof.get('amount_p10', 0):,.0f}–₹{prof.get('amount_p90', 0):,.0f}), "
                   f"hours {prof.get('typical_hours', '')}, {prof.get('typical_locations', '')}, "
                   f"devices {prof.get('typical_devices', '') or 'card-present only'}, "
                   f"{prof.get('avg_daily_txns', 0):.1f} txns/day.")
    if not go_btn:
        return
    mid = merchants.set_index("merchant_name")["merchant_id"].get(merchant, "") if kind == "purchase" else ""
    when = t["timestamp"].max().normalize() + pd.Timedelta(days=1, hours=hh,
                                                           minutes=preset.get("minute", 0))
    ip = hist.loc[hist["ip_address"] != "", "ip_address"]
    event = {"txn_id": f"LIVE-{len(st.session_state.get('lab_events', [])) + 1:03d}",
             "timestamp": when, "account_id": acct, "merchant_id": mid,
             "payee_account_id": payee if kind == "transfer" else "", "amount_inr": amount,
             "device_id": "D-9981" if device.startswith("D-9981") else device,
             "ip_address": ip.iloc[-1] if len(ip) and city not in COUNTRY else "203.0.113.7",
             "city": city, "country": COUNTRY.get(city, "IN"),
             "channel": "upi" if kind == "transfer" and amount <= 100000 else
             ("netbanking" if kind == "transfer" else "app")}
    try:
        scorer = state.live_scorer()
        row = scorer.score(event, state.weights())
    except Exception as exc:
        st.error(f"Live scoring failed: {exc}")
        return
    st.session_state.setdefault("lab_events", []).append(
        {"Transaction": row["txn_id"], "Account": acct, "Amount (₹)": round(amount),
         "Risk": row["final_risk"], "Status": STATUS_ICON[row["status"]]})
    show_result(row)


def show_result(row):
    st.divider()
    kind = {"CRITICAL": "crit", "HIGH": "high", "MEDIUM": "info", "LOW": "ok"}[row["band"]]
    alert(f"<b>{row['txn_id']}</b> · ₹{row['amount_inr']:,.0f} · {row['account_id']} → "
          f"{esc(row['payee_account_id'] or row.get('merchant_category') or '')} · "
          f"<b>{row['action']}</b> {badge(row['band'])}", kind)
    c = st.columns(4)
    score_card(c[0], "Transaction risk", row["transaction_risk"], "rules · anomaly · ML")
    score_card(c[1], "Behavioral risk", row["behavioral_risk"], "vs account profile")
    score_card(c[2], "Network / ring risk", row["network_risk"], "graph evidence")
    score_card(c[3], "Final risk", row["final_risk"], row["action"])
    lines, _ = formula_text(row["transaction_risk"], row["behavioral_risk"], row["network_risk"],
                            row["network_evidence"], state.weights())
    st.code("\n".join(lines), language=None)
    left, right = st.columns([1, 1])
    with left:
        st.markdown("##### WHY? Evidence checklist")
        checklist(row["checklist"])
    with right:
        st.markdown("##### Explanation")
        st.markdown(f"<div class='fi-card'>{esc(row['explanation'])}</div>", unsafe_allow_html=True)
        st.markdown("##### Reasons and weights")
        for r in sorted(row["reasons"], key=lambda r: -r["weight"]):
            st.markdown(f"- **{r['code']}** `{r.get('layer', 'rule')}` · w={r['weight']:.2f} — "
                        f"{r['evidence']}")
        if not row["reasons"]:
            st.success("No risk factors — consistent with this account's normal behaviour.")
