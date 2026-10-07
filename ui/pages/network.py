"""Network Graph: interactive heterogeneous entity graph with selection, filtering and
suspicious-cluster highlighting."""

import pandas as pd
import streamlit as st

from fraud.graph_analysis import (NODE_STYLE, communities, neighbourhood, nid, risk_propagation,
                                  suspicious_view)
from ui import nav, state
from ui.components import alert, badge, chips, esc, hero, kpi_row, network_figure


@st.cache_data(show_spinner=False)
def _analytics(key, wkey):
    G, P = state.graphs()
    risk = state.acc().set_index("account_id")["account_risk"].to_dict()
    return communities(P, risk), risk_propagation(P, risk)


def render():
    G, P = state.graphs()
    a = state.acc()
    rings = state.rings()
    hero("Network Graph",
         "Accounts · customers · devices · merchants · beneficiaries · locations · flagged "
         "transactions — click a node to inspect it")
    comms, prop = _analytics(st.session_state.get("upload_key", "default"),
                             tuple(sorted(state.weights().items())))

    c1, c2, c3, c4 = st.columns([1.4, 1.6, 1, 1])
    modes = ["Suspicious network (rings + riskiest accounts)", "Neighbourhood of an entity"] + \
        [f"Ring {r['ring_id']}" for r in rings]
    focus_node = st.session_state.get("sel_node")
    mode = c1.selectbox("View", modes, index=1 if focus_node else 0)
    types = c2.multiselect("Node types", list(NODE_STYLE),
                           default=["account", "device", "beneficiary", "merchant", "transaction"],
                           format_func=lambda x: NODE_STYLE[x][2])
    min_risk = c3.slider("Min. account risk", 0, 100, 0, 5)
    show_comm = c4.toggle("Highlight suspicious clusters", value=True)

    if mode.startswith("Neighbourhood"):
        accounts = list(a["account_id"])
        default = focus_node.split(":", 1)[1] if focus_node and focus_node.startswith("ACC:") else accounts[0]
        cc1, cc2 = st.columns([2, 1])
        center_acc = cc1.selectbox("Centre account", accounts,
                                   index=accounts.index(default) if default in accounts else 0)
        radius = cc2.slider("Hops", 1, 3, 2)
        center = focus_node if focus_node and focus_node in G and not focus_node.startswith("ACC:") \
            else nid("account", center_acc)
        H = neighbourhood(G, center, radius=radius, max_nodes=160, types=tuple(types))
        highlight = [center]
    elif mode.startswith("Ring "):
        ring = state.ring(mode.split(" ", 1)[1])
        keep = {nid("account", m) for m in ring["members"]}
        for n in list(keep):
            if n in G:
                keep |= {m for m in G.neighbors(n) if G.nodes[m]["type"] in types and
                         (G.nodes[m]["type"] not in ("merchant", "location") or
                          G.nodes[m].get("risk", 0) >= 61)}
        H = G.subgraph([n for n in keep if n in G]).copy()
        highlight = [nid("account", m) for m in ring["mules"]]
    else:
        H = suspicious_view(G, rings, a, top_n=20, types=tuple(types))
        highlight = []
    drop = [n for n, d in H.nodes(data=True)
            if d["type"] not in types or (d["type"] == "account" and d.get("risk", 0) < min_risk
                                          and n not in highlight)]
    H = H.copy()
    H.remove_nodes_from(drop)
    sus = [c["accounts"] for c in comms if c["suspicious"]] if show_comm else []
    if show_comm:
        members = {nid("account", x) for c in sus for x in c}
        highlight = list(set(highlight) | (members & set(H.nodes)))

    left, right = st.columns([2.3, 1])
    with left:
        fig, _ = network_figure(H, height=640, highlight=highlight,
                                communities=sus if show_comm else None)
        event = st.plotly_chart(fig, width="stretch", key="net_graph", on_select="rerun",
                                selection_mode="points", config={"scrollZoom": True})
        st.caption(f"{H.number_of_nodes()} nodes · {H.number_of_edges()} edges · scroll to zoom, "
                   f"drag to pan, click a node to inspect it. White outline = highlighted "
                   f"(selected / suspicious cluster); red outline = CRITICAL entity.")
    from ui.components import selected_node
    picked = selected_node(event) or (focus_node if focus_node in H else None)
    with right:
        details(G, picked, a, rings, prop)

    st.markdown("### Suspicious clusters (Louvain communities)")
    if comms:
        top = [c for c in comms if c["suspicious"]] or comms[:5]
        st.dataframe(pd.DataFrame({
            "Cluster": [f"C{i + 1}" for i in range(len(top))],
            "Accounts": [c["size"] for c in top],
            "Mean risk": [round(c["mean_risk"], 1) for c in top],
            "Max risk": [round(c["max_risk"], 1) for c in top],
            "Suspicious": ["⚠️ yes" if c["suspicious"] else "" for c in top],
            "Members": [", ".join(c["accounts"][:12]) for c in top]}),
            hide_index=True, width="stretch")
        st.caption("Communities found on the account projection (accounts linked through a "
                   "shared device, a shared personal beneficiary, or a transfer). Business "
                   "beneficiaries such as landlords are excluded so rent day is not a 'cluster'.")
    else:
        st.info("No account clusters found.")


def details(G, node, a, rings, prop):
    if not node or node not in G:
        st.markdown("<div class='fi-card'><b>Select a node</b><br><span class='fi-muted'>Click "
                    "any account, device, beneficiary or merchant in the graph.</span></div>",
                    unsafe_allow_html=True)
        return
    d = G.nodes[node]
    typ, label = d["type"], d["label"]
    if typ == "account":
        row = state.account_row(label)
        nb = list(G.neighbors(node))

        def of(kind):
            return [G.nodes[n]["label"] for n in nb if G.nodes[n]["type"] == kind]
        connected = [G.nodes[n]["label"] for n in nb if G.nodes[n]["type"] == "account"]
        t = state.txn()
        sus = t[(t["account_id"] == label) & (t["final_risk"] >= 61)]
        st.markdown(f"<div class='fi-card'><b style='font-size:1.1rem'>Account {label}</b> "
                    f"{badge(row['band']) if row is not None else ''}<br>"
                    f"Account risk <b>{row['account_risk'] if row is not None else 0:.0f}</b>/100 · "
                    f"risk propagation {prop.get(label, 0):.0f}</div>", unsafe_allow_html=True)
        st.markdown(f"**Connected accounts** {chips(connected[:10]) or '—'}", unsafe_allow_html=True)
        st.markdown(f"**Devices** {chips(of('device')[:8]) or '—'}", unsafe_allow_html=True)
        st.markdown(f"**Merchants** {chips(of('merchant')[:8]) or '—'}", unsafe_allow_html=True)
        st.markdown(f"**Beneficiaries** {chips(of('beneficiary')[:8]) or '—'}", unsafe_allow_html=True)
        st.markdown(f"**Locations** {chips(of('location')[:6]) or '—'}", unsafe_allow_html=True)
        st.markdown(f"**Suspicious transactions** {len(sus)}"
                    + (f": {chips(list(sus['txn_id'].head(6)))}" if len(sus) else ""),
                    unsafe_allow_html=True)
        ring = d.get("ring")
        st.markdown(f"**Fraud ring membership** {chips([ring], '#ef4444') if ring else 'none'}",
                    unsafe_allow_html=True)
        nav.button("Open account intelligence →", "account", "net_acc", sel_account=label)
        if ring:
            nav.button(f"Open {ring} →", "rings", "net_ring", sel_ring=ring)
        if len(sus):
            nav.button("Investigate top transaction →", "transaction", "net_tx",
                       sel_txn=sus.sort_values("final_risk").iloc[-1]["txn_id"])
    else:
        accts = [G.nodes[n]["label"] for n in G.neighbors(node) if G.nodes[n]["type"] == "account"]
        risk = state.acc().set_index("account_id")["account_risk"]
        risky = [x for x in accts if risk.get(x, 0) >= 61]
        st.markdown(f"<div class='fi-card'><b style='font-size:1.1rem'>{NODE_STYLE[typ][2]} "
                    f"{esc(label)}</b><br>max transaction risk {d.get('risk', 0):.0f} · "
                    f"{len(accts)} connected account(s), {len(risky)} high-risk</div>",
                    unsafe_allow_html=True)
        if len(accts) > 1:
            st.markdown("**Accounts linked through this entity** " +
                        chips([f"{x} · {risk.get(x, 0):.0f}" for x in sorted(
                            accts, key=lambda x: -risk.get(x, 0))[:15]]), unsafe_allow_html=True)
        if typ in ("device", "beneficiary") and len(risky) >= 2:
            alert(f"⚠️ {len(risky)} high-risk accounts share this {typ}.", "high")
        if typ == "transaction":
            nav.button("Investigate transaction →", "transaction", "net_tx2", sel_txn=label)
