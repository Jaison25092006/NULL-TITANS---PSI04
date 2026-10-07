"""Heterogeneous entity graph and graph analytics.

Nodes   Account · Customer · Device · Merchant · Beneficiary · Location · Transaction
Edges   ACCOUNT -USES-> DEVICE          ACCOUNT -TRANSACTS_WITH-> MERCHANT
        ACCOUNT -SENDS_TO-> BENEFICIARY ACCOUNT -LOCATED_AT-> LOCATION
        ACCOUNT -MAKES-> TRANSACTION    ACCOUNT -CONNECTED_TO-> ACCOUNT (ring evidence)
        ACCOUNT -OWNED_BY-> CUSTOMER    BENEFICIARY -IS_ACCOUNT-> ACCOUNT

Analytics: account projection (accounts linked through devices, beneficiaries and
transfers), neighbour risk, community detection (Louvain) and risk propagation
(personalised PageRank seeded with account risk).
"""

from collections import defaultdict
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd

NODE_STYLE = {  # type: (colour, symbol, label)
    "account": ("#60a5fa", "circle", "Account"),
    "customer": ("#94a3b8", "circle-open", "Customer"),
    "device": ("#a78bfa", "square", "Device"),
    "merchant": ("#34d399", "diamond", "Merchant"),
    "beneficiary": ("#f472b6", "star", "Beneficiary"),
    "location": ("#fbbf24", "triangle-up", "Location"),
    "transaction": ("#f87171", "x", "Transaction"),
}
RELATION_COLOR = {"USES": "#a78bfa", "TRANSACTS_WITH": "#34d399", "SENDS_TO": "#f472b6",
                  "LOCATED_AT": "#fbbf24", "MAKES": "#f87171", "CONNECTED_TO": "#ef4444",
                  "OWNED_BY": "#64748b", "IS_ACCOUNT": "#f472b6"}


def nid(kind, value):
    return f"{kind[:3].upper()}:{value}"


def build_entity_graph(txn, acc, merchants, rings, flag_threshold=61):
    G = nx.Graph()
    risk = acc.set_index("account_id")["account_risk"].to_dict() if len(acc) else {}
    ring_of = {m: r["ring_id"] for r in rings for m in r["members"]}
    kind = acc.set_index("account_id")["account_type"].to_dict() if len(acc) else {}
    customer = acc.set_index("account_id")["customer_id"].to_dict() if len(acc) else {}
    mname = merchants.set_index("merchant_id")["merchant_name"].to_dict()
    loc = np.where(txn["country"] == "IN", txn["city"], txn["city"] + ", " + txn["country"])
    t = txn.assign(location=loc)
    # per-entity max transaction risk
    for a in t["account_id"].unique():
        G.add_node(nid("account", a), type="account", label=a, risk=float(risk.get(a, 0)),
                   ring=ring_of.get(a, ""), account_type=kind.get(a, "personal"))
        if customer.get(a):
            c = nid("customer", customer[a])
            G.add_node(c, type="customer", label=customer[a], risk=float(risk.get(a, 0)))
            G.add_edge(nid("account", a), c, relation="OWNED_BY", count=1)

    def hub(kind_, key, label, frame):
        r = frame.groupby(key)["final_risk"].max()
        n = frame.groupby(key)["account_id"].nunique()
        for k in r.index:
            if k != "":
                G.add_node(nid(kind_, k), type=kind_, label=label(k), risk=float(r[k]),
                           n_accounts=int(n[k]))

    hub("device", "device_id", str, t[t["device_id"] != ""])
    hub("merchant", "merchant_id", lambda m: mname.get(m, m), t[t["merchant_id"] != ""])
    hub("beneficiary", "payee_account_id", str, t[t["payee_account_id"] != ""])
    hub("location", "location", str, t)

    for kind_, key, rel in (("device", "device_id", "USES"),
                            ("merchant", "merchant_id", "TRANSACTS_WITH"),
                            ("beneficiary", "payee_account_id", "SENDS_TO"),
                            ("location", "location", "LOCATED_AT")):
        sub = t[t[key] != ""]
        agg = sub.groupby(["account_id", key]).agg(count=("txn_id", "size"),
                                                   amount=("amount_inr", "sum"),
                                                   risk=("final_risk", "max")).reset_index()
        for r in agg.itertuples(index=False):
            G.add_edge(nid("account", r.account_id), nid(kind_, getattr(r, key)), relation=rel,
                       count=int(r.count), amount=float(r.amount), risk=float(r.risk))
    for b in set(t["payee_account_id"]) - {""}:
        if G.has_node(nid("account", b)):
            G.add_edge(nid("beneficiary", b), nid("account", b), relation="IS_ACCOUNT", count=1)
    for r in rings:
        for e in r["edges"]:
            G.add_edge(nid("account", e["a"]), nid("account", e["b"]), relation="CONNECTED_TO",
                       kinds=e["kinds"], count=len(e["evidence"]), ring=r["ring_id"])
    flagged = t[t["final_risk"] >= flag_threshold]
    for r in flagged.itertuples(index=False):
        n = nid("transaction", r.txn_id)
        G.add_node(n, type="transaction", label=r.txn_id, risk=float(r.final_risk),
                   amount=float(r.amount_inr))
        G.add_edge(nid("account", r.account_id), n, relation="MAKES", count=1)
    return G


def neighbourhood(G, center, radius=2, max_nodes=160, hub_degree=40,
                  types=("account", "device", "merchant", "beneficiary", "location",
                         "transaction", "customer")):
    """BFS around `center`; high-degree hubs (popular merchants, big cities) are shown but
    not expanded, so the view stays readable."""
    if center not in G:
        return nx.Graph()
    keep, frontier = {center}, [center]
    for _ in range(radius):
        nxt = []
        for n in frontier:
            if n != center and G.degree(n) > hub_degree:
                continue
            nbrs = sorted(G.neighbors(n), key=lambda m: -G.nodes[m].get("risk", 0))
            for m in nbrs:
                if G.nodes[m]["type"] in types and m not in keep:
                    keep.add(m)
                    nxt.append(m)
                    if len(keep) >= max_nodes:
                        return G.subgraph(keep).copy()
        frontier = nxt
    return G.subgraph(keep).copy()


def suspicious_view(G, rings, acc, top_n=25, types=("account", "device", "beneficiary",
                                                    "merchant", "location", "transaction")):
    """Ring members + the riskiest accounts with their non-hub entities."""
    seeds = {nid("account", m) for r in rings for m in r["members"]}
    seeds |= {nid("account", a) for a in acc.nlargest(top_n, "account_risk")["account_id"]}
    keep = set(s for s in seeds if s in G)
    for s in list(keep):
        for m in G.neighbors(s):
            t = G.nodes[m]["type"]
            if t not in types:
                continue
            if t in ("merchant", "location") and G.nodes[m].get("risk", 0) < 61:
                continue  # popular, low-risk hubs would connect everyone
            keep.add(m)
    return G.subgraph(keep).copy()


# --------------------------------------------------------------------------- #
# Account projection and analytics
# --------------------------------------------------------------------------- #

def account_projection(txn, accounts):
    """Accounts linked by a shared device, a shared personal beneficiary or a transfer."""
    P = nx.Graph()
    P.add_nodes_from(txn["account_id"].unique())
    kind = accounts.set_index("account_id")["account_type"].to_dict()
    for col, rel, cap in (("device_id", "shared_device", 10),
                          ("payee_account_id", "shared_beneficiary", 30)):
        sub = txn[txn[col] != ""]
        for value, accts in sub.groupby(col)["account_id"].unique().items():
            if col == "payee_account_id" and kind.get(value) == "business":
                continue  # landlords / suppliers collect from many people legitimately
            if 2 <= len(accts) <= cap:
                for a, b in combinations(sorted(accts), 2):
                    P.add_edge(a, b, relation=rel, via=value)
    tr = txn[(txn["payee_account_id"] != "")]
    for r in tr[["account_id", "payee_account_id"]].drop_duplicates().itertuples(index=False):
        if r.payee_account_id in P and kind.get(r.payee_account_id) != "business" \
                and r.account_id != r.payee_account_id:
            P.add_edge(r.account_id, r.payee_account_id, relation="transfer",
                       via=r.payee_account_id)
    return P


def neighbour_risk(P, seed_risk):
    """For each account: (max seed risk among neighbours, that neighbour, relation)."""
    out = {}
    for a in P.nodes:
        best = (0.0, "", "")
        for b in P.neighbors(a):
            r = seed_risk.get(b, 0.0)
            if r > best[0]:
                best = (r, b, P[a][b].get("relation", ""))
        out[a] = best
    return out


def communities(P, risk, min_size=3):
    """Louvain communities on the account projection, with mean / max risk."""
    if P.number_of_edges() == 0:
        return []
    H = P.subgraph([n for n in P.nodes if P.degree(n) > 0])
    try:
        comms = nx.community.louvain_communities(H, seed=7)
    except Exception:
        comms = list(nx.connected_components(H))
    out = []
    for c in comms:
        if len(c) < min_size:
            continue
        rs = [risk.get(a, 0) for a in c]
        out.append({"accounts": sorted(c), "size": len(c), "mean_risk": float(np.mean(rs)),
                    "max_risk": float(np.max(rs)),
                    "suspicious": bool(np.mean(rs) >= 50 or sum(r >= 81 for r in rs) >= 3)})
    return sorted(out, key=lambda x: -x["mean_risk"])


def risk_propagation(P, risk):
    """Personalised PageRank seeded with account risk: how much risk 'flows' to each
    account through its connections (0-100, relative)."""
    if P.number_of_edges() == 0:
        return {}
    pers = {a: (risk.get(a, 0) / 100) ** 2 for a in P.nodes}
    if sum(pers.values()) == 0:
        return {}
    pr = nx.pagerank(P, alpha=0.85, personalization=pers, max_iter=200)
    top = max(pr.values()) or 1
    return {a: round(100 * v / top, 1) for a, v in pr.items()}


def degree_table(G, types=("device", "beneficiary", "merchant")):
    rows = defaultdict(list)
    for n, d in G.nodes(data=True):
        if d["type"] in types:
            rows["node"].append(n)
            rows["type"].append(d["type"])
            rows["label"].append(d["label"])
            rows["accounts"].append(d.get("n_accounts", G.degree(n)))
            rows["max_risk"].append(d.get("risk", 0))
    return pd.DataFrame(rows)
