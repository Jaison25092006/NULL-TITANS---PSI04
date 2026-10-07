"""Network analysis: link accounts through shared evidence and extract fraud rings.

Edges between accounts (each carries the evidence that created it):
  shared_device       a device used by 3+ accounts that were new when they used it
  linked_device       a device that is *new* to 3+ established accounts within 7 days
                      (a fraud controller's phone; household phones are each member's
                      usual device, so they never qualify)
  shared_subnet       an IP /24 range used by 4+ new accounts
  same_sequence       the same ordered trio of merchants within 72 hours (2+ rarely used,
                      none an everyday merchant), repeated by 4+ accounts (no single
                      purchase looks bad)
  mule_payee          senders linked to an account that receives from 3+ accounts and
                      passes the money straight through to gift cards / crypto
  collusive_merchant  customers of a newly onboarded merchant whose sales are dominated by
                      round amounts AND come in coordinated bursts

Households (old accounts sharing a phone) and landlords (many tenants, but the money
stays put) do not create edges.
"""

from collections import defaultdict
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd

from .stream import RISKY_CATEGORIES, YOUNG_DAYS

EDGE_WEIGHT = {"shared_device": 0.5, "linked_device": 0.45, "shared_subnet": 0.3,
               "same_sequence": 0.6, "mule_payee": 0.6, "collusive_merchant": 0.55}
EDGE_LABEL = {"shared_device": "Shared device (new accounts)",
              "linked_device": "Linked controller device",
              "shared_subnet": "Shared IP range", "same_sequence": "Same purchase sequence",
              "mule_payee": "Common beneficiary (mule)",
              "collusive_merchant": "Collusive merchant"}
RARE_MERCHANT_SHARE = 0.10      # merchant used by fewer than 10% of accounts
EVERYDAY_MERCHANT_SHARE = 0.50  # groceries, fuel, ...: never part of a sequence signature
SEQUENCE_WINDOW_H = 72
MIN_SEQUENCE_ACCOUNTS = 4


def _add(g, a, b, kind, detail, txns):
    if a == b:
        return
    if not g.has_edge(a, b):
        g.add_edge(a, b, evidence=[])
    g[a][b]["evidence"].append({"kind": kind, "detail": detail})
    for n in (a, b):
        g.nodes[n].setdefault("evidence_txns", set()).update(txns.get(n, ()))


def _is_round(amount):
    cents = round(amount % 1000, 2)
    return cents in (0, 999) or round(amount % 500, 2) == 0 or round(amount % 100, 2) == 99


def build_graph(tx, accounts, merchants):
    g = nx.Graph()
    g.add_nodes_from(accounts["account_id"])
    signup = pd.to_datetime(accounts.set_index("account_id")["signup_date"])
    tx = tx.assign(age=(tx["timestamp"] - tx["account_id"].map(signup)).dt.days.fillna(365))
    category = merchants.set_index("merchant_id")["category"]
    name = merchants.set_index("merchant_id")["merchant_name"]

    # 1. Shared devices / subnets among accounts that were new when they used them.
    young = tx[(tx["age"] < YOUNG_DAYS) & (tx["device_id"] != "")]
    for kind, col, key, min_n in (("shared_device", "device_id", lambda v: v, 3),
                                  ("shared_subnet", "ip_address", lambda v: v.rsplit(".", 1)[0], 4)):
        groups = defaultdict(lambda: defaultdict(list))
        for t in young.itertuples(index=False):
            if getattr(t, col):
                groups[key(getattr(t, col))][t.account_id].append(t.txn_id)
        for value, accts in groups.items():
            if len(accts) >= min_n:
                label = f"{value}.x" if kind == "shared_subnet" else value
                for a, b in combinations(sorted(accts), 2):
                    _add(g, a, b, kind, f"{kind.replace('_', ' ')} {label} "
                                        f"({len(accts)} new accounts)", accts)

    # 1b. Linked devices: new to several established accounts at about the same time.
    digital = tx[tx["device_id"] != ""]
    first_use = digital.groupby(["account_id", "device_id"])["timestamp"].min().reset_index()
    prior = digital.groupby("account_id")["timestamp"].apply(lambda s: s.sort_values().to_numpy())
    newcomers = defaultdict(dict)  # device -> account -> first use
    for r in first_use.itertuples(index=False):
        times = prior[r.account_id]
        n_before = np.searchsorted(times, np.datetime64(r.timestamp))
        if n_before >= 3:  # account was established on other devices first
            newcomers[r.device_id][r.account_id] = r.timestamp
    for device, accts in newcomers.items():
        if len(accts) < 3:
            continue
        times = sorted(accts.values())
        if (times[-1] - times[0]) > pd.Timedelta(days=7):
            continue
        txns = {a: list(digital[(digital["account_id"] == a) & (digital["device_id"] == device)]
                        ["txn_id"]) for a in accts}
        for a, b in combinations(sorted(accts), 2):
            _add(g, a, b, "linked_device", f"device {device} newly appeared on {len(accts)} "
                                           f"established accounts within "
                                           f"{(times[-1] - times[0]).total_seconds() / 3600:.1f}h",
                 txns)

    # 2. Same ordered trio of rare merchants within 72h, repeated across accounts.
    purchases = tx[tx["merchant_id"] != ""]
    n_accounts = max(tx["account_id"].nunique(), 1)
    share = purchases.groupby("merchant_id")["account_id"].nunique() / n_accounts
    rare = set(share[share < RARE_MERCHANT_SHARE].index)
    everyday = set(share[share >= EVERYDAY_MERCHANT_SHARE].index)
    seqs = defaultdict(dict)  # (m1, m2, m3) -> account -> txn ids
    for acct, p in purchases.groupby("account_id"):
        rows = list(p[["timestamp", "merchant_id", "txn_id"]].itertuples(index=False))
        for i, (t1, m1, x1) in enumerate(rows):
            window = [r for r in rows[i + 1:] if (r[0] - t1).total_seconds() <= SEQUENCE_WINDOW_H * 3600]
            for (t2, m2, x2), (t3, m3, x3) in combinations(window, 2):
                trio = (m1, m2, m3)
                if len(set(trio)) == 3 and sum(m in rare for m in trio) >= 2                         and not any(m in everyday for m in trio):
                    seqs[trio].setdefault(acct, [x1, x2, x3])
    for trio, accts in seqs.items():
        if len(accts) >= MIN_SEQUENCE_ACCOUNTS:
            label = " → ".join(name.get(m, m) for m in trio)
            for a, b in combinations(sorted(accts), 2):
                _add(g, a, b, "same_sequence", f"same purchase sequence {label} "
                                               f"({len(accts)} accounts)", accts)

    # 3. Mule payees: collect from 3+ senders and pass most of it to risky merchants.
    transfers = tx[tx["payee_account_id"] != ""]
    risky = purchases[purchases["merchant_id"].map(category).isin(RISKY_CATEGORIES)]
    mules = {}
    for payee, inc in transfers.groupby("payee_account_id"):
        if inc["account_id"].nunique() < 3:
            continue
        out = risky[risky["account_id"] == payee]
        passed, delays = 0.0, []
        for t in inc.itertuples(index=False):
            after = out[(out["timestamp"] >= t.timestamp) &
                        (out["timestamp"] <= t.timestamp + pd.Timedelta(hours=24))]
            passed += min(after["amount_inr"].sum(), t.amount_inr)
            if len(after):
                delays.append((after["timestamp"].min() - t.timestamp).total_seconds() / 3600)
        ratio = passed / max(inc["amount_inr"].sum(), 1e-9)
        if ratio >= 0.6:
            mules[payee] = {"senders": int(inc["account_id"].nunique()),
                            "received": float(inc["amount_inr"].sum()), "pass_through": ratio,
                            "median_delay_h": float(np.median(delays)) if delays else None}
            txns = {payee: list(out["txn_id"])}
            for sender, s in inc.groupby("account_id"):
                txns[sender] = list(s["txn_id"])
                _add(g, sender, payee, "mule_payee",
                     f"sent ₹{s['amount_inr'].sum():,.0f} to {payee}, which passed "
                     f"{ratio:.0%} of its inflows to gift cards/crypto within 24h", txns)

    # 4. Collusive merchants: newly onboarded, many accounts, round amounts or bursts.
    collusive = {}
    start = tx["timestamp"].min()
    for merchant, p in purchases.groupby("merchant_id"):
        accts = p["account_id"].unique()
        if len(accts) < 4 or (p["timestamp"].min() - start) < pd.Timedelta(days=7):
            continue
        round_share = float(p["amount_inr"].map(_is_round).mean())
        ts = p["timestamp"].sort_values()
        burst = 0
        for i, t0 in enumerate(ts):
            window = p[(p["timestamp"] >= t0) & (p["timestamp"] <= t0 + pd.Timedelta(minutes=30))]
            burst = max(burst, window["account_id"].nunique())
        if round_share >= 0.6 and burst >= 3:
            collusive[merchant] = {"accounts": len(accts), "round_share": round_share,
                                   "burst": burst, "first_seen": str(p["timestamp"].min())}
            txns = {a: list(p.loc[p["account_id"] == a, "txn_id"]) for a in accts}
            detail = (f"customer of {name.get(merchant, merchant)}, onboarded "
                      f"{(p['timestamp'].min() - start).days} days into the period: "
                      f"{round_share:.0%} round-amount sales, up to {burst} accounts buying "
                      f"within 30 minutes")
            for a, b in combinations(sorted(accts), 2):
                _add(g, a, b, "collusive_merchant", detail, txns)
    g.graph["collusive_merchants"] = collusive
    return g, mules


def describe_pattern(kinds, size):
    """Name the fraud pattern from the kinds of evidence that link the ring."""
    parts = []
    if "same_sequence" in kinds:
        parts.append("Coordinated purchase ring: accounts in different places, on different "
                     "devices, repeat the same unusual purchase sequence")
    if "shared_device" in kinds or "shared_subnet" in kinds:
        parts.append("Device farm: newly opened (likely synthetic) accounts operated from the "
                     "same phones / network")
    if "collusive_merchant" in kinds:
        parts.append("Merchant collusion: accounts pushing round-amount purchases through a "
                     "newly onboarded merchant in coordinated bursts")
    if "mule_payee" in kinds and not parts:
        parts.append(f"Mule cash-out: {size - 1} accounts sent money to one mule account, "
                     "consistent with compromised accounts being drained")
    elif "mule_payee" in kinds:
        parts.append("money collected by a mule that passes it straight on to gift cards / crypto")
    return "; ".join(parts)


def find_rings(g, mules):
    rings = []
    linked = g.edge_subgraph([e for e in g.edges if g.edges[e]["evidence"]])
    for comp in sorted(nx.connected_components(linked), key=len, reverse=True):
        if len(comp) < 3:
            continue
        kinds = defaultdict(int)
        details = set()
        for a, b in linked.subgraph(comp).edges:
            for ev in g[a][b]["evidence"]:
                kinds[ev["kind"]] += 1
                details.add(ev["detail"])
        score = 1.0
        for kind in kinds:
            score *= 1 - EDGE_WEIGHT[kind]
        edges = [{"a": a, "b": b,
                  "kinds": sorted({ev["kind"] for ev in g[a][b]["evidence"]}),
                  "evidence": [dict(t) for t in {tuple(ev.items()) for ev in g[a][b]["evidence"]}]}
                 for a, b in linked.subgraph(comp).edges]
        rings.append({
            "ring_id": f"RING-{len(rings) + 1}", "members": sorted(comp), "size": len(comp),
            "pattern": describe_pattern(kinds, len(comp)),
            "edges": edges,
            "score": round(1 - score, 3), "evidence_kinds": dict(kinds),
            "evidence": sorted(details),
            "mules": sorted(m for m in comp if m in mules),
            "evidence_txns": sorted(set().union(*(g.nodes[n].get("evidence_txns", set())
                                                  for n in comp))),
        })
    return rings
