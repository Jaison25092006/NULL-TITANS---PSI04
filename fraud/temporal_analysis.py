"""Temporal pattern detection: fraud is about WHEN things happen, not only who is connected.

  coordination_bursts   3+ distinct accounts hitting the same rare merchant or the same
                        beneficiary inside a configurable time window (5 min ... 24 h)
  coordinated_sequences bursts that share the same accounts, chained in time: "3 accounts
                        bought at Merchant X within 12 minutes, then all paid Beneficiary Z"
  peak_coordination     tightest window in which most of a ring's members were active

Everyday merchants (used by >= 50% of accounts) are ignored, and beneficiaries that are
business accounts (landlords, suppliers) are marked as benign hubs, so rent day does not
look like a fraud ring.
"""

import numpy as np
import pandas as pd

RARE_SHARE = 0.50  # ignore everyday merchants (groceries, fuel, ...)


def _targets(txn, merchants, accounts):
    """Events with a 'target': a rare merchant or a beneficiary account."""
    name = merchants.set_index("merchant_id")["merchant_name"].to_dict()
    kind = accounts.set_index("account_id")["account_type"].to_dict()
    n_acc = max(txn["account_id"].nunique(), 1)
    share = txn[txn["merchant_id"] != ""].groupby("merchant_id")["account_id"].nunique() / n_acc
    rare = set(share[share < RARE_SHARE].index)
    m = txn[txn["merchant_id"].isin(rare)]
    b = txn[txn["payee_account_id"] != ""]
    ev = pd.concat([
        pd.DataFrame({"target": "M:" + m["merchant_id"], "target_type": "merchant",
                      "target_name": m["merchant_id"].map(name).fillna(m["merchant_id"]),
                      "benign_hub": False, "timestamp": m["timestamp"],
                      "account_id": m["account_id"], "txn_id": m["txn_id"],
                      "amount_inr": m["amount_inr"]}),
        pd.DataFrame({"target": "B:" + b["payee_account_id"], "target_type": "beneficiary",
                      "target_name": b["payee_account_id"],
                      "benign_hub": b["payee_account_id"].map(kind).eq("business"),
                      "timestamp": b["timestamp"], "account_id": b["account_id"],
                      "txn_id": b["txn_id"], "amount_inr": b["amount_inr"]}),
    ], ignore_index=True)
    return ev.sort_values("timestamp", kind="stable").reset_index(drop=True)


def coordination_bursts(txn, merchants, accounts, window_min=15, min_accounts=3):
    """Non-overlapping windows where >= min_accounts distinct accounts hit one target."""
    ev = _targets(txn, merchants, accounts)
    w = np.timedelta64(int(window_min * 60), "s")
    rows = []
    for target, g in ev.groupby("target", sort=False):
        if g["account_id"].nunique() < min_accounts:
            continue
        t = g["timestamp"].to_numpy()
        accts = g["account_id"].to_numpy()
        i, n = 0, len(g)
        while i < n:
            j = np.searchsorted(t, t[i] + w, side="right")
            members = set(accts[i:j])
            if len(members) >= min_accounts:
                sub = g.iloc[i:j]
                rows.append({
                    "target": target, "target_type": sub["target_type"].iloc[0],
                    "target_name": sub["target_name"].iloc[0],
                    "benign_hub": bool(sub["benign_hub"].iloc[0]),
                    "n_accounts": len(members), "accounts": sorted(members),
                    "start": sub["timestamp"].min(), "end": sub["timestamp"].max(),
                    "span_min": round((sub["timestamp"].max() - sub["timestamp"].min())
                                      .total_seconds() / 60, 1),
                    "n_txns": len(sub), "amount_inr": round(float(sub["amount_inr"].sum()), 2),
                    "txn_ids": list(sub["txn_id"])})
                i = j
            else:
                i += 1
    cols = ["target", "target_type", "target_name", "benign_hub", "n_accounts", "accounts",
            "start", "end", "span_min", "n_txns", "amount_inr", "txn_ids"]
    return pd.DataFrame(rows, columns=cols).sort_values(["n_accounts", "start"],
                                                        ascending=[False, True]) \
        .reset_index(drop=True)


def coordinated_sequences(bursts, window_min=15, min_shared=3):
    """Chain bursts that share >= min_shared accounts and follow each other closely."""
    cols = ["accounts", "n_accounts", "steps", "step_types", "start", "end", "span_min",
            "narrative", "txn_ids"]
    if bursts.empty:
        return pd.DataFrame(columns=cols)
    b = bursts[~bursts["benign_hub"].astype(bool)].sort_values("start").reset_index(drop=True)
    gap = pd.Timedelta(minutes=max(window_min, 15) * 4)
    used, chains = set(), []
    for i in range(len(b)):
        if i in used:
            continue
        chain, shared = [i], set(b.at[i, "accounts"])
        for j in range(i + 1, len(b)):
            if b.at[j, "start"] - b.at[chain[-1], "end"] > gap:
                if b.at[j, "start"] - b.at[chain[-1], "start"] > gap:
                    break
            common = shared & set(b.at[j, "accounts"])
            if len(common) >= min_shared and b.at[j, "target"] not in \
                    {b.at[k, "target"] for k in chain}:
                chain.append(j)
                shared = common
        if len(chain) >= 2:
            used.update(chain)
            steps = b.loc[chain]
            span = (steps["end"].max() - steps["start"].min()).total_seconds() / 60
            names = list(steps["target_name"])
            types = list(steps["target_type"])
            ends_in_transfer = types[-1] == "beneficiary"
            text = (f"{len(shared)} accounts performed similar transactions at "
                    f"{', '.join(n for n, t in zip(names, types) if t == 'merchant') or names[0]} "
                    f"within a {span:.0f}-minute window")
            if ends_in_transfer:
                text += f" and transferred funds to the same beneficiary ({names[-1]})"
            chains.append({"accounts": sorted(shared), "n_accounts": len(shared),
                           "steps": names, "step_types": types,
                           "start": steps["start"].min(), "end": steps["end"].max(),
                           "span_min": round(span, 1), "narrative": text,
                           "txn_ids": sorted({x for ids in steps["txn_ids"] for x in ids})})
    return pd.DataFrame(chains, columns=cols)


def peak_coordination(events, members, window_min=60):
    """events: DataFrame[timestamp, account_id] of a ring's evidence activity.
    Returns the max number of members active inside any window, and the tightest span
    (minutes) that covers that many members."""
    members = set(members)
    e = events[events["account_id"].isin(members)].sort_values("timestamp")
    if e.empty:
        return {"members_active": 0, "span_min": None, "start": None, "end": None}
    t = e["timestamp"].to_numpy()
    a = e["account_id"].to_numpy()
    w = np.timedelta64(int(window_min * 60), "s")
    best = (0, None, None, None)
    for i in range(len(e)):
        j = np.searchsorted(t, t[i] + w, side="right")
        k = len(set(a[i:j]))
        if k > best[0]:
            best = (k, i, j, None)
    k, i, j, _ = best
    # tightest span covering k members, starting at i
    seen, end = set(), i
    for end in range(i, j):
        seen.add(a[end])
        if len(seen) >= k:
            break
    span = (t[end] - t[i]) / np.timedelta64(60, "s")
    return {"members_active": int(k), "span_min": round(float(span), 1),
            "start": str(pd.Timestamp(t[i])), "end": str(pd.Timestamp(t[end]))}


def hour_day_heatmap(txn, mask=None):
    t = txn if mask is None else txn[mask]
    if t.empty:
        return pd.DataFrame(0, index=range(7), columns=range(24))
    return pd.crosstab(t["timestamp"].dt.dayofweek, t["timestamp"].dt.hour) \
        .reindex(index=range(7), columns=range(24), fill_value=0)
