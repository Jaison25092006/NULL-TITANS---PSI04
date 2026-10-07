"""Fraud DNA: a behavioural fingerprint per account.

Each account gets a vector describing *how* it behaves in its most relevant activity
(its "incident window": ring-evidence transactions and high-risk transactions, or its
latest activity when nothing is suspicious):

  amount pattern · time pattern · device pattern · location pattern · merchant pattern ·
  velocity pattern · network pattern · transaction sequence (order of categories)

Vectors are standardised against the whole population and compared with cosine
similarity. Accounts with different identities, cities and phones that still share a
~90% fingerprint are acting from the same playbook, which raises network risk.
"""

import numpy as np
import pandas as pd

CATS = ["grocery", "restaurants", "fuel", "pharmacy", "fashion", "online_marketplace",
        "utilities", "electronics", "travel", "gift_cards", "crypto_exchange", "transfer"]
RADAR = ["Amount", "Time", "Device", "Location", "Merchant", "Velocity", "Network", "Sequence"]
SIMILAR = 0.80  # "highly similar" fingerprints


def _incident(g, evidence, risk_col):
    inc = g[g["txn_id"].isin(evidence) | (g[risk_col] >= 50)]
    return inc if len(inc) else g.tail(5)


def fingerprints(txn, evidence_txns=frozenset(), risk_col="transaction_risk"):
    """Returns (vectors DataFrame indexed by account, radar DataFrame indexed by account)."""
    evidence = set(evidence_txns)
    cat = txn["merchant_category"].where(txn["merchant_category"].isin(CATS), "other") \
        if "merchant_category" in txn else pd.Series("other", index=txn.index)
    t = txn.assign(_cat=cat)
    vec_rows, radar_rows = {}, {}
    for acct, g in t.groupby("account_id", sort=False):
        inc = _incident(g, evidence, risk_col).sort_values("timestamp")
        n = len(inc)
        v = {}
        shares = inc["_cat"].value_counts(normalize=True)
        for c in CATS:
            v[f"share_{c}"] = float(shares.get(c, 0))
        order = {c: i for i, c in enumerate(inc["_cat"])}
        for c in CATS:  # where in the sequence each category first appears (0 = absent)
            v[f"pos_{c}"] = (1 - order[c] / max(n, 1)) if c in order else 0.0
        amt = np.log1p(inc["amount_inr"].to_numpy(float))
        v["amount_mean"], v["amount_std"] = float(amt.mean()), float(amt.std())
        span = (inc["timestamp"].max() - inc["timestamp"].min()).total_seconds() / 60
        v["duration"], v["n"] = float(np.log1p(span)), float(np.log1p(n))
        ang = 2 * np.pi * inc["timestamp"].dt.hour.to_numpy() / 24
        v["hour_sin"], v["hour_cos"] = float(np.sin(ang).mean()), float(np.cos(ang).mean())
        v["night"] = float((inc["timestamp"].dt.hour < 5).mean())
        v["foreign"] = float((inc["country"] != "IN").mean())
        v["new_device"] = float(inc.get("f_new_device", pd.Series(0)).mean())
        v["devices"] = float(inc.loc[inc["device_id"] != "", "device_id"].nunique())
        v["payee_young"] = float(inc.get("f_payee_young", pd.Series(0)).mean())
        v["payee_fan_in"] = float(np.log1p(inc.get("f_payee_senders", pd.Series(0)).mean()))
        v["velocity"] = float(inc.get("f_velocity_1h", pd.Series(0)).max())
        vec_rows[acct] = v

        def m(col):
            return float(inc[col].mean()) if col in inc else 0.0
        radar_rows[acct] = {
            "Amount": m("b_amount"), "Time": m("b_time"), "Device": m("b_device"),
            "Location": m("b_location"), "Merchant": m("b_merchant"),
            "Velocity": max(m("b_velocity"), min(v["velocity"] / 5, 1)),
            "Network": min(1.0, v["payee_fan_in"] / 2 + v["payee_young"] / 2
                           + (len(set(inc["txn_id"]) & evidence) > 0) * 0.5),
            "Sequence": min(1.0, shares.get("gift_cards", 0) + shares.get("crypto_exchange", 0)
                            + shares.get("transfer", 0) * 0.5),
        }
    vectors = pd.DataFrame.from_dict(vec_rows, orient="index").fillna(0)
    radar = pd.DataFrame.from_dict(radar_rows, orient="index").fillna(0).clip(0, 1)
    return vectors, radar


def standardise(vectors):
    if vectors.empty:
        return vectors
    std = vectors.std().replace(0, 1).clip(lower=0.05)
    return (vectors - vectors.mean()) / std


def similarity(z, accounts=None):
    """Cosine similarity matrix (DataFrame) between the given accounts (default: all)."""
    sub = z if accounts is None else z.loc[[a for a in accounts if a in z.index]]
    if sub.empty:
        return pd.DataFrame()
    x = sub.to_numpy(float)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm[norm == 0] = 1
    x = x / norm
    return pd.DataFrame(x @ x.T, index=sub.index, columns=sub.index)


def most_similar(z, account, k=5, exclude=()):
    if account not in z.index:
        return pd.Series(dtype=float)
    x = z.to_numpy(float)
    norm = np.linalg.norm(x, axis=1)
    norm[norm == 0] = 1
    q = x[z.index.get_loc(account)]
    sims = pd.Series((x @ q) / (norm * norm[z.index.get_loc(account)]), index=z.index)
    sims = sims.drop([account, *[e for e in exclude if e in sims.index]])
    return sims.sort_values(ascending=False).head(k)


def mean_pairwise(z, accounts):
    s = similarity(z, accounts)
    if len(s) < 2:
        return None
    vals = s.to_numpy()[np.triu_indices(len(s), 1)]
    return float(vals.mean())
