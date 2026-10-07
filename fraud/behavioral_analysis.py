"""Behavioral profiling: learn each account's normal behaviour and measure how far every
new transaction deviates from it.

Two views:
  BehaviorTracker   causal (past-only) deviation of each transaction from the account's
                    profile so far, along 8 dimensions; this feeds the Behavioral Risk.
  build_profiles    full-history profile of each account, shown to the analyst
                    (normal amount range, typical hours, locations, devices, ...).

Behavioral Risk is a weighted average of the dimension deviations, so a single unusual
dimension (e.g. only the amount) can never make it high on its own:
"high amount alone is insufficient evidence".
"""

import copy
import math
from collections import Counter, defaultdict, deque

import numpy as np
import pandas as pd

# dimension: (weight, label)
DIMENSIONS = {
    "amount": (0.22, "Amount"),
    "device": (0.18, "Device"),
    "location": (0.14, "Location"),
    "time": (0.14, "Time of day"),
    "velocity": (0.12, "Velocity"),
    "merchant": (0.10, "Merchant / category"),
    "beneficiary": (0.10, "Beneficiary"),
}
MIN_HISTORY = 5   # below this, the baseline is too thin to call anything unusual
LEARN_LAG_H = 24  # a transaction joins the baseline only once it is 24h old, so an
                  # attacker's first few transactions cannot "teach" the profile
RISKY = {"gift_cards", "crypto_exchange"}


class _Profile:
    def __init__(self):
        self.n = 0
        self.first = None
        self.type_stats = defaultdict(lambda: [0, 0.0, 0.0])  # type -> n, sum log, sum log^2
        self.amount_sum = defaultdict(float)
        self.hours = np.zeros(24)
        self.cities = Counter()
        self.devices = set()
        self.digital = 0
        self.merchants, self.categories, self.channels, self.payees = set(), set(), set(), set()
        self.recent = deque()  # timestamps within 24h
        self.pending = deque()  # transactions not yet old enough to join the baseline


def _circular_share(hours, h, width=2):
    total = hours.sum()
    if total == 0:
        return 1.0
    idx = [(h + d) % 24 for d in range(-width, width + 1)]
    return hours[idx].sum() / total


class BehaviorTracker:
    def __init__(self, merchants):
        self.category = merchants.set_index("merchant_id")["category"].to_dict()
        self.p = defaultdict(_Profile)

    def peek(self, t):
        existed = t.account_id in self.p
        saved = copy.deepcopy(self.p.get(t.account_id))
        try:
            return self.score(t)
        finally:
            if existed:
                self.p[t.account_id] = saved
            else:
                self.p.pop(t.account_id, None)

    def score(self, t):
        p = self.p[t.account_id]
        now, amount = t.timestamp, float(t.amount_inr)
        kind = "transfer" if t.payee_account_id else "purchase"
        cat = self.category.get(t.merchant_id, "transfer" if t.payee_account_id else "general")
        while p.recent and (now - p.recent[0]).total_seconds() > 86400:
            p.recent.popleft()
        while p.pending and (now - p.pending[0][0]).total_seconds() > LEARN_LAG_H * 3600:
            self._learn(p, *p.pending.popleft())
        dev = dict.fromkeys(DIMENSIONS, 0.0)
        info = {}
        enough = p.n >= MIN_HISTORY

        # Amount: compare with the same kind of transaction when there is enough of it, so a
        # customer who routinely moves lakhs is judged against their own transfers.
        n_t, s_t, q_t = p.type_stats[kind]
        if n_t >= 3:
            n, s, q, avg = n_t, s_t, q_t, p.amount_sum[kind] / n_t
        else:
            n = sum(v[0] for v in p.type_stats.values())
            s = sum(v[1] for v in p.type_stats.values())
            q = sum(v[2] for v in p.type_stats.values())
            avg = sum(p.amount_sum.values()) / n if n else 0.0
        info["avg_amount"] = avg
        info["amount_ratio"] = amount / avg if avg > 0 else 1.0
        if enough and n >= 3:
            mean = s / n
            std = max(math.sqrt(max(q / n - mean * mean, 0)), 0.35)
            z = (math.log1p(amount) - mean) / std
            dev["amount"] = float(np.clip((z - 1.5) / 2.5, 0, 1))
            info["amount_z"] = z

        if enough:
            share = _circular_share(p.hours, now.hour)
            dev["time"] = 1.0 if share < 0.03 else (0.5 if share < 0.08 else 0.0)
            info["hour_share"] = share
            location = t.city if t.country == "IN" else f"{t.city}, {t.country}"
            if location not in p.cities:
                dev["location"] = 1.0 if t.country != "IN" else 0.6
            days = max((now - p.first).total_seconds() / 86400, 1.0)
            avg_daily = p.n / days
            today = len(p.recent) + 1
            info["avg_daily"], info["today"] = avg_daily, today
            ratio = today / max(avg_daily, 0.5)
            dev["velocity"] = float(np.clip((ratio - 3) / 5, 0, 1))
            if cat not in p.categories:
                dev["merchant"] = 1.0 if cat in RISKY else 0.6
            elif t.merchant_id and t.merchant_id not in p.merchants:
                dev["merchant"] = 0.2
        if t.channel != "pos" and t.device_id and p.digital >= 3 and t.device_id not in p.devices:
            dev["device"] = 1.0
        if t.payee_account_id and p.n >= MIN_HISTORY and t.payee_account_id not in p.payees:
            dev["beneficiary"] = 1.0 if dev["amount"] > 0.3 else 0.5

        b = sum(DIMENSIONS[k][0] * v for k, v in dev.items())
        row = {"txn_id": t.txn_id, "behavioral_risk": round(100 * b, 1),
               "b_history": p.n, **{f"b_{k}": round(v, 3) for k, v in dev.items()},
               "b_avg_amount": round(info["avg_amount"], 2),
               "b_amount_ratio": round(info["amount_ratio"], 2)}

        p.recent.append(now)
        p.pending.append((now, amount, kind, cat, t.city, t.country, t.channel, t.device_id,
                          t.merchant_id, t.payee_account_id))
        return row

    @staticmethod
    def _learn(p, now, amount, kind, cat, city, country, channel, device, merchant, payee):
        p.n += 1
        p.first = p.first or now
        st = p.type_stats[kind]
        st[0] += 1
        st[1] += math.log1p(amount)
        st[2] += math.log1p(amount) ** 2
        p.amount_sum[kind] += amount
        p.hours[now.hour] += 1
        p.cities[city if country == "IN" else f"{city}, {country}"] += 1
        if channel != "pos" and device:
            p.devices.add(device)
            p.digital += 1
        if merchant:
            p.merchants.add(merchant)
        p.categories.add(cat)
        p.channels.add(channel)
        if payee:
            p.payees.add(payee)


def score_behavior(tx, merchants, tracker=None):
    tr = tracker or BehaviorTracker(merchants)
    out = pd.DataFrame([tr.score(t) for t in tx.itertuples(index=False)])
    return out, tr


def deviation_findings(row):
    """Human-readable findings for one scored transaction (row with b_* columns)."""
    out = []
    if row.get("b_amount", 0) > 0:
        out.append(("amount", f"Amount {row.get('b_amount_ratio', 1):.1f}× the account's average "
                              f"(₹{row.get('b_avg_amount', 0):,.0f})"))
    if row.get("b_time", 0) > 0:
        out.append(("time", "Unusual time of day for this account"))
    if row.get("b_location", 0) > 0:
        out.append(("location", "Location never seen for this account"))
    if row.get("b_device", 0) > 0:
        out.append(("device", "New device for this account"))
    if row.get("b_velocity", 0) > 0:
        out.append(("velocity", "Many more transactions today than this account's daily norm"))
    if row.get("b_merchant", 0) > 0:
        out.append(("merchant", "Merchant / category this account has never used"))
    if row.get("b_beneficiary", 0) > 0:
        out.append(("beneficiary", "First transfer to this beneficiary"))
    return out


def level(score):
    return "HIGH" if score >= 60 else "MEDIUM" if score >= 30 else "LOW"


def build_profiles(tx, merchants):
    """Full-history behavioural profile per account (for display)."""
    category = merchants.set_index("merchant_id")["category"]
    name = merchants.set_index("merchant_id")["merchant_name"]
    t = tx.assign(hour=tx["timestamp"].dt.hour, day=tx["timestamp"].dt.date,
                  location=np.where(tx["country"] == "IN", tx["city"],
                                    tx["city"] + ", " + tx["country"]))
    rows = []
    for acct, g in t.groupby("account_id", sort=False):
        days = max((g["timestamp"].max() - g["timestamp"].min()).days, 1)
        hrs = g["hour"].to_numpy()

        def top(series, k=3):
            vc = series[series != ""].value_counts()
            return ", ".join(f"{i}" for i in vc.index[:k])
        rows.append({
            "account_id": acct, "n_txns": len(g),
            "avg_amount": round(g["amount_inr"].mean(), 2),
            "amount_p10": round(g["amount_inr"].quantile(0.10), 2),
            "amount_p90": round(g["amount_inr"].quantile(0.90), 2),
            "avg_daily_txns": round(len(g) / days, 2),
            "typical_hours": f"{int(np.percentile(hrs, 10)):02d}:00–{int(np.percentile(hrs, 90)):02d}:59",
            "typical_locations": top(g["location"]),
            "typical_devices": top(g["device_id"]),
            "n_devices": g.loc[g["device_id"] != "", "device_id"].nunique(),
            "typical_merchants": ", ".join(name.get(m, m) for m in
                                           g.loc[g["merchant_id"] != "", "merchant_id"]
                                           .value_counts().index[:3]),
            "typical_categories": top(g["merchant_id"].map(category).fillna("")),
            "typical_channels": top(g["channel"]),
            "normal_beneficiaries": top(g["payee_account_id"]),
            "n_locations": g["location"].nunique(),
            "n_merchants": g.loc[g["merchant_id"] != "", "merchant_id"].nunique(),
            "n_beneficiaries": g.loc[g["payee_account_id"] != "", "payee_account_id"].nunique(),
        })
    return pd.DataFrame(rows)
