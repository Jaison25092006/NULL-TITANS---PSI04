"""Live scoring of a single new transaction against the full history.

The real-time and behavioural layers are replayed once over the history (a couple of
seconds), after which any hypothetical transaction is scored in milliseconds with
`peek` (state is restored afterwards, so what-if scoring never pollutes the history).
Network context (rings, mules, linked accounts) comes from the latest batch run.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from . import action_engine as actions
from .anomaly_detection import ANOMALY_WEIGHT, anomaly_pct
from .behavioral_analysis import BehaviorTracker
from .data_loader import TX_COLUMNS
from .explainability import checklist, narrative
from .fraud_model import ML_WEIGHT_MAX, predict
from .risk_engine import band, final_risk
from .stream import StreamScorer, noisy_or


class LiveScorer:
    def __init__(self, txn, accounts, merchants, acc, rings, bundle=None, anomaly=None):
        raw = txn[TX_COLUMNS].sort_values("timestamp", kind="stable")
        self.stream = StreamScorer(accounts, merchants)
        self.behavior = BehaviorTracker(merchants)
        for t in raw.itertuples(index=False):
            self.stream.score(t)
            self.behavior.score(t)
        self.bundle, self.anomaly = bundle, anomaly or {}
        self.merchants = merchants
        self.rings = {r["ring_id"]: r for r in rings}
        self.ring_of = {m: r for r in rings for m in r["members"]}
        self.mule_ring = {m: r for r in rings for m in r["mules"]}
        self.controlled = set(self.mule_ring)
        for r in rings:
            self.controlled |= set(r.get("synthetic_identities", []))
        self.neighbour = acc.set_index("account_id")["neighbour_risk"].to_dict() \
            if "neighbour_risk" in acc else {}
        self.n_history = len(raw)

    def score(self, event, weights=None):
        e = {c: event.get(c, "") for c in TX_COLUMNS}
        e["timestamp"] = pd.Timestamp(event["timestamp"])
        e["amount_inr"] = float(event.get("amount_inr", 0) or 0)
        e["txn_id"] = event.get("txn_id") or "LIVE-1"
        e["txn_type"] = "transfer" if e["payee_account_id"] else "purchase"
        e["channel"] = e["channel"] or ("upi" if e["payee_account_id"] else "app")
        e["country"] = e["country"] or "IN"
        e["city"] = e["city"] or "unknown"
        t = SimpleNamespace(**e)
        s = self.stream.peek(t)
        b = self.behavior.peek(t)
        feats = {**{k: v for k, v in s.items() if k.startswith("f_")},
                 **{k: v for k, v in b.items() if k.startswith("b_")}}
        reasons = [{**r, "layer": "rule"} for r in s["reasons"]]

        pct = 0.0
        model, ref, cols = (self.anomaly.get("model"), self.anomaly.get("reference"),
                            self.anomaly.get("cols"))
        if model is not None and ref is not None and cols:
            X = np.array([[float(feats.get(c, 0) or 0) for c in cols]])
            _, p = anomaly_pct(model, ref, X)
            pct = float(p[0])
            if pct >= 99.5:
                reasons.append({"code": "ANOMALY", "weight": ANOMALY_WEIGHT, "layer": "anomaly",
                                "evidence": f"more unusual than {pct:.1f}% of baseline activity"})
        p_ml = float(predict(self.bundle, pd.DataFrame([feats]))[0]) if self.bundle else 0.0
        if p_ml >= 0.2:
            reasons.append({"code": "ML_MODEL", "weight": round(ML_WEIGHT_MAX * p_ml, 3),
                            "layer": "model",
                            "evidence": f"{self.bundle['name']} trained on confirmed fraud cases "
                                        f"rates this {p_ml:.0%} likely to be fraud"})
        T = round(100 * noisy_or(reasons), 1)

        acct, payee = e["account_id"], e["payee_account_id"]
        N, ev, ring_id = 0.0, False, ""
        if payee in self.mule_ring:
            r = self.mule_ring[payee]
            N, ev, ring_id = r["ring_risk"], True, r["ring_id"]
            reasons.append({"code": "RING_EVIDENCE", "weight": N / 100, "layer": "network",
                            "evidence": f"transfer to {payee}, the collecting (mule) account of "
                                        f"{r['ring_id']} ({r['size']} accounts)"})
        if acct in self.controlled and acct in self.ring_of:
            r = self.ring_of[acct]
            if r["ring_risk"] > N:
                N = r["ring_risk"]
            ev, ring_id = True, r["ring_id"]
            reasons.append({"code": "CONTROLLED_ACCOUNT", "weight": r["ring_risk"] / 100,
                            "layer": "network",
                            "evidence": f"account is fraudster-controlled ({r['ring_id']})"})
        elif acct in self.ring_of and not ev:
            r = self.ring_of[acct]
            N, ring_id = 0.4 * r["ring_risk"], r["ring_id"]
            reasons.append({"code": "RING_MEMBER_ACCOUNT", "weight": N / 100, "layer": "network",
                            "evidence": f"account is a member of {r['ring_id']}"})
        elif self.neighbour.get(acct, 0) > 0 and N == 0:
            N = min(40.0, 0.4 * self.neighbour[acct])
            reasons.append({"code": "CONNECTED_RISK", "weight": N / 100, "layer": "network",
                            "evidence": f"account is linked to a high-risk account "
                                        f"(neighbour risk {self.neighbour[acct]:.0f})"})

        fr = final_risk([T], [b["behavioral_risk"]], [N], [ev], weights).iloc[0]
        bnd = band(fr["final_risk"])
        cat = self.merchants.set_index("merchant_id")["category"].get(e["merchant_id"], "")
        row = {**e, **feats, "transaction_risk": T, "behavioral_risk": b["behavioral_risk"],
               "network_risk": round(N, 1), "network_evidence": ev,
               "b_avg_amount": b["b_avg_amount"], "b_amount_ratio": b["b_amount_ratio"],
               "ml_probability": round(p_ml, 4), "anomaly_pct": round(pct, 2),
               "blend": fr["blend"], "final_risk": fr["final_risk"], "guardrail": fr["guardrail"],
               "band": bnd, "action": actions.txn_action(bnd), "status": actions.STATUS[bnd],
               "ring_id": ring_id, "merchant_category": cat or ("transfer" if payee else ""),
               "realtime_risk": s["realtime_risk"]}
        ring = self.rings.get(ring_id)
        row["explanation"] = narrative(row, reasons, ring)
        row["checklist"] = checklist(row, reasons, ring)
        row["reasons"] = reasons
        return row
