"""Fraud Ring Detection: turn linked account groups into scored, explained rings.

network.py finds the groups (connected components of strong evidence links). This
module adds the evidence that makes a ring convincing and scores it:

  coordinated_timing   most members were active inside one short window
  dna_similarity       members share a highly similar behavioural fingerprint (Fraud DNA)
  rapid_movement       the common beneficiary moves the money on within hours

  Ring Risk = 100 x (1 - prod(1 - w_k)) over every kind of evidence present (capped at 99),
  so the score is a transparent sum of named evidence, shown kind by kind.
"""

import numpy as np

from .action_engine import ring_actions
from .fraud_dna import SIMILAR, mean_pairwise
from .network import EDGE_LABEL, EDGE_WEIGHT
from .risk_engine import band
from .temporal_analysis import peak_coordination

ANNOTATION_WEIGHT = {"coordinated_timing": 0.5, "dna_similarity": 0.4, "rapid_movement": 0.3}
ANNOTATION_LABEL = {"coordinated_timing": "Coordinated timing",
                    "dna_similarity": "Behavioural similarity (Fraud DNA)",
                    "rapid_movement": "Rapid fund movement"}
WEIGHTS = {**EDGE_WEIGHT, **ANNOTATION_WEIGHT}
PREFIX = {"shared_device": "shared device", "shared_subnet": "shared subnet",
          "linked_device": "device ", "same_sequence": "same purchase sequence",
          "mule_payee": "sent ₹", "collusive_merchant": "customer of"}
LABELS = {**EDGE_LABEL, **ANNOTATION_LABEL}


def annotate(rings, g, mules, txn, merchants, dna_z):
    name = merchants.set_index("merchant_id")["merchant_name"].to_dict()
    collusive = g.graph.get("collusive_merchants", {})
    out = []
    for r in rings:
        members = r["members"]
        actors = [m for m in members if m not in r["mules"]] or members
        ev = txn[txn["txn_id"].isin(r["evidence_txns"])]
        breakdown = []
        for kind, n_links in sorted(r["evidence_kinds"].items(), key=lambda kv: -WEIGHTS[kv[0]]):
            example = next((d for d in r["evidence"] if d.startswith(PREFIX[kind])), "")
            breakdown.append({"kind": kind, "label": LABELS[kind], "weight": WEIGHTS[kind],
                              "detail": f"{n_links} account links · e.g. {example}"})

        peak = peak_coordination(ev[ev["account_id"].isin(actors)], actors, window_min=60)
        if peak["members_active"] >= 3 and peak["members_active"] >= 0.5 * len(actors):
            breakdown.append({"kind": "coordinated_timing", "label": LABELS["coordinated_timing"],
                              "weight": WEIGHTS["coordinated_timing"],
                              "detail": f"{peak['members_active']} of {len(actors)} accounts "
                                        f"active within {peak['span_min']:.0f} minutes"})
        dna = mean_pairwise(dna_z, actors)
        if dna is not None and dna >= SIMILAR:
            breakdown.append({"kind": "dna_similarity", "label": LABELS["dna_similarity"],
                              "weight": WEIGHTS["dna_similarity"],
                              "detail": f"average pairwise behavioural similarity {dna:.0%}"})
        delays = [mules[m]["median_delay_h"] for m in r["mules"]
                  if mules.get(m, {}).get("median_delay_h") is not None]
        if delays and min(delays) <= 6:
            breakdown.append({"kind": "rapid_movement", "label": LABELS["rapid_movement"],
                              "weight": WEIGHTS["rapid_movement"],
                              "detail": f"beneficiary passed funds on to crypto/gift cards "
                                        f"{min(delays):.1f}h after receiving them (median)"})

        risk = min(99.0, round(100 * (1 - np.prod([1 - b["weight"] for b in breakdown])), 1))
        member_tx = txn[txn["account_id"].isin(members)]
        dev_users = member_tx[member_tx["device_id"] != ""].groupby("device_id")["account_id"] \
            .nunique()
        shared_devices = sorted(dev_users[dev_users >= 2].index)
        synthetic = sorted({m for e in r["edges"] if "shared_device" in e["kinds"]
                            for m in (e["a"], e["b"])})
        ring = {
            **r,
            "synthetic_identities": synthetic,
            "ring_risk": risk, "band": band(risk),
            "evidence_breakdown": breakdown,
            "evidence_kinds": {**r["evidence_kinds"],
                               **{b["kind"]: 1 for b in breakdown if b["kind"] in ANNOTATION_WEIGHT}},
            "devices": shared_devices or sorted(set(ev["device_id"]) - {""})[:12],
            "all_devices": sorted(set(ev["device_id"]) - {""}),
            "merchants": sorted({name.get(m, m) for m in set(ev["merchant_id"]) - {""}}),
            "beneficiaries": sorted(set(ev["payee_account_id"]) - {""}),
            "locations": sorted(set(ev["city"]) - {"", "unknown"}),
            "collusive_merchants": sorted(name.get(m, m) for m in collusive
                                          if m in set(ev["merchant_id"])),
            "peak_window": peak, "dna_similarity": None if dna is None else round(dna, 3),
            "first_seen": str(ev["timestamp"].min()) if len(ev) else "",
            "last_seen": str(ev["timestamp"].max()) if len(ev) else "",
            "amount_inr": round(float(ev["amount_inr"].sum()), 2),
        }
        ring["recommended_action"], steps = ring_actions(ring)
        ring["actions"] = [{"step": s, "detail": d} for s, d in steps]
        out.append(ring)
    out.sort(key=lambda x: (-x["ring_risk"], -x["size"]))
    for i, ring in enumerate(out):
        ring["ring_id"] = f"FR-{i + 1:03d}"
    return out
