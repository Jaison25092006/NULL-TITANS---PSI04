"""Explainable AI: every score comes with its evidence.

  checklist()      the "WHY WAS THIS FLAGGED?" list: which warning signs fired (✓) and
                   which were checked but not observed (context for the analyst)
  narrative()      a plain-English sentence explaining the decision
  waterfall()      how each reason moved the Transaction Risk (noisy-OR increments)
  fp_protection()  explicit note when a high amount was *not* treated as suspicious

ML feature contributions are in fraud_model.contributions().
"""

import json

LAYER = {"ML_MODEL": "model", "ANOMALY": "anomaly", "RING_EVIDENCE": "network",
         "CONTROLLED_ACCOUNT": "network", "RING_MEMBER_ACCOUNT": "network",
         "CONNECTED_RISK": "network"}
HIGH_AMOUNT = 50000


def parse_reasons(value):
    if isinstance(value, list):
        return value
    try:
        return json.loads(value) if value else []
    except (TypeError, ValueError):
        return []


def _codes(reasons):
    return {r["code"] for r in reasons}


def checklist(row, reasons, ring=None):
    """List of dicts: key, hit, label, detail."""
    c = _codes(reasons)
    ratio = float(row.get("b_amount_ratio", 1) or 1)
    hour = row["timestamp"].strftime("%H:%M") if hasattr(row.get("timestamp"), "strftime") else ""
    kinds = set((ring or {}).get("evidence_kinds", {}))
    in_ring_evidence = "RING_EVIDENCE" in c
    items = [
        ("device", row.get("b_device", 0) > 0 or bool(c & {"NEW_DEVICE", "NEW_DEVICE_ABROAD"}),
         "New device", f"device {row.get('device_id') or '—'} not in this account's history"),
        ("amount", row.get("b_amount", 0) > 0 or bool(c & {"AMOUNT_SPIKE", "AMOUNT_SPIKE_NEW_DEVICE"}),
         f"Amount {ratio:.1f}× account average",
         f"₹{row.get('amount_inr', 0):,.0f} vs typical ₹{row.get('b_avg_amount', 0):,.0f}"),
        ("time", row.get("b_time", 0) > 0 or "NIGHT_ACTIVITY" in c, "Unusual transaction time",
         f"at {hour}, outside this account's usual hours"),
        ("velocity", row.get("b_velocity", 0) > 0 or bool(c & {"VELOCITY", "CARD_TEST_BURST"}),
         "High transaction velocity", "far more transactions than this account's daily norm"),
        ("location", row.get("b_location", 0) > 0 or "NEW_DEVICE_ABROAD" in c, "Unusual location",
         f"{row.get('city', '')}, {row.get('country', '')} not seen before for this account"),
        ("beneficiary", bool(c & {"PAYEE_FAN_IN", "NEW_PAYEE_LARGE", "STRUCTURING"})
         or (in_ring_evidence and "mule_payee" in kinds and bool(row.get("payee_account_id"))),
         "Suspicious beneficiary",
         f"{row.get('payee_account_id') or '—'} collects from many accounts / passes money on"),
        ("connected", row.get("network_risk", 0) >= 31, "Connected to high-risk account",
         f"network risk {row.get('network_risk', 0):.0f}/100"),
        ("sequence", in_ring_evidence and bool(kinds & {"same_sequence", "coordinated_timing",
                                                         "collusive_merchant"}),
         "Similar transaction sequence detected",
         "same steps, same order, same time window as other ring members"),
        ("anomaly", "ANOMALY" in c, "Statistical anomaly",
         f"more unusual than {row.get('anomaly_pct', 0):.1f}% of baseline activity"),
        ("model", row.get("ml_probability", 0) >= 0.5, "ML model: resembles confirmed fraud",
         f"probability {row.get('ml_probability', 0):.0%}"),
    ]
    return [{"key": k, "hit": bool(h), "label": lab, "detail": d} for k, h, lab, d in items]


def narrative(row, reasons, ring=None, band=None):
    band = band or row.get("band", "")
    c = _codes(reasons)
    risk = row.get("final_risk", 0)
    tid = row.get("txn_id", "this transaction")
    clauses = []
    ratio = float(row.get("b_amount_ratio", 1) or 1)
    if "RING_EVIDENCE" in c and ring:
        how = []
        kinds = ring.get("evidence_kinds", {})
        if "same_sequence" in kinds:
            how.append("repeating the same purchase sequence")
        if "mule_payee" in kinds:
            how.append("paying the same beneficiary")
        if "shared_device" in kinds or "linked_device" in kinds:
            how.append("sharing devices")
        if "collusive_merchant" in kinds:
            how.append("buying through the same newly onboarded merchant")
        clauses.append(f"it is part of the evidence linking fraud ring {ring['ring_id']} "
                       f"({ring['size']} accounts {', '.join(how) or 'acting together'})")
    if "CONTROLLED_ACCOUNT" in c:
        clauses.append("it was made by a fraudster-controlled account (mule or synthetic identity)")
    if "CARD_TEST_CASHOUT" in c or "CARD_TEST_BURST" in c:
        clauses.append("it is part of a card-testing pattern (tiny test payments, then a large purchase)")
    if "STRUCTURING" in c:
        clauses.append("it is one of several transfers kept just under the ₹50,000 reporting threshold")
    if "PASS_THROUGH" in c:
        clauses.append("it passes on money the account received hours earlier (mule behaviour)")
    if row.get("b_amount", 0) > 0 and ratio >= 1.5:
        clauses.append(f"the amount is {ratio:.1f}× the account's normal spending")
    if row.get("b_device", 0) > 0 or c & {"NEW_DEVICE", "NEW_DEVICE_ABROAD"}:
        clauses.append("it originates from a newly observed device")
    if row.get("b_location", 0) > 0:
        clauses.append(f"it comes from {row.get('city')}, a location this account has not used")
    if row.get("b_time", 0) > 0 or "NIGHT_ACTIVITY" in c:
        hour = row["timestamp"].strftime("%H:%M") if hasattr(row.get("timestamp"), "strftime") else ""
        clauses.append(f"it happened at {hour}, outside the account's usual hours")
    if "PAYEE_FAN_IN" in c:
        clauses.append("the beneficiary is a new account collecting money from many senders")
    if row.get("ml_probability", 0) >= 0.5:
        clauses.append(f"the ML model rates it {row['ml_probability']:.0%} similar to confirmed fraud")
    if band in ("LOW", "MEDIUM") and not clauses:
        text = (f"{tid} is {band} risk ({risk:.0f}/100): it is consistent with this account's "
                f"normal behaviour.")
    elif band in ("LOW", "MEDIUM"):
        text = (f"{tid} is {band} risk ({risk:.0f}/100). Some signals were observed "
                f"({'; '.join(clauses[:2])}), but not enough independent evidence to act.")
    else:
        text = f"{tid} is {band} risk ({risk:.0f}/100) because " + _join(clauses[:4]) + "."
    note = fp_protection(row)
    return text + (" " + note if note else "")


def _join(parts):
    if not parts:
        return "several weak signals add up"
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + ", and " + parts[-1]


def fp_protection(row):
    amount = float(row.get("amount_inr", 0))
    if amount >= HIGH_AMOUNT and row.get("final_risk", 0) < 61:
        avg = float(row.get("b_avg_amount", 0) or 0)
        ctx = f"this account's comparable transactions average ₹{avg:,.0f}" if avg else \
            "the surrounding behaviour is normal"
        return (f"High amount alone is insufficient evidence: ₹{amount:,.0f} matches context "
                f"({ctx}; known device and beneficiary patterns).")
    return ""


def waterfall(reasons):
    """[(code, layer, weight, increment_in_points, evidence)] in order of weight."""
    out, p = [], 1.0
    for r in sorted(reasons, key=lambda r: -r["weight"]):
        before = 1 - p
        p *= 1 - r["weight"]
        out.append((r["code"], r.get("layer", LAYER.get(r["code"], "rule")), r["weight"],
                    round(100 * ((1 - p) - before), 1), r["evidence"]))
    return out
