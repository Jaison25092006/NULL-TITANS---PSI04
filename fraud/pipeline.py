"""End-to-end intelligence pipeline.

  data → preprocessing/enrichment → transaction features (real-time rules)
       → behavioral features (profile deviation) → anomaly detection → supervised model
       → graph construction → fraud-ring detection (+ temporal, Fraud DNA)
       → risk engine (Transaction / Behavioral / Network → Final) → explanations → actions

Reads data/, writes outputs/:
  transaction_scores.csv.gz  three risk scores, final risk, band, action, reasons, narrative
  account_scores.csv         account risk, action, profile, ring, connected accounts
  rings.json                 rings with members, entities, evidence breakdown, actions
  fraud_dna.csv              behavioural fingerprints (vectors + radar dimensions)
  run_summary.json           what ran, how long, on how much data

Run: python -m fraud.pipeline
"""

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import action_engine as actions
from .anomaly_detection import ANOMALY_WEIGHT, fit_detect
from .behavioral_analysis import build_profiles, level, score_behavior
from .config import FLAG_THRESHOLD, RISK_WEIGHTS
from .data_loader import enrich, load
from .explainability import narrative
from .fraud_dna import fingerprints, standardise
from .fraud_model import ML_WEIGHT_MAX, load_model, predict
from .fraud_ring import annotate
from .graph_analysis import account_projection, neighbour_risk
from .network import build_graph, find_rings
from .risk_engine import bands, final_risk
from .stream import REASONS, StreamScorer, noisy_or

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT, MODELS = ROOT / "data", ROOT / "outputs", ROOT / "models"
TXN_FILE = OUT / "transaction_scores.csv.gz"

# kept for backwards compatibility with older scripts
TXN_ACTIONS = [(0.81, "BLOCK + INVESTIGATE"), (0.61, "STEP-UP AUTHENTICATION"),
               (0.31, "MONITOR"), (0.0, "ALLOW")]


def action(score, table):
    return next(name for cut, name in table if score >= cut)


def _r(code, weight, evidence, layer):
    return {"code": code, "weight": round(float(weight), 3), "evidence": evidence, "layer": layer}


def run_frames(tx, accounts, merchants, bundle="auto", weights=None, verbose=False):
    """Run every layer on in-memory tables. Returns a results dict (nothing is written)."""
    clock, t0 = {}, time.time()

    def tick(name):
        nonlocal t0
        clock[name] = round(time.time() - t0, 2)
        if verbose:
            print(f"  {name:28s} {clock[name]:6.2f}s")
        t0 = time.time()

    tx = enrich(tx, accounts, merchants)
    tick("preprocessing")
    scorer = StreamScorer(accounts, merchants)
    s = pd.DataFrame([scorer.score(t) for t in tx.itertuples(index=False)])
    tick("transaction features")
    b, _ = score_behavior(tx, merchants)
    tick("behavioral profiling")
    feats = s.merge(b, on="txn_id")
    an = fit_detect(feats, tx["timestamp"])
    tick("anomaly detection")
    if isinstance(bundle, str):
        bundle = load_model()
    ml_p = predict(bundle, feats)
    tick("supervised model")

    # ---------------- Transaction Risk: rules + retro patterns + anomaly + model -------
    retro = {}
    for ids in feats["burst_members"]:
        for x in ids:
            retro.setdefault(x, []).append(_r(
                "CARD_TEST_BURST", 0.7, "one of a burst of tiny test payments at the same "
                "merchant (recognised after the burst, flagged retroactively)", "rule"))
    for ids in feats["struct_members"]:
        for x in ids:
            retro.setdefault(x, []).append(_r(
                "STRUCTURING", REASONS["STRUCTURING"][0], "one of several transfers just under "
                "the ₹50,000 reporting threshold within 48h (recognised later, flagged "
                "retroactively)", "rule"))
    model_name = bundle["name"] if bundle else "model"
    t_reasons = []
    for i, row in enumerate(feats.itertuples(index=False)):
        rs = [{**r, "layer": "rule"} for r in row.reasons]
        have = {r["code"] for r in rs}
        rs += [r for r in retro.get(row.txn_id, []) if r["code"] not in have]
        if an["is_anomaly"][i]:
            rs.append(_r("ANOMALY", ANOMALY_WEIGHT,
                         f"more unusual than {an['anomaly_pct'][i]:.1f}% of baseline activity "
                         f"(Isolation Forest on amount, time, device, velocity and profile "
                         f"deviation)", "anomaly"))
        if ml_p[i] >= 0.2:
            rs.append(_r("ML_MODEL", ML_WEIGHT_MAX * ml_p[i],
                         f"{model_name} trained on confirmed fraud cases rates this "
                         f"{ml_p[i]:.0%} likely to be fraud", "model"))
        t_reasons.append(rs)
    T = np.array([100 * noisy_or(rs) for rs in t_reasons])
    txn = tx.merge(feats.drop(columns=["reasons"]), on="txn_id")
    txn["ml_probability"] = np.round(ml_p, 4)
    txn["anomaly_score"] = np.round(an["anomaly_score"], 4)
    txn["anomaly_pct"] = np.round(an["anomaly_pct"], 2)
    txn["is_anomaly"] = an["is_anomaly"]
    txn["lof_score"] = np.round(an["lof_score"], 4)
    txn["transaction_risk"] = np.round(T, 1)
    tick("transaction risk")

    # ---------------- Graph, rings, Fraud DNA ----------------------------------------
    g, mules = build_graph(tx, accounts, merchants)
    raw_rings = find_rings(g, mules)
    tick("graph + ring detection")
    evidence_all = set().union(*(set(r["evidence_txns"]) for r in raw_rings)) if raw_rings else set()
    vectors, radar = fingerprints(txn, evidence_all)
    z = standardise(vectors)
    rings = annotate(raw_rings, g, mules, txn, merchants, z)
    tick("temporal + Fraud DNA + rings")

    # ---------------- Network Risk -----------------------------------------------------
    ring_of = {m: r for r in rings for m in r["members"]}
    evidence_of = {x: r for r in rings for x in r["evidence_txns"]}
    controlled = {m: f"mule account in {ring_of[m]['ring_id']}" if m in ring_of else "mule account"
                  for m in mules}
    for r in rings:  # only accounts that themselves share a device with new accounts
        for m in r["synthetic_identities"]:
            controlled.setdefault(m, f"synthetic identity in {r['ring_id']} "
                                     f"(new account on a shared device)")
    P = account_projection(tx, accounts)
    max_t = txn.groupby("account_id")["transaction_risk"].max()
    seed = {a: (ring_of[a]["ring_risk"] if a in ring_of else (v if v >= 61 else 0.0))
            for a, v in max_t.items()}
    nbr = neighbour_risk(P, seed)
    N, n_ev, n_reasons = np.zeros(len(txn)), np.zeros(len(txn), bool), []
    for i, row in enumerate(txn[["txn_id", "account_id"]].itertuples(index=False)):
        rs, n, ev = [], 0.0, False
        ring = evidence_of.get(row.txn_id)
        if ring:
            n, ev = ring["ring_risk"], True
            rs.append(_r("RING_EVIDENCE", n / 100, f"part of the evidence linking "
                         f"{ring['ring_id']} ({ring['size']} accounts): "
                         f"{ring['evidence_breakdown'][0]['label'].lower()}", "network"))
        why = controlled.get(row.account_id)
        if why:
            r_risk = ring_of[row.account_id]["ring_risk"] if row.account_id in ring_of else 70.0
            if r_risk > n:
                n = r_risk
            ev = True
            rs.append(_r("CONTROLLED_ACCOUNT", r_risk / 100,
                         f"made by a fraudster-controlled account: {why}", "network"))
        member = ring_of.get(row.account_id)
        if member and not rs:
            n = 0.4 * member["ring_risk"]
            rs.append(_r("RING_MEMBER_ACCOUNT", n / 100, f"account is a member of "
                         f"{member['ring_id']}; this transaction is not part of the ring's "
                         f"evidence", "network"))
        risk_nb, who, rel = nbr.get(row.account_id, (0.0, "", ""))
        if risk_nb > 0 and not rs:
            n = min(40.0, 0.4 * risk_nb)
            rs.append(_r("CONNECTED_RISK", n / 100, f"linked to high-risk account {who} "
                         f"(via {rel.replace('_', ' ')}, risk {risk_nb:.0f})", "network"))
        N[i], n_ev[i] = n, ev
        n_reasons.append(rs)
    txn["network_risk"] = np.round(N, 1)
    txn["network_evidence"] = n_ev
    txn["ring_id"] = [evidence_of[x]["ring_id"] if x in evidence_of else
                      (ring_of[a]["ring_id"] if a in ring_of else "")
                      for x, a in zip(txn["txn_id"], txn["account_id"])]
    tick("network risk")

    # ---------------- Final risk, bands, actions ---------------------------------------
    fr = final_risk(txn["transaction_risk"], txn["behavioral_risk"], N, n_ev, weights)
    txn["blend"], txn["final_risk"], txn["guardrail"] = fr["blend"].to_numpy(), \
        fr["final_risk"].to_numpy(), fr["guardrail"].to_numpy()
    txn["band"] = bands(txn["final_risk"])
    txn["action"] = txn["band"].map(actions.txn_action)
    txn["status"] = txn["band"].map(actions.STATUS)
    all_reasons = [t + n for t, n in zip(t_reasons, n_reasons)]
    txn["reasons"] = [json.dumps(rs, ensure_ascii=False) for rs in all_reasons]
    name = merchants.set_index("merchant_id")["merchant_name"]
    txn["merchant_name"] = txn["merchant_id"].map(name).fillna("")
    ring_by_id = {r["ring_id"]: r for r in rings}
    txn["explanation"] = [
        narrative(row, rs, ring_by_id.get(row["ring_id"]))
        if row["final_risk"] >= 31 or row["amount_inr"] >= 50000 else
        f"{row['txn_id']} is LOW risk ({row['final_risk']:.0f}/100): consistent with this "
        f"account's normal behaviour."
        for row, rs in zip(txn[["txn_id", "final_risk", "band", "amount_inr", "b_amount",
                                "b_amount_ratio", "b_avg_amount", "b_device", "b_location",
                                "b_time", "city", "timestamp", "ml_probability", "ring_id"]]
                           .to_dict("records"), all_reasons)]
    tick("final risk + explanations")

    # ---------------- Account risk -------------------------------------------------------
    profiles = build_profiles(tx, merchants).set_index("account_id")
    acc_rows = []
    info = accounts.set_index("account_id")
    for acct, t in txn.groupby("account_id"):
        top = t.loc[t["final_risk"].idxmax()]
        reasons = []
        if top["final_risk"] >= 31:
            reasons.append(_r("HIGH_RISK_TRANSACTION", top["final_risk"] / 100,
                              f"{top['txn_id']} scored {top['final_risk']:.0f} ({top['band']})",
                              "transaction"))
        ring = ring_of.get(acct)
        if ring:
            reasons.append(_r("RING_MEMBER", ring["ring_risk"] / 100,
                              f"member of {ring['ring_id']} ({ring['size']} accounts, ring risk "
                              f"{ring['ring_risk']:.0f})", "network"))
        if acct in mules:
            m = mules[acct]
            reasons.append(_r("MULE", 0.7, f"received ₹{m['received']:,.0f} from {m['senders']} "
                                           f"accounts and passed {m['pass_through']:.0%} straight "
                                           f"to gift cards/crypto", "network"))
        if acct in controlled and acct not in mules:
            reasons.append(_r("CONTROLLED", 0.6, controlled[acct], "network"))
        max_b = float(t["behavioral_risk"].max())
        if max_b >= 60:
            reasons.append(_r("BEHAVIOR_DEVIATION", 0.15, f"behaviour deviated strongly from "
                                                          f"profile (max {max_b:.0f}/100)",
                              "behavioral"))
        risk_nb, who, rel = nbr.get(acct, (0.0, "", ""))
        if risk_nb >= 61 and not ring:
            reasons.append(_r("CONNECTED", 0.2, f"linked to high-risk account {who} via "
                                                f"{rel.replace('_', ' ')}", "network"))
        risk = min(99.0, round(100 * noisy_or(reasons), 1))
        neighbours = sorted(P.neighbors(acct), key=lambda x: -seed.get(x, 0)) if acct in P else []
        linked = [x for x in (ring["members"] if ring else []) if x != acct] + \
            [x for x in neighbours if seed.get(x, 0) >= 61 and (not ring or x not in ring["members"])]
        prof = profiles.loc[acct].to_dict() if acct in profiles.index else {}
        acc_rows.append({
            "account_id": acct,
            "customer_id": info["customer_id"].get(acct, ""),
            "account_type": info["account_type"].get(acct, "personal"),
            "home_city": info["home_city"].get(acct, ""),
            "signup_date": info["signup_date"].get(acct, ""),
            "account_risk": risk, "ring_id": ring["ring_id"] if ring else "",
            "is_mule": acct in mules, "n_transactions": len(t),
            "n_flagged": int((t["final_risk"] >= FLAG_THRESHOLD).sum()),
            "total_inr": round(float(t["amount_inr"].sum()), 2),
            "max_txn_risk": float(t["final_risk"].max()),
            "max_behavioral": max_b, "behavior_level": level(float(t["behavioral_risk"]
                                                                    .quantile(0.98))),
            "neighbour_risk": round(risk_nb, 1),
            "connected_accounts": ";".join(linked[:15]),
            "reasons": json.dumps(reasons, ensure_ascii=False),
            "explanation": "; ".join(f"{r['code']}: {r['evidence']}" for r in reasons)
                           or "no risk factors",
            **{k: v for k, v in prof.items() if k != "n_txns"},
        })
    acc = pd.DataFrame(acc_rows)
    acc["band"] = bands(acc["account_risk"])
    acc["action"] = acc["band"].map(actions.account_action)
    acc = acc.sort_values("account_risk", ascending=False).reset_index(drop=True)
    tick("account risk")

    radar = radar.reset_index().rename(columns={"index": "account_id"})
    dna = vectors.reset_index().rename(columns={"index": "account_id"}).merge(radar, on="account_id")
    return {
        "txn": txn, "acc": acc, "rings": rings, "dna": dna, "accounts": accounts,
        "merchants": merchants,
        "anomaly": {k: an.get(k) for k in ("model", "reference", "cols", "threshold")},
        "model": bundle,
        "summary": {"transactions": int(len(txn)), "accounts": int(len(acc)),
                    "rings": len(rings), "flagged": int((txn["final_risk"] >= FLAG_THRESHOLD).sum()),
                    "model": model_name, "weights": weights or RISK_WEIGHTS,
                    "period": [str(txn["timestamp"].min()), str(txn["timestamp"].max())],
                    "seconds": clock},
    }


def drop_heavy(txn):
    return txn.drop(columns=["burst_members", "struct_members"], errors="ignore")


def write_outputs(res, out=OUT):
    out.mkdir(exist_ok=True)
    txn = drop_heavy(res["txn"]).copy()
    txn["timestamp"] = txn["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    txn.to_csv(out / TXN_FILE.name, index=False, compression="gzip")
    res["acc"].to_csv(out / "account_scores.csv", index=False)
    (out / "rings.json").write_text(json.dumps(res["rings"], indent=2, default=str), "utf-8")
    res["dna"].round(4).to_csv(out / "fraud_dna.csv", index=False)
    (out / "run_summary.json").write_text(json.dumps(res["summary"], indent=2, default=str),
                                          "utf-8")
    MODELS.mkdir(exist_ok=True)
    if res["anomaly"].get("model") is not None:
        with open(MODELS / "anomaly_model.pkl", "wb") as f:
            pickle.dump(res["anomaly"], f)
    old = out / "transaction_scores.csv"
    if old.exists():
        old.unlink()


def run(verbose=False):
    tx, accounts, merchants = load()
    res = run_frames(tx, accounts, merchants, verbose=verbose)
    write_outputs(res)
    return res


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    res = run(verbose=True)
    txn, rings = res["txn"], res["rings"]
    print(f"scored {len(txn):,} transactions and {len(res['acc'])} accounts in "
          f"{time.time() - t0:.1f}s; {(txn['final_risk'] >= FLAG_THRESHOLD).sum()} flagged "
          f"(HIGH/CRITICAL), {len(rings)} rings")
    for r in rings:
        print(f"  {r['ring_id']}: {r['size']} accounts, ring risk {r['ring_risk']}, "
              f"{list(r['evidence_kinds'])}, mules {r['mules']}")
