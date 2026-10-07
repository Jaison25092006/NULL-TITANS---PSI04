"""Score the detector's outputs against the hidden labels.

Run after `python -m fraud.pipeline`. Prints a scorecard and writes eval/scorecard.json.
Flagged = Final Risk >= 61 (HIGH or CRITICAL: step-up authentication or block).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT, EVAL = ROOT / "outputs", ROOT / "eval"
FLAG = 61


def prf(flagged, truth):
    flagged, truth = np.asarray(flagged, bool), np.asarray(truth, bool)
    tp = int((flagged & truth).sum())
    fp = int((flagged & ~truth).sum())
    fn = int((~flagged & truth).sum())
    tn = int((~flagged & ~truth).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4),
            "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
            "fpr": round(fp / (fp + tn), 6) if fp + tn else 0.0,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def auc(truth, score):
    try:
        return round(float(roc_auc_score(truth, score)), 4)
    except ValueError:
        return None


def scorecard(txn, acc, rings, tl, al):
    t = txn.merge(tl, on="txn_id")
    a = acc.merge(al, on="account_id")
    truth = t["is_fraud"].astype(str).eq("True").to_numpy()
    a_truth = a["is_fraud"].astype(str).eq("True").to_numpy()
    final = t["final_risk"].to_numpy()
    methods = {
        "final (all layers)": final >= FLAG,
        "transaction risk only": t["transaction_risk"] >= FLAG,
        "real-time rules only": t["realtime_risk"] >= 0.5,
        "ML model only (p >= 0.5)": t["ml_probability"] >= 0.5,
        "Isolation Forest only": t["is_anomaly"].astype(str).eq("True"),
        "behavioral deviation only": t["behavioral_risk"] >= FLAG,
        "naive: flag top 1% amounts": t["amount_inr"] >= t["amount_inr"].quantile(0.99),
    }
    card = {
        "threshold": FLAG,
        "transactions": {k: prf(v, truth) for k, v in methods.items()},
        "roc_auc": {"final": auc(truth, final), "transaction": auc(truth, t["transaction_risk"]),
                    "behavioral": auc(truth, t["behavioral_risk"]),
                    "network": auc(truth, t["network_risk"]),
                    "ml_model": auc(truth, t["ml_probability"]),
                    "isolation_forest": auc(truth, t["anomaly_score"]),
                    "lof": auc(truth, t["lof_score"]), "amount": auc(truth, t["amount_inr"])},
        "average_precision": round(float(average_precision_score(truth, final)), 4),
        "confusion_matrix": confusion_matrix(truth, final >= FLAG, labels=[False, True]).tolist(),
        "recall_by_pattern": {p: round(float((g["final_risk"] >= FLAG).mean()), 3)
                              for p, g in t[truth].groupby("pattern")},
        "false_alarms_on_benign_lookalikes": {
            tag: {"transactions": len(g), "flagged": int((g["final_risk"] >= FLAG).sum())}
            for tag, g in t[~truth & (t["benign_tag"] != "")].groupby("benign_tag")},
        "accounts": {"flagged (risk >= 61)": prf(a["account_risk"] >= FLAG, a_truth),
                     "roc_auc": auc(a_truth, a["account_risk"])},
        "rings": [],
    }
    true_rings = al[al["ring_id"] != ""].groupby("ring_id")["account_id"].apply(set).to_dict()
    for rid, members in true_rings.items():
        best = max(rings, key=lambda r: len(members & set(r["members"])) /
                   len(members | set(r["members"])), default=None)
        jac = len(members & set(best["members"])) / len(members | set(best["members"])) if best else 0
        card["rings"].append({"true_ring": rid, "size": len(members),
                              "matched": best["ring_id"] if best and jac > 0 else None,
                              "jaccard": round(jac, 3)})
    return card


def main():
    txn = pd.read_csv(OUT / "transaction_scores.csv.gz", keep_default_na=False)
    acc = pd.read_csv(OUT / "account_scores.csv", keep_default_na=False)
    rings = json.loads((OUT / "rings.json").read_text("utf-8"))
    tl = pd.read_csv(EVAL / "transaction_labels.csv", keep_default_na=False)
    al = pd.read_csv(EVAL / "account_labels.csv", keep_default_na=False)
    card = scorecard(txn, acc, rings, tl, al)
    (EVAL / "scorecard.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    print(json.dumps(card, indent=2))


if __name__ == "__main__":
    main()
