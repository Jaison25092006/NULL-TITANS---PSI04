"""Run the unchanged detector on datasets generated with other seeds (in memory; data/
and outputs/ are not touched). The ML model is trained on seeds 11 and 13, so none of
these seeds were seen in training; only seed 2026 was used during development.

Writes eval/robustness.json.   Run: python eval/robustness.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_gen.generate import generate  # noqa: E402
from eval.evaluate import scorecard  # noqa: E402
from fraud.data_loader import normalize  # noqa: E402
from fraud.pipeline import run_frames  # noqa: E402

SEEDS = [2026, 7, 42, 99, 123, 314, 777]


def main():
    rows = []
    for seed in SEEDS:
        tx, accounts, merchants, tl, al = generate(seed)
        tx, accounts, merchants = normalize(tx, accounts, merchants)
        res = run_frames(tx, accounts, merchants)
        tl["is_fraud"], al["is_fraud"] = tl["is_fraud"].astype(str), al["is_fraud"].astype(str)
        c = scorecard(res["txn"], res["acc"], res["rings"], tl, al)
        t = c["transactions"]["final (all layers)"]
        a = c["accounts"]["flagged (risk >= 61)"]
        rows.append({
            "seed": seed, "role": "development" if seed == 2026 else "unseen",
            "transactions": len(tx), "txn_precision": t["precision"], "txn_recall": t["recall"],
            "txn_f1": t["f1"], "txn_false_alarms": t["fp"], "txn_fpr": t["fpr"],
            "roc_auc": c["roc_auc"]["final"],
            "fraud_accounts_caught": f"{a['tp']}/{a['tp'] + a['fn']}",
            "innocent_accounts_flagged": a["fp"],
            "rings_exact": f"{sum(r['jaccard'] == 1 for r in c['rings'])}/{len(c['rings'])}",
            "lookalike_false_alarms": sum(v["flagged"] for v in
                                          c["false_alarms_on_benign_lookalikes"].values()),
        })
        r = rows[-1]
        print(f"seed {seed:>4} ({r['role']:11}): precision {r['txn_precision']:.3f}  recall "
              f"{r['txn_recall']:.3f}  false alarms {r['txn_false_alarms']}  accounts "
              f"{r['fraud_accounts_caught']} (+{r['innocent_accounts_flagged']} innocent)  rings "
              f"{r['rings_exact']}  look-alike FPs {r['lookalike_false_alarms']}", flush=True)
    (ROOT / "eval" / "robustness.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
