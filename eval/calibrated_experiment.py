"""Experiment: run the unchanged detector on datasets calibrated to web-scraped statistics.

The generator's everyday payment amounts are scaled to the real average UPI ticket size
(scraped from Wikipedia's NPCI-sourced table) and fraud victims are placed according to
scraped NCRB state crime rates (scraping/calibrate.py). If the detector depended on the
exact synthetic amounts or cities it was developed on, this would expose it.

Writes eval/calibrated_experiment.json.   Run: python eval/calibrated_experiment.py
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

SEEDS = [2026, 7, 42]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    cal = json.loads((ROOT / "data" / "scraped" / "calibration.json").read_text("utf-8"))
    rows = []
    for seed in SEEDS:
        tx, accounts, merchants, tl, al = generate(seed, profile="calibrated")
        every = tx.merge(tl, on="txn_id").merge(accounts[["account_id", "account_type"]],
                                                on="account_id")
        every = every[(every["account_type"] == "personal") & (~every["is_fraud"]) &
                      (every["benign_tag"] == "") & (every["txn_type"] == "purchase")]
        victims = al[al["role"].isin(["compromised", "structurer"])].merge(
            accounts[["account_id", "home_city"]], on="account_id")
        tx, accounts, merchants = normalize(tx, accounts, merchants)
        res = run_frames(tx, accounts, merchants)
        tl["is_fraud"], al["is_fraud"] = tl["is_fraud"].astype(str), al["is_fraud"].astype(str)
        c = scorecard(res["txn"], res["acc"], res["rings"], tl, al)
        t = c["transactions"]["final (all layers)"]
        a = c["accounts"]["flagged (risk >= 61)"]
        rows.append({
            "seed": seed, "transactions": len(tx),
            "everyday_mean_inr": round(float(every["amount_inr"].mean()), 2),
            "target_avg_ticket_inr": cal["target_avg_ticket_inr"],
            "victim_cities": victims["home_city"].value_counts().to_dict(),
            "txn_precision": t["precision"], "txn_recall": t["recall"], "txn_f1": t["f1"],
            "txn_false_alarms": t["fp"], "txn_fpr": t["fpr"], "roc_auc": c["roc_auc"]["final"],
            "fraud_accounts_caught": f"{a['tp']}/{a['tp'] + a['fn']}",
            "innocent_accounts_flagged": a["fp"],
            "rings_exact": f"{sum(r['jaccard'] == 1 for r in c['rings'])}/{len(c['rings'])}",
        })
        r = rows[-1]
        print(f"seed {seed:>4}: everyday mean ₹{r['everyday_mean_inr']:,.0f} (target "
              f"₹{r['target_avg_ticket_inr']:,.0f})  precision {r['txn_precision']:.3f}  recall "
              f"{r['txn_recall']:.3f}  FPs {r['txn_false_alarms']}  accounts "
              f"{r['fraud_accounts_caught']} (+{r['innocent_accounts_flagged']})  rings "
              f"{r['rings_exact']}  victims {r['victim_cities']}", flush=True)
    out = {"calibration": {k: cal[k] for k in ("amount_scale", "target_avg_ticket_inr",
                                               "target_years", "city_risk_weight")},
           "runs": rows}
    (ROOT / "eval" / "calibrated_experiment.json").write_text(json.dumps(out, indent=2), "utf-8")


if __name__ == "__main__":
    main()
