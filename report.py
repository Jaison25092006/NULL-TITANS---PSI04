"""Print a readable summary of the detector's output.

Run: python -m fraud.pipeline && python eval/evaluate.py && python report.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
LINE = "=" * 90


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    acc = pd.read_csv(OUT / "account_scores.csv", keep_default_na=False)
    txn = pd.read_csv(OUT / "transaction_scores.csv.gz", keep_default_na=False)
    rings = json.loads((OUT / "rings.json").read_text("utf-8"))

    print(f"{LINE}\nFRAUD RINGS DETECTED\n{LINE}")
    for r in rings:
        print(f"\n{r['ring_id']}  |  {r['size']} accounts  |  ring risk {r['ring_risk']:.0f}/100 "
              f"({r['band']})  |  ₹{r['amount_inr']:,.0f} involved")
        print(f"  pattern  : {r['pattern']}")
        print(f"  members  : {', '.join(r['members'])}")
        print(f"  mule     : {', '.join(r['mules']) or '-'}")
        print(f"  devices  : {', '.join(r['devices'][:6]) or '-'}")
        print(f"  period   : {r['first_seen']}  →  {r['last_seen']}")
        for b in r["evidence_breakdown"]:
            print(f"    - {b['label']} (w={b['weight']}): {b['detail'][:110]}")
        print(f"  ACTION   : {r['recommended_action']}")

    print(f"\n{LINE}\nTOP 10 RISKIEST ACCOUNTS\n{LINE}")
    for a in acc.head(10).itertuples():
        print(f"{a.account_id}  risk {a.account_risk:5.1f}  {a.action:22}  {a.ring_id or '-':7}  "
              f"{a.n_flagged} flagged txns")

    print(f"\n{LINE}\nEXAMPLE TRANSACTION EXPLANATIONS\n{LINE}")
    for title, code in [("Fraud ring", "RING_EVIDENCE"), ("Account takeover", "NEW_DEVICE_ABROAD"),
                        ("Card testing", "CARD_TEST_CASHOUT"), ("Structuring", "STRUCTURING"),
                        ("Mule cash-out", "PASS_THROUGH")]:
        hits = txn[txn["reasons"].str.contains(f'"{code}"')]
        if hits.empty:
            continue
        t = hits.nlargest(1, "final_risk").iloc[0]
        print(f"\n[{title}] {t.txn_id}  {t.timestamp}  {t.account_id}  ₹{t.amount_inr:,.2f}")
        print(f"  T {t.transaction_risk:.0f} · B {t.behavioral_risk:.0f} · N {t.network_risk:.0f}"
              f"  →  final {t.final_risk:.0f} ({t.band})  →  {t.action}")
        print(f"  {t.explanation}")
    big = txn[(txn["amount_inr"] >= 200000) & (txn["final_risk"] < 31)]
    if not big.empty:
        t = big.nlargest(1, "amount_inr").iloc[0]
        print(f"\n[Big but legitimate] {t.txn_id}  {t.account_id}  ₹{t.amount_inr:,.2f}  "
              f"final {t.final_risk:.0f} → {t.action}")
        print(f"  {t.explanation}")

    card_path = ROOT / "eval" / "scorecard.json"
    if card_path.exists():
        c = json.loads(card_path.read_text("utf-8"))
        f = c["transactions"]["final (all layers)"]
        a = c["accounts"]["flagged (risk >= 61)"]
        n = c["transactions"]["naive: flag top 1% amounts"]
        print(f"\n{LINE}\nSCORECARD (flagged = final risk ≥ {c['threshold']})\n{LINE}")
        print(f"Transactions : precision {f['precision']:.1%}, recall {f['recall']:.1%}, F1 "
              f"{f['f1']:.1%}, FPR {f['fpr']:.4%}, ROC-AUC {c['roc_auc']['final']}")
        print(f"Accounts     : precision {a['precision']:.1%}, recall {a['recall']:.1%} "
              f"({a['tp']} of {a['tp'] + a['fn']} fraud accounts caught, {a['fp']} innocent flagged)")
        print("Rings        : " + ", ".join(f"{r['true_ring']} → {r['matched']} "
                                            f"({r['jaccard']:.0%})" for r in c["rings"]))
        print("Look-alikes  : " + ", ".join(f"{k} {v['flagged']}/{v['transactions']}" for k, v in
                                            c["false_alarms_on_benign_lookalikes"].items()))
        print(f"Naive 'flag the biggest amounts': precision {n['precision']:.0%}, "
              f"recall {n['recall']:.0%}")


if __name__ == "__main__":
    main()
