"""Benchmark on the public PaySim dataset (Lopez-Rojas et al., 2016).

Source: Hugging Face mirror flwrlabs/fed-fraud-paysim-banks (CC BY 4.0), test split
(636k transactions). Downloaded on first run to data/external/ (not committed).

Sample: every fraud in the split plus a random sample of legitimate transactions, kept in
time order. Because legitimate rows are subsampled, precision is reported both on the
sample and re-weighted to PaySim's natural fraud rate; FPR and ROC-AUC do not depend on
the base rate.

Time split: earlier simulation hours train, later hours test (no future leakage).

Compared:
  A. Our system, zero-shot — the unchanged pipeline (rules, behaviour, graph, model
     trained on our synthetic data). Shows how it transfers to a different simulator.
  B. Random Forest on PaySim's native fields (amount, balances, balance-error terms,
     type) — the standard PaySim baseline.
  C. Random Forest on OUR engineered features (real-time + behavioural), retrained on
     PaySim — tests whether our feature engineering transfers.
  D. Hybrid: our features + PaySim balance features.

Writes eval/benchmark_paysim.json.   Run: python eval/benchmark_paysim.py [n_legit]
"""

import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fraud.data_loader import adapt_external  # noqa: E402
from fraud.fraud_model import FEATURE_COLS  # noqa: E402
from fraud.pipeline import run_frames  # noqa: E402

URL = ("https://huggingface.co/datasets/flwrlabs/fed-fraud-paysim-banks/resolve/main/"
       "data/test-00000-of-00001.parquet")
LOCAL = ROOT / "data" / "external" / "paysim_test.parquet"
N_LEGIT = int(sys.argv[1]) if len(sys.argv) > 1 else 24000
FLAG = 61
SEED = 0


def download():
    if LOCAL.exists():
        return
    LOCAL.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading PaySim test split → {LOCAL} …", flush=True)
    req = urllib.request.Request(URL, headers={"User-Agent": "FraudIntelligenceStudentProject/1.0"})
    with urllib.request.urlopen(req, timeout=300) as r, open(LOCAL, "wb") as f:
        f.write(r.read())


def balance_features(df):
    f = pd.DataFrame(index=df.index)
    f["amount"] = df["amount"]
    for c in ("oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"):
        f[c] = df[c]
    f["errorBalanceOrig"] = df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
    f["errorBalanceDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    f["hour"] = df["step"] % 24
    for t in ("TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"):
        f[f"type_{t}"] = (df["type"] == t).astype(int)
    return f


def metrics(y, score, flag, natural_rate, sample_rate):
    y, flag = np.asarray(y, bool), np.asarray(flag, bool)
    tp, fp = int((flag & y).sum()), int((flag & ~y).sum())
    fn, tn = int((~flag & y).sum()), int((~flag & ~y).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    # precision at PaySim's natural fraud rate (legit rows were subsampled)
    w = (1 - natural_rate) / natural_rate * sample_rate / (1 - sample_rate)
    p_nat = tp / (tp + fp * w) if tp + fp else 0.0
    try:
        auc = float(roc_auc_score(y, score))
        ap = float(average_precision_score(y, score))
    except ValueError:
        auc = ap = None
    return {"precision_sample": round(p, 4), "precision_natural_rate": round(p_nat, 4),
            "recall": round(r, 4), "f1_sample": round(2 * p * r / (p + r), 4) if p + r else 0.0,
            "fpr": round(fpr, 5), "roc_auc": None if auc is None else round(auc, 4),
            "average_precision": None if ap is None else round(ap, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def pipeline_features(part):
    tx, accounts, merchants, labels = adapt_external(part.reset_index(drop=True))
    res = run_frames(tx, accounts, merchants)
    t = res["txn"].merge(labels, on="txn_id")
    t["row"] = t["txn_id"].str[1:].astype(int)
    return t.sort_values("row").reset_index(drop=True)


def rf():
    return RandomForestClassifier(n_estimators=200, min_samples_leaf=3, n_jobs=-1,
                                  class_weight="balanced_subsample", random_state=SEED)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    download()
    full = pd.read_parquet(LOCAL)
    natural_rate = float(full["isFraud"].mean())
    fraud = full[full["isFraud"] == 1]
    legit = full[full["isFraud"] == 0].sample(n=min(N_LEGIT, int((full["isFraud"] == 0).sum())),
                                              random_state=SEED)
    data = pd.concat([fraud, legit]).sort_values("step", kind="stable").reset_index(drop=True)
    sample_rate = float(data["isFraud"].mean())
    cut = int(data["step"].median())
    train, test = data[data["step"] <= cut], data[data["step"] > cut]
    print(f"PaySim split: {len(full):,} rows, natural fraud rate {natural_rate:.4%}; sample "
          f"{len(data):,} rows ({int(data['isFraud'].sum())} fraud, {sample_rate:.2%}); "
          f"train steps ≤{cut}: {len(train):,}, test: {len(test):,}", flush=True)

    print("running the full pipeline on train and test parts …", flush=True)
    ptrain, ptest = pipeline_features(train), pipeline_features(test)
    ytr, yte = ptrain["is_fraud"].to_numpy(), ptest["is_fraud"].to_numpy()

    results = {}
    results["A. Our system, zero-shot (final risk ≥ 61)"] = metrics(
        yte, ptest["final_risk"], ptest["final_risk"] >= FLAG, natural_rate, sample_rate)
    results["A2. Our transaction-risk score only, zero-shot"] = metrics(
        yte, ptest["transaction_risk"], ptest["transaction_risk"] >= FLAG, natural_rate,
        sample_rate)

    btr, bte = balance_features(train.reset_index(drop=True)), balance_features(test.reset_index(drop=True))
    m = rf().fit(btr, train["isFraud"])
    p = m.predict_proba(bte)[:, 1]
    results["B. Random Forest on PaySim balance fields (retrained)"] = metrics(
        test["isFraud"], p, p >= 0.5, natural_rate, sample_rate)

    ours = [c for c in FEATURE_COLS if c in ptrain.columns]
    m = rf().fit(ptrain[ours], ytr)
    p = m.predict_proba(ptest[ours])[:, 1]
    results["C. Random Forest on our engineered features (retrained)"] = metrics(
        yte, p, p >= 0.5, natural_rate, sample_rate)

    htr = pd.concat([ptrain[ours].reset_index(drop=True), btr.reset_index(drop=True)], axis=1)
    hte = pd.concat([ptest[ours].reset_index(drop=True), bte.reset_index(drop=True)], axis=1)
    m = rf().fit(htr, ytr)
    p = m.predict_proba(hte)[:, 1]
    results["D. Hybrid: our features + balance fields (retrained)"] = metrics(
        yte, p, p >= 0.5, natural_rate, sample_rate)
    imp = sorted(zip(htr.columns, m.feature_importances_), key=lambda x: -x[1])[:10]

    out = {"source": {"dataset": "PaySim (Lopez-Rojas, Elmir & Axelsson, 2016)",
                      "mirror": "huggingface.co/datasets/flwrlabs/fed-fraud-paysim-banks",
                      "license": "CC BY 4.0", "split": "test"},
           "design": {"rows_in_split": len(full), "natural_fraud_rate": round(natural_rate, 6),
                      "sample_rows": len(data), "sample_fraud": int(data["isFraud"].sum()),
                      "sample_fraud_rate": round(sample_rate, 5), "time_split_step": cut,
                      "train_rows": len(train), "test_rows": len(test),
                      "test_fraud": int(test["isFraud"].sum()), "legit_sample_seed": SEED},
           "results": results, "hybrid_top_features": [[k, round(float(v), 4)] for k, v in imp],
           "rings_found_zero_shot": "see run log", "seconds": round(time.time() - t0, 1)}
    (ROOT / "eval" / "benchmark_paysim.json").write_text(json.dumps(out, indent=2), "utf-8")
    for k, v in results.items():
        print(f"{k:58s} recall {v['recall']:.3f}  precision(sample) {v['precision_sample']:.3f}  "
              f"precision(natural) {v['precision_natural_rate']:.3f}  FPR {v['fpr']:.4f}  "
              f"AUC {v['roc_auc']}")
    print("hybrid top features:", imp[:6])
    print(f"done in {time.time() - t0:.0f}s → eval/benchmark_paysim.json")


if __name__ == "__main__":
    main()
