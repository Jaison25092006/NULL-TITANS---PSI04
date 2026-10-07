"""Supervised fraud model.

Trained on *separately generated* labelled datasets (seeds in config.TRAIN_SEEDS) that
stand in for a bank's history of confirmed fraud, then applied unchanged to the dataset
being analysed. The evaluation dataset's labels are never used for training or model
selection, so its scorecard is an honest out-of-sample result.

Three model families are compared (Logistic Regression, Random Forest, Gradient
Boosting); the one with the best validation average precision is used. Each prediction
is explained by per-feature contributions (SHAP when installed, otherwise an exact
baseline-substitution method that works for any model).

Run: python -m fraud.fraud_model     (writes models/fraud_model.pkl)
"""

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import TRAIN_SEEDS

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
MODEL_PATH = MODELS / "fraud_model.pkl"

FEATURES = {
    "f_log_amount": "Amount (log)", "f_amount_z": "Amount vs account history",
    "f_new_device": "New device", "f_hour": "Hour of day", "f_night": "Night-time",
    "f_velocity_1h": "Transactions in last hour", "f_account_age": "Account age",
    "f_foreign": "Foreign location", "f_risky_category": "Gift card / crypto",
    "f_transfer": "Is a transfer", "f_device_accounts": "Accounts on this device",
    "f_payee_senders": "Payee's distinct senders (14d)", "f_payee_young": "Payee is a new account",
    "f_new_payee": "First transfer to payee", "f_pass_through": "Share of inflow passed on",
    "f_business": "Business account", "f_near_threshold": "Just under ₹50k threshold",
    "b_amount": "Amount deviation", "b_device": "Device deviation",
    "b_location": "Location deviation", "b_time": "Time deviation",
    "b_velocity": "Velocity deviation", "b_merchant": "Merchant deviation",
    "b_beneficiary": "Beneficiary deviation",
}
FEATURE_COLS = list(FEATURES)
ML_WEIGHT_MAX = 0.6   # the ML reason's weight is p x 0.6, so the model alone cannot block

try:  # optional
    import shap  # noqa: F401
    HAS_SHAP = True
except Exception:  # pragma: no cover
    HAS_SHAP = False


def candidates():
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, min_samples_leaf=4, class_weight="balanced_subsample",
            n_jobs=-1, random_state=0),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0,
            random_state=0),
    }


def training_frame(seed):
    """Generate a labelled dataset and compute the same features the pipeline uses."""
    from data_gen.generate import generate

    from .behavioral_analysis import score_behavior
    from .data_loader import normalize
    from .stream import score_stream

    tx, accounts, merchants, labels, _ = generate(seed)
    tx, accounts, merchants = normalize(tx, accounts, merchants)
    s = score_stream(tx, accounts, merchants)
    b, _ = score_behavior(tx, merchants)
    feats = s.merge(b, on="txn_id").merge(labels[["txn_id", "is_fraud"]], on="txn_id")
    feats["is_fraud"] = feats["is_fraud"].astype(str).eq("True")
    return feats[["txn_id"] + FEATURE_COLS + ["is_fraud"]]


def _metrics(y, p, thr=0.5):
    pred = p >= thr
    return {"precision": round(float(precision_score(y, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y, p)), 4),
            "average_precision": round(float(average_precision_score(y, p)), 4)}


def train(seeds=TRAIN_SEEDS, verbose=True):
    t0 = time.time()
    frames = [training_frame(s) for s in seeds]
    train_df, val_df = frames[0], frames[-1]
    Xtr, ytr = train_df[FEATURE_COLS].to_numpy(float), train_df["is_fraud"].to_numpy()
    Xva, yva = val_df[FEATURE_COLS].to_numpy(float), val_df["is_fraud"].to_numpy()
    comparison = {}
    for name, model in candidates().items():
        model.fit(Xtr, ytr)
        comparison[name] = _metrics(yva, model.predict_proba(Xva)[:, 1])
        if verbose:
            print(f"  {name:20s} validation {comparison[name]}")
    best = max(comparison, key=lambda k: comparison[k]["average_precision"])
    full = pd.concat(frames, ignore_index=True)
    X, y = full[FEATURE_COLS].to_numpy(float), full["is_fraud"].to_numpy()
    model = candidates()[best].fit(X, y)
    # global importance: permutation importance on the validation set (refit-free)
    model_val = candidates()[best].fit(Xtr, ytr)
    sub = np.random.default_rng(0).choice(len(Xva), size=min(12000, len(Xva)), replace=False)
    pi = permutation_importance(model_val, Xva[sub], yva[sub], scoring="average_precision",
                                n_repeats=3, random_state=0, n_jobs=-1)
    importance = sorted(((FEATURE_COLS[i], float(pi.importances_mean[i]))
                         for i in range(len(FEATURE_COLS))), key=lambda x: -x[1])
    bundle = {
        "model": model, "name": best, "features": FEATURE_COLS,
        "baseline": np.median(X[~y], axis=0),  # a typical legitimate transaction
        "comparison": comparison, "importance": importance,
        "train_seeds": list(seeds), "train_rows": int(len(X)), "train_fraud": int(y.sum()),
        "sklearn": __import__("sklearn").__version__,
    }
    MODELS.mkdir(exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    card = {k: v for k, v in bundle.items() if k not in ("model", "baseline")}
    (MODELS / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    if verbose:
        print(f"selected {best}; trained on {len(X):,} rows ({int(y.sum())} fraud) "
              f"in {time.time() - t0:.0f}s -> {MODEL_PATH}")
    return bundle


def load_model(auto_train=True):
    """Load the trained model; retrain if missing or unreadable (e.g. sklearn upgrade)."""
    try:
        with open(MODEL_PATH, "rb") as f:
            bundle = pickle.load(f)
        if bundle.get("features") == FEATURE_COLS:
            return bundle
    except Exception:
        pass
    if not auto_train:
        return None
    try:
        return train(verbose=False)
    except Exception:
        return None


def predict(bundle, feats):
    if bundle is None or feats.empty:
        return np.zeros(len(feats))
    X = feats.reindex(columns=bundle["features"], fill_value=0).to_numpy(float)
    return bundle["model"].predict_proba(X)[:, 1]


def contributions(bundle, row):
    """Per-feature contribution to one prediction, as [(feature, label, value, delta)],
    most influential first. delta is in probability points."""
    if bundle is None:
        return []
    x = np.array([[float(row.get(c, 0) or 0) for c in bundle["features"]]])
    model = bundle["model"]
    if HAS_SHAP and bundle["name"] != "Logistic Regression":
        try:
            import shap
            sv = shap.TreeExplainer(model).shap_values(x)
            vals = np.asarray(sv[1] if isinstance(sv, list) else sv).reshape(-1)[:x.shape[1]]
            return sorted(((c, FEATURES[c], x[0, i], float(vals[i]))
                           for i, c in enumerate(bundle["features"])), key=lambda r: -abs(r[3]))
        except Exception:
            pass
    # Baseline substitution: how much does the prediction drop if this feature took the
    # value of a typical legitimate transaction instead?
    p = model.predict_proba(x)[0, 1]
    probes = np.repeat(x, x.shape[1], axis=0)
    for i in range(x.shape[1]):
        probes[i, i] = bundle["baseline"][i]
    drops = p - model.predict_proba(probes)[:, 1]
    return sorted(((c, FEATURES[c], x[0, i], float(drops[i]))
                   for i, c in enumerate(bundle["features"])), key=lambda r: -abs(r[3]))


if __name__ == "__main__":
    train()
