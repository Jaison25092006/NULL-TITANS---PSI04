"""Unsupervised anomaly detection: learn what "normal" looks like from the baseline period
(no labels), then measure how unusual every transaction is.

  Isolation Forest   main detector; flags the most unusual 0.5% relative to the baseline
  Local Outlier Factor (novelty mode)   second opinion, reported in Model Performance
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from .config import BASELINE_DAYS

ANOMALY_WEIGHT = 0.20
ANOMALY_QUANTILE = 0.995


def feature_columns(df):
    return [c for c in df.columns if c.startswith("f_") or
            (c.startswith("b_") and c not in ("b_avg_amount", "b_amount_ratio", "b_history"))]


def baseline_mask(timestamps):
    start, end = timestamps.min(), timestamps.max()
    days = min(BASELINE_DAYS, max((end - start).days / 3, 0))
    mask = (timestamps < start + pd.Timedelta(days=days)).to_numpy()
    return mask if mask.sum() >= 50 else np.ones(len(timestamps), bool)


def fit_detect(features, timestamps, seed=0):
    """Returns dict with anomaly_score, anomaly_pct (0-100 vs baseline), is_anomaly,
    lof_score, and the fitted models."""
    cols = feature_columns(features)
    X = features[cols].to_numpy(float)
    n = len(X)
    out = {"cols": cols}
    if n < 10:  # too little data to model "normal"
        zeros = np.zeros(n)
        out.update(anomaly_score=zeros, anomaly_pct=zeros, is_anomaly=zeros.astype(bool),
                   lof_score=zeros, model=None)
        return out
    base = baseline_mask(timestamps)
    model = IsolationForest(n_estimators=200, random_state=seed).fit(X[base])
    raw = -model.score_samples(X)
    ref = np.sort(raw[base])
    out["anomaly_score"] = raw
    out["anomaly_pct"] = 100 * np.searchsorted(ref, raw, side="right") / len(ref)
    out["is_anomaly"] = raw > np.quantile(ref, ANOMALY_QUANTILE)
    out["model"] = model
    out["threshold"] = float(np.quantile(ref, ANOMALY_QUANTILE))
    out["reference"] = ref

    scaler = StandardScaler().fit(X[base])
    idx = np.flatnonzero(base)
    rng = np.random.default_rng(seed)
    sample = rng.choice(idx, size=min(4000, len(idx)), replace=False)
    lof = LocalOutlierFactor(n_neighbors=min(35, len(sample) - 1), novelty=True)
    lof.fit(scaler.transform(X[sample]))
    out["lof_score"] = -lof.score_samples(scaler.transform(X))
    return out


def anomaly_pct(model, reference, X):
    """Percentile of new rows' anomaly scores against the baseline (live scoring)."""
    raw = -model.score_samples(X)
    return raw, 100 * np.searchsorted(reference, raw, side="right") / len(reference)
