"""Data access for the dashboard: load results once, apply the analyst's risk weights,
and hand pages consistent DataFrames. Pages must treat returned frames as read-only."""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from fraud import action_engine as actions
from fraud.config import FLAG_THRESHOLD, RISK_WEIGHTS
from fraud.data_loader import load as load_raw
from fraud.fraud_dna import standardise
from fraud.risk_engine import bands, final_risk

ROOT = Path(__file__).resolve().parents[1]
OUT, EVAL, MODELS = ROOT / "outputs", ROOT / "eval", ROOT / "models"
BOOL_COLS = ("is_anomaly", "network_evidence", "is_mule")


def _bools(df):
    for c in BOOL_COLS:
        if c in df and df[c].dtype == object:
            df[c] = df[c].astype(str).eq("True")
    return df


def prepare(res, labels=None, account_labels=None, source="Built-in synthetic dataset"):
    txn = res["txn"].copy()
    txn["timestamp"] = pd.to_datetime(txn["timestamp"])
    txn = _bools(txn).sort_values("timestamp", kind="stable").reset_index(drop=True)
    acc = _bools(res["acc"].copy())
    for c in ("connected_accounts", "ring_id", "typical_devices", "typical_locations",
              "normal_beneficiaries", "typical_merchants"):
        if c in acc:
            acc[c] = acc[c].fillna("").astype(str)
    # factor of all non-transaction account reasons, so account risk can follow the weights
    other = []
    for rs in acc["reasons"]:
        p = 1.0
        for r in (json.loads(rs) if isinstance(rs, str) and rs else rs or []):
            if r["code"] != "HIGH_RISK_TRANSACTION":
                p *= 1 - r["weight"]
        other.append(p)
    acc["_other_factor"] = other
    dna = res["dna"].set_index("account_id") if "account_id" in res["dna"] else res["dna"]
    radar_cols = ["Amount", "Time", "Device", "Location", "Merchant", "Velocity", "Network",
                  "Sequence"]
    vec = dna.drop(columns=[c for c in radar_cols if c in dna])
    return {
        "txn": txn, "acc": acc, "rings": res["rings"],
        "ring_by_id": {r["ring_id"]: r for r in res["rings"]},
        "dna_z": standardise(vec), "dna_radar": dna[[c for c in radar_cols if c in dna]],
        "accounts": res["accounts"], "merchants": res["merchants"],
        "anomaly": res.get("anomaly") or {}, "model": res.get("model"),
        "labels": labels, "account_labels": account_labels, "source": source,
        "summary": res.get("summary", {}),
    }


@st.cache_resource(show_spinner="Loading fraud intelligence results…")
def load_default():
    if not (OUT / "transaction_scores.csv.gz").exists():
        from fraud.pipeline import run
        run()
    txn = pd.read_csv(OUT / "transaction_scores.csv.gz", keep_default_na=False,
                      dtype={"device_id": str, "payee_account_id": str, "merchant_id": str,
                             "ring_id": str, "guardrail": str, "ip_address": str})
    acc = pd.read_csv(OUT / "account_scores.csv", keep_default_na=False)
    rings = json.loads((OUT / "rings.json").read_text("utf-8"))
    dna = pd.read_csv(OUT / "fraud_dna.csv")
    _, accounts, merchants = load_raw()
    anomaly = {}
    try:
        with open(MODELS / "anomaly_model.pkl", "rb") as f:
            anomaly = pickle.load(f)
    except Exception:
        anomaly = {}
    from fraud.fraud_model import load_model
    bundle = load_model(auto_train=True)  # retrains (~40s) only if the pickle cannot be read
    summary = {}
    if (OUT / "run_summary.json").exists():
        summary = json.loads((OUT / "run_summary.json").read_text("utf-8"))
    labels = account_labels = None
    if (EVAL / "transaction_labels.csv").exists():
        labels = pd.read_csv(EVAL / "transaction_labels.csv", keep_default_na=False)
        labels["is_fraud"] = labels["is_fraud"].astype(str).eq("True")
        if not set(labels["txn_id"]) >= set(txn["txn_id"].head(100)):
            labels = None  # labels belong to another dataset
    if (EVAL / "account_labels.csv").exists():
        account_labels = pd.read_csv(EVAL / "account_labels.csv", keep_default_na=False)
        account_labels["is_fraud"] = account_labels["is_fraud"].astype(str).eq("True")
    res = {"txn": txn, "acc": acc, "rings": rings, "dna": dna, "accounts": accounts,
           "merchants": merchants, "anomaly": anomaly, "model": bundle, "summary": summary}
    return prepare(res, labels, account_labels)


def data():
    """The active dataset (an upload if the analyst provided one, else the default)."""
    up = st.session_state.get("upload_data")
    return up if up is not None else load_default()


def weights():
    w = st.session_state.get("weights")
    return w if w else dict(RISK_WEIGHTS)


@st.cache_resource(max_entries=16, show_spinner=False)
def _weighted(key, wt, wb, wn):
    d = data_by_key(key)
    txn = d["txn"]
    fr = final_risk(txn["transaction_risk"], txn["behavioral_risk"], txn["network_risk"],
                    txn["network_evidence"], {"transaction": wt, "behavioral": wb, "network": wn})
    out = txn.copy()
    out["blend"], out["final_risk"], out["guardrail"] = (fr["blend"].to_numpy(),
                                                         fr["final_risk"].to_numpy(),
                                                         fr["guardrail"].to_numpy())
    out["band"] = bands(out["final_risk"])
    out["action"] = out["band"].map(actions.txn_action)
    out["status"] = out["band"].map(actions.STATUS)
    acc = d["acc"].copy()
    top = out.groupby("account_id")["final_risk"].max()
    t = acc["account_id"].map(top).fillna(0).to_numpy() / 100
    t = np.where(t >= 0.31, t, 0.0)
    acc["account_risk"] = np.minimum(99.0, np.round(100 * (1 - (1 - t) * acc["_other_factor"]), 1))
    acc["max_txn_risk"] = acc["account_id"].map(top).fillna(0)
    acc["n_flagged"] = acc["account_id"].map(
        out[out["final_risk"] >= FLAG_THRESHOLD].groupby("account_id").size()).fillna(0).astype(int)
    acc["band"] = bands(acc["account_risk"])
    acc["action"] = acc["band"].map(actions.account_action)
    acc = acc.sort_values("account_risk", ascending=False).reset_index(drop=True)
    return out, acc


def data_by_key(key):
    return data()


def txn():
    d, w = data(), weights()
    if w == dict(RISK_WEIGHTS) and st.session_state.get("upload_data") is None:
        return d["txn"]
    return _weighted(_key(), w["transaction"], w["behavioral"], w["network"])[0]


def acc():
    d, w = data(), weights()
    if w == dict(RISK_WEIGHTS) and st.session_state.get("upload_data") is None:
        return d["acc"]
    return _weighted(_key(), w["transaction"], w["behavioral"], w["network"])[1]


def _key():
    return st.session_state.get("upload_key", "default")


def rings():
    return data()["rings"]


def ring(rid):
    return data()["ring_by_id"].get(rid)


def account_row(account_id):
    a = acc()
    hit = a[a["account_id"] == account_id]
    return hit.iloc[0] if len(hit) else None


def txn_row(txn_id):
    t = txn()
    hit = t[t["txn_id"] == txn_id]
    return hit.iloc[0] if len(hit) else None


@st.cache_resource(show_spinner="Warming up the live scorer (replaying history)…")
def _live(key):
    from fraud.live import LiveScorer
    d = data()
    return LiveScorer(d["txn"], d["accounts"], d["merchants"], d["acc"], d["rings"],
                      bundle=d["model"], anomaly=d["anomaly"])


def live_scorer():
    return _live(_key())


@st.cache_resource(show_spinner="Building the entity graph…")
def _graphs(key, wkey):
    from fraud.graph_analysis import account_projection, build_entity_graph
    d = data()
    G = build_entity_graph(txn(), acc(), d["merchants"], d["rings"])
    P = account_projection(d["txn"], d["accounts"])
    return G, P


def graphs():
    w = weights()
    return _graphs(_key(), tuple(sorted(w.items())))


def robustness():
    p = EVAL / "robustness.json"
    return json.loads(p.read_text("utf-8")) if p.exists() else []


def model_card():
    p = MODELS / "model_card.json"
    return json.loads(p.read_text("utf-8")) if p.exists() else {}
