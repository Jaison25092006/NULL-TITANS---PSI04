"""Loading, cleaning and enrichment of transaction data.

The detector works on three tables:
  transactions  txn_id, timestamp, account_id, txn_type, channel, merchant_id,
                payee_account_id, amount_inr, device_id, ip_address, city, country
  accounts      account_id, customer_id, account_type, home_city, signup_date
  merchants     merchant_id, merchant_name, category, city

`load()` reads the bundled synthetic dataset. `adapt_external()` maps public fraud
datasets (PaySim / Kaggle "financial fraud", IEEE-CIS, or any CSV with recognisable
column names) onto the same schema, so the whole pipeline runs on them unchanged.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

TX_COLUMNS = ["txn_id", "timestamp", "account_id", "txn_type", "channel", "merchant_id",
              "payee_account_id", "amount_inr", "device_id", "ip_address", "city", "country"]
SPEC_COLUMNS = ["transaction_id", "account_id", "customer_id", "timestamp", "amount",
                "merchant_id", "merchant_category", "device_id", "location", "payment_channel",
                "beneficiary_id", "transaction_type", "account_age", "transaction_frequency",
                "historical_average_amount", "previous_transaction_count"]


def load(data_dir=DATA):
    data_dir = Path(data_dir)
    tx = pd.read_csv(data_dir / "transactions.csv", dtype=str, keep_default_na=False)
    accounts = pd.read_csv(data_dir / "accounts.csv", dtype=str, keep_default_na=False)
    merchants = pd.read_csv(data_dir / "merchants.csv", dtype=str, keep_default_na=False)
    return normalize(tx, accounts, merchants)


def normalize(tx, accounts=None, merchants=None):
    """Make any transaction table safe for the pipeline: required columns exist, types are
    right, missing values are blank strings, and every account / merchant is known."""
    tx = tx.copy()
    for col in TX_COLUMNS:
        if col not in tx.columns:
            tx[col] = ""
    tx = tx[TX_COLUMNS + [c for c in tx.columns if c not in TX_COLUMNS]]
    for col in TX_COLUMNS:
        if col not in ("timestamp", "amount_inr"):
            tx[col] = tx[col].fillna("").astype(str).str.strip()
            tx.loc[tx[col].str.lower().isin(["nan", "none", "null"]), col] = ""
    tx["timestamp"] = pd.to_datetime(tx["timestamp"], errors="coerce")
    tx["amount_inr"] = pd.to_numeric(tx["amount_inr"], errors="coerce").fillna(0.0).clip(lower=0)
    tx = tx[tx["timestamp"].notna() & (tx["account_id"] != "")]
    if tx.empty:
        raise ValueError("no usable transactions (need at least a timestamp and an account)")
    missing_ids = tx["txn_id"] == ""
    tx.loc[missing_ids, "txn_id"] = [f"TX{i:07d}" for i in np.flatnonzero(missing_ids.to_numpy())]
    tx["txn_type"] = np.where(tx["payee_account_id"] != "", "transfer",
                              np.where(tx["txn_type"] == "", "purchase", tx["txn_type"]))
    tx.loc[tx["channel"] == "", "channel"] = "app"
    tx.loc[tx["country"] == "", "country"] = "IN"
    tx.loc[tx["city"] == "", "city"] = "unknown"
    tx = tx.sort_values("timestamp", kind="stable").reset_index(drop=True)

    first_seen = tx.groupby("account_id")["timestamp"].min()
    known = set(accounts["account_id"]) if accounts is not None and len(accounts) else set()
    payees = set(tx["payee_account_id"]) - {""}
    unknown = sorted((set(first_seen.index) | payees) - known)
    if unknown:  # accounts we only know from transactions: assume established (1 year old)
        start = tx["timestamp"].min()
        extra = pd.DataFrame({
            "account_id": unknown, "customer_id": "", "account_type": "personal",
            "home_city": "unknown",
            "signup_date": [(first_seen.get(a, start) - pd.Timedelta(days=365)).date()
                            for a in unknown]})
        accounts = extra if accounts is None or not len(accounts) else \
            pd.concat([accounts, extra], ignore_index=True)
    accounts = accounts.copy()
    for col, default in (("customer_id", ""), ("account_type", "personal"),
                         ("home_city", "unknown")):
        if col not in accounts.columns:
            accounts[col] = default
        accounts[col] = accounts[col].fillna(default).astype(str)
    accounts.loc[accounts["customer_id"] == "", "customer_id"] = "C-" + accounts["account_id"]
    accounts["signup_date"] = pd.to_datetime(accounts["signup_date"], errors="coerce") \
        .fillna(tx["timestamp"].min() - pd.Timedelta(days=365)).dt.strftime("%Y-%m-%d")
    accounts = accounts.drop_duplicates("account_id").reset_index(drop=True)

    used = set(tx["merchant_id"]) - {""}
    merchants = merchants.copy() if merchants is not None and len(merchants) else \
        pd.DataFrame(columns=["merchant_id", "merchant_name", "category", "city"])
    new = sorted(used - set(merchants["merchant_id"]))
    if new:
        merchants = pd.concat([merchants, pd.DataFrame({
            "merchant_id": new, "merchant_name": new, "category": "general", "city": "online"})],
            ignore_index=True)
    for col in ("merchant_name", "category", "city"):
        merchants[col] = merchants[col].fillna("").astype(str)
    return tx, accounts, merchants


def enrich(tx, accounts, merchants):
    """Add the spec-named, derived columns. Everything is computed from the past only."""
    tx = tx.sort_values("timestamp", kind="stable").reset_index(drop=True)
    acc = accounts.set_index("account_id")
    category = merchants.set_index("merchant_id")["category"]
    signup = pd.to_datetime(tx["account_id"].map(acc["signup_date"]))
    out = tx.copy()
    out["transaction_id"] = tx["txn_id"]
    out["customer_id"] = tx["account_id"].map(acc["customer_id"]).fillna("")
    out["amount"] = tx["amount_inr"]
    out["merchant_category"] = np.where(tx["merchant_id"] != "",
                                        tx["merchant_id"].map(category).fillna("general"),
                                        "transfer")
    out["location"] = np.where(tx["country"] == "IN", tx["city"],
                               tx["city"] + ", " + tx["country"])
    out["payment_channel"] = tx["channel"]
    out["beneficiary_id"] = tx["payee_account_id"]
    out["transaction_type"] = tx["txn_type"]
    out["account_age"] = (tx["timestamp"] - signup).dt.days.fillna(365).astype(int)
    g = tx.groupby("account_id", sort=False)
    out["previous_transaction_count"] = g.cumcount()
    prev_sum = g["amount_inr"].cumsum() - tx["amount_inr"]
    out["historical_average_amount"] = (prev_sum / out["previous_transaction_count"]
                                        .replace(0, np.nan)).fillna(0).round(2)
    # transactions per day over the previous 7 days
    counts = np.zeros(len(tx))
    ts = tx["timestamp"].to_numpy()
    for _, idx in g.indices.items():
        t = ts[idx]
        lo = np.searchsorted(t, t - np.timedelta64(7, "D"), side="left")
        counts[idx] = np.arange(len(idx)) - lo
    out["transaction_frequency"] = (counts / 7).round(3)
    return out


# --------------------------------------------------------------------------- #
# External datasets
# --------------------------------------------------------------------------- #

CANDIDATES = {
    "txn_id": ["txn_id", "transaction_id", "transactionid", "trans_num", "id"],
    "timestamp": ["timestamp", "trans_date_trans_time", "datetime", "date", "time",
                  "transaction_time", "event_time"],
    "account_id": ["account_id", "account", "nameorig", "cc_num", "card_number", "customer",
                   "customer_id", "user_id", "sender", "card1"],
    "amount_inr": ["amount_inr", "amount", "amt", "transactionamt", "value"],
    "merchant_id": ["merchant_id", "merchant", "merchant_name", "productcd"],
    "payee_account_id": ["payee_account_id", "beneficiary_id", "beneficiary", "namedest",
                         "receiver", "recipient"],
    "device_id": ["device_id", "device", "deviceinfo", "device_info"],
    "ip_address": ["ip_address", "ip"],
    "city": ["city", "location", "addr1"],
    "country": ["country", "addr2"],
    "channel": ["channel", "payment_channel", "devicetype"],
    "txn_type": ["txn_type", "transaction_type", "type"],
    "category": ["category", "merchant_category", "mcc"],
    "label": ["is_fraud", "isfraud", "fraud", "label", "class"],
}


def _find(cols, key):
    lower = {c.lower().replace(" ", "_"): c for c in cols}
    return next((lower[c] for c in CANDIDATES[key] if c in lower), None)


def adapt_external(df, max_rows=None):
    """Map an external fraud dataset onto the pipeline schema.

    Returns (tx, accounts, merchants, labels) where labels is a DataFrame
    [txn_id, is_fraud] or None when the file has no fraud label.
    """
    if max_rows:
        df = df.head(max_rows)
    cols = set(df.columns)
    start = pd.Timestamp("2026-08-01")
    tx = pd.DataFrame(index=df.index)

    if {"step", "type", "amount", "nameOrig", "nameDest"} <= cols:          # PaySim
        is_transfer = df["type"].isin(["TRANSFER", "CASH_OUT"])
        tx["timestamp"] = start + pd.to_timedelta(df["step"].astype(float), unit="h")
        tx["account_id"] = df["nameOrig"].astype(str)
        tx["amount_inr"] = df["amount"]
        tx["payee_account_id"] = np.where(is_transfer, df["nameDest"].astype(str), "")
        tx["merchant_id"] = np.where(is_transfer, "", df["nameDest"].astype(str))
        tx["txn_type"] = np.where(is_transfer, "transfer", "purchase")
        tx["channel"] = df["type"].str.lower()
        category = pd.Series(np.where(is_transfer, "", df["type"].str.lower()), index=df.index)
    elif {"TransactionDT", "TransactionAmt"} <= cols:                       # IEEE-CIS
        tx["timestamp"] = start + pd.to_timedelta(df["TransactionDT"].astype(float), unit="s")
        card = [c for c in ("card1", "card2", "addr1") if c in cols]
        key = df[card[0]].astype(object).fillna("na").astype(str) if card else None
        for c in card[1:]:
            key = key + "-" + df[c].astype(object).fillna("na").astype(str)
        tx["account_id"] = key if key is not None else "A0"
        tx["amount_inr"] = df["TransactionAmt"]
        tx["merchant_id"] = "P-" + df["ProductCD"].astype(str) if "ProductCD" in cols else "P-0"
        tx["device_id"] = df["DeviceInfo"].fillna("").astype(str) if "DeviceInfo" in cols else ""
        tx["city"] = df["addr1"].fillna("").astype(str) if "addr1" in cols else ""
        tx["country"] = df["addr2"].fillna("").astype(str) if "addr2" in cols else ""
        tx["channel"] = df["DeviceType"].fillna("online").astype(str) if "DeviceType" in cols \
            else "online"
        category = df["ProductCD"].astype(str) if "ProductCD" in cols else \
            pd.Series("general", index=df.index)
        if "TransactionID" in cols:
            tx["txn_id"] = df["TransactionID"].astype(str)
    else:                                                                   # generic
        for key in ("txn_id", "timestamp", "account_id", "amount_inr", "merchant_id",
                    "payee_account_id", "device_id", "ip_address", "city", "country",
                    "channel", "txn_type"):
            col = _find(df.columns, key)
            if col is not None:
                tx[key] = df[col]
        if "account_id" not in tx or "amount_inr" not in tx:
            raise ValueError("could not find an account column and an amount column; "
                             "rename them to account_id and amount")
        if "timestamp" not in tx:
            tx["timestamp"] = start + pd.to_timedelta(np.arange(len(df)), unit="min")
        elif pd.api.types.is_numeric_dtype(tx["timestamp"]):
            tx["timestamp"] = start + pd.to_timedelta(tx["timestamp"].astype(float), unit="s")
        cat_col = _find(df.columns, "category")
        category = df[cat_col].astype(str) if cat_col else pd.Series("general", index=df.index)

    if "txn_id" not in tx:
        tx["txn_id"] = [f"X{i:07d}" for i in range(len(tx))]
    tx["txn_id"] = tx["txn_id"].astype(str)
    label_col = _find(df.columns, "label")
    labels = None
    if label_col is not None:
        labels = pd.DataFrame({"txn_id": tx["txn_id"].to_numpy(),
                               "is_fraud": pd.to_numeric(df[label_col], errors="coerce")
                               .fillna(0).astype(int).astype(bool).to_numpy()})
    if "merchant_id" not in tx:
        tx["merchant_id"] = ""
    merchants = pd.DataFrame({"merchant_id": tx["merchant_id"].astype(object).fillna("")
                              .astype(str).to_numpy(),
                              "category": category.astype(object).fillna("general")
                              .astype(str).to_numpy()})
    merchants = merchants[merchants["merchant_id"].str.strip().ne("") &
                          merchants["merchant_id"].ne("nan")].drop_duplicates("merchant_id")
    merchants["merchant_name"] = merchants["merchant_id"]
    merchants["city"] = "online"
    merchants.loc[merchants["category"].isin(["", "nan"]), "category"] = "general"
    tx, accounts, merchants = normalize(tx, None, merchants)
    if labels is not None:
        labels = labels[labels["txn_id"].isin(set(tx["txn_id"]))].reset_index(drop=True)
    return tx, accounts, merchants, labels
