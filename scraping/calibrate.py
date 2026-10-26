"""Turn the web-scraped statistics into generator calibration parameters.

  amount_scale        real average UPI ticket size (mean of the last 3 full years) divided
                      by the generator's average everyday personal payment, so synthetic
                      everyday spending matches real Indian digital-payment amounts
  city_risk_weight    NCRB 2019 crime rate of each generator city's state, normalised to
                      mean 1; used to pick where account-takeover / card-testing /
                      structuring victims live (higher-crime states get more victims)

Writes data/scraped/calibration.json. The default dataset is NOT changed; the calibrated
variant is generated only with DATA_PROFILE=calibrated (see eval/calibrated_experiment.py).

Run: python scraping/calibrate.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRAPED = ROOT / "data" / "scraped"
CITY_STATE = {"Chennai": "Tamil Nadu", "Coimbatore": "Tamil Nadu", "Bengaluru": "Karnataka",
              "Mumbai": "Maharashtra", "Pune": "Maharashtra", "Delhi": "Delhi",
              "Hyderabad": "Telangana", "Kochi": "Kerala"}


def synthetic_everyday_mean():
    """Average everyday (non-business, non-fraud, non-look-alike) personal payment in the
    default synthetic dataset."""
    tx = pd.read_csv(ROOT / "data" / "transactions.csv", keep_default_na=False)
    acc = pd.read_csv(ROOT / "data" / "accounts.csv", keep_default_na=False)
    lab = pd.read_csv(ROOT / "eval" / "transaction_labels.csv", keep_default_na=False)
    t = tx.merge(lab, on="txn_id").merge(acc[["account_id", "account_type"]], on="account_id")
    every = t[(t["account_type"] == "personal") & (t["is_fraud"].astype(str) != "True") &
              (t["benign_tag"] == "") & (t["txn_type"] == "purchase")]
    return float(every["amount_inr"].astype(float).mean()), len(every)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    upi = pd.read_csv(SCRAPED / "upi_yearly.csv")
    crime = pd.read_csv(SCRAPED / "state_crime.csv")
    full = upi[upi["year"] <= 2025].tail(3)
    target = float(full["avg_ticket_inr"].mean())
    current, n = synthetic_everyday_mean()
    scale = round(target / current, 4)

    rate = crime.set_index("state")["crime_rate_2019"]
    weights = {c: float(rate.get(s, np.nan)) for c, s in CITY_STATE.items()}
    known = [v for v in weights.values() if not np.isnan(v)]
    mean = float(np.mean(known))
    weights = {c: round((v if not np.isnan(v) else mean) / mean, 4) for c, v in weights.items()}

    cal = {
        "amount_scale": scale,
        "target_avg_ticket_inr": round(target, 2),
        "target_years": [int(y) for y in full["year"]],
        "synthetic_everyday_mean_inr": round(current, 2),
        "synthetic_everyday_n": n,
        "city_risk_weight": weights,
        "city_state_map": CITY_STATE,
        "state_crime_rate_2019": {s: float(rate.get(s)) for s in sorted(set(CITY_STATE.values()))
                                  if s in rate},
        "sources": json.loads((SCRAPED / "sources.json").read_text("utf-8")).get("tables", {}),
    }
    (SCRAPED / "calibration.json").write_text(json.dumps(cal, indent=2), "utf-8")
    print(f"real UPI avg ticket {target:,.0f} (years {cal['target_years']}) vs synthetic "
          f"everyday mean {current:,.0f} → amount_scale {scale}")
    print("city risk weights:", weights)


if __name__ == "__main__":
    main()
