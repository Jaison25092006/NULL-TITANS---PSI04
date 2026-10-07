"""Central configuration. Every value can be overridden with an environment variable,
so nothing needs editing in code to retune a deployment (no secrets live here).
"""

import os


def _floats(name, default):
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        vals = [float(x) for x in raw.split(",")]
        return vals if len(vals) == len(default) else default
    except ValueError:
        return default


# Final Risk = wT x Transaction Risk + wB x Behavioral Risk + wN x Network/Ring Risk
_w = _floats("FRAUD_RISK_WEIGHTS", [0.45, 0.25, 0.30])
RISK_WEIGHTS = {"transaction": _w[0], "behavioral": _w[1], "network": _w[2]}

# Risk bands (inclusive lower bounds on the 0-100 scale) and the actions they map to.
BANDS = [(81, "CRITICAL"), (61, "HIGH"), (31, "MEDIUM"), (0, "LOW")]
FLAG_THRESHOLD = 61        # HIGH or CRITICAL counts as "flagged" in the evaluation
GUARDRAIL = 81             # a CRITICAL, evidence-backed layer is never averaged away

TIME_WINDOWS = {"5 minutes": 5, "15 minutes": 15, "30 minutes": 30, "1 hour": 60,
                "24 hours": 1440}

BASELINE_DAYS = int(os.environ.get("FRAUD_BASELINE_DAYS", 20))  # "normal" period
TRAIN_SEEDS = tuple(int(s) for s in os.environ.get("FRAUD_TRAIN_SEEDS", "11,13").split(","))
MAX_UPLOAD_ROWS = int(os.environ.get("FRAUD_MAX_UPLOAD_ROWS", 60000))
