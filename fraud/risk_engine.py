"""Risk engine: three risk scores and one transparent final score (all 0-100).

  Transaction Risk (T)  evidence about the transaction itself: real-time rules, the
                        supervised model and the anomaly detector, combined with a
                        noisy-OR  T = 100 x (1 - prod(1 - w_i)).
  Behavioral Risk (B)   how far the transaction deviates from the account's own profile
                        (weighted average of 7 deviation dimensions).
  Network Risk (N)      ring and graph evidence: the transaction is part of a detected
                        ring's evidence, comes from a fraudster-controlled account, or the
                        account is linked to risky accounts.

  Weighted blend  = wT x T + wB x B + wN x N          (weights configurable, sum to 1)
  Final Risk      = max(blend, guardrail)
  Guardrail       = T when T >= 81, or N when N >= 81 and N rests on direct ring
                    evidence. A CRITICAL, evidence-backed layer is never averaged away by
                    the other two; behavioral deviation alone can never trigger it.
"""

import numpy as np
import pandas as pd

from .config import BANDS, GUARDRAIL, RISK_WEIGHTS

BAND_COLORS = {"LOW": "#22c55e", "MEDIUM": "#eab308", "HIGH": "#f97316", "CRITICAL": "#ef4444"}


def band(score):
    score = 0 if score is None or pd.isna(score) else score
    return next(name for cut, name in BANDS if round(score) >= cut)


def bands(scores):
    s = np.round(np.asarray(scores, float))
    return np.select([s >= c for c, _ in BANDS], [n for _, n in BANDS], default="LOW")


def normalise(weights=None):
    w = dict(RISK_WEIGHTS if weights is None else weights)
    total = sum(max(v, 0) for v in w.values()) or 1.0
    return {k: max(v, 0) / total for k, v in w.items()}


def final_risk(T, B, N, network_evidence, weights=None, guardrail=True):
    """Vectorised final score. Returns DataFrame[blend, final_risk, guardrail]."""
    w = normalise(weights)
    T, B, N = (np.asarray(x, float) for x in (T, B, N))
    ev = np.asarray(network_evidence, bool)
    blend = w["transaction"] * T + w["behavioral"] * B + w["network"] * N
    final = blend.copy()
    rule = np.full(len(T), "", dtype=object)
    if guardrail:
        t_guard = T >= GUARDRAIL
        n_guard = (N >= GUARDRAIL) & ev
        use_t = t_guard & (T > final) & (~n_guard | (T >= N))
        final = np.where(use_t, T, final)
        rule = np.where(use_t, "transaction", rule)
        use_n = n_guard & (N > final)
        final = np.where(use_n, N, final)
        rule = np.where(use_n, "network", rule)
    return pd.DataFrame({"blend": np.round(blend, 1), "final_risk": np.round(final, 1),
                         "guardrail": rule})


def formula_text(T, B, N, network_evidence, weights=None):
    w = normalise(weights)
    r = final_risk([T], [B], [N], [network_evidence], weights).iloc[0]
    lines = [f"Weighted blend = {w['transaction']:.2f} × {T:.0f} (Transaction) + "
             f"{w['behavioral']:.2f} × {B:.0f} (Behavioral) + {w['network']:.2f} × {N:.0f} "
             f"(Network) = {r['blend']:.1f}"]
    if r["guardrail"] == "transaction":
        lines.append(f"Guardrail: Transaction Risk {T:.0f} is CRITICAL (≥ {GUARDRAIL}) on its "
                     f"own evidence → Final = {T:.0f}")
    elif r["guardrail"] == "network":
        lines.append(f"Guardrail: Network Risk {N:.0f} is CRITICAL and rests on direct ring "
                     f"evidence → Final = {N:.0f}")
    else:
        lines.append(f"No guardrail needed → Final = {r['final_risk']:.0f}")
    return lines, r


def noisy_or_100(reasons):
    p = 1.0
    for r in reasons:
        p *= 1 - float(r["weight"])
    return round(100 * (1 - p), 1)
