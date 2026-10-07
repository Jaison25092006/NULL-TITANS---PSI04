"""Model Performance: precision-first evaluation against labels the detector never saw."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

from ui import state
from ui.components import alert, hero, kpi_row, style_fig


def _prf(flag, y):
    tp, fp = int((flag & y).sum()), int((flag & ~y).sum())
    fn, tn = int((~flag & y).sum()), int((~flag & ~y).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "fpr": fp / (fp + tn) if fp + tn else 0.0, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def render():
    d, t = state.data(), state.txn()
    hero("Model Performance",
         "Scored against hidden labels the detector never reads · false positives are "
         "penalised, so precision comes first")
    if d["labels"] is None:
        alert("This dataset has no fraud labels, so accuracy cannot be measured. Score "
              "distributions are shown on the Executive Dashboard.", "info")
        return
    x = t.merge(d["labels"][["txn_id", "is_fraud"] + [c for c in ("pattern", "benign_tag")
                                                     if c in d["labels"]]], on="txn_id")
    y = x["is_fraud"].to_numpy(bool)
    if y.sum() == 0 or (~y).sum() == 0:
        alert("Labels contain only one class; metrics are undefined.", "info")
        return
    thr = st.slider("Decision threshold (Final Risk ≥ … counts as flagged)", 31, 95, 61, 1,
                    help="61 = HIGH or CRITICAL (step-up or block).")
    m = _prf(x["final_risk"].to_numpy() >= thr, y)
    auc = roc_auc_score(y, x["final_risk"])
    kpi_row([("Precision", f"{100 * m['precision']:.1f}%", f"{m['fp']} false alarms", "#22c55e"),
             ("Recall", f"{100 * m['recall']:.1f}%", f"{m['fn']} missed"),
             ("F1 score", f"{100 * m['f1']:.1f}%", ""),
             ("False positive rate", f"{100 * m['fpr']:.3f}%", f"of {m['fp'] + m['tn']:,} legit",
              "#22c55e"),
             ("ROC-AUC", f"{auc:.4f}", "ranking quality")])
    st.caption(f"Evaluated on {len(x):,} transactions ({int(y.sum())} fraud). The ML component "
               f"was trained on separately generated datasets (seeds "
               f"{state.model_card().get('train_seeds', [11, 13])}); these labels were never used "
               f"for training or model selection. Metrics follow the risk weights in the sidebar.")

    c1, c2, c3 = st.columns([1, 1.2, 1.2])
    with c1:
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        fig = go.Figure(go.Heatmap(z=cm, x=["Allowed", "Flagged"], y=["Legitimate", "Fraud"],
                                   text=cm, texttemplate="%{text:,}", colorscale="Blues",
                                   showscale=False))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style_fig(fig, height=330, legend=False).update_layout(
            title="Confusion matrix"), width="stretch")
    scores = {"Final risk": x["final_risk"], "Transaction risk": x["transaction_risk"],
              "ML model": x["ml_probability"], "Isolation Forest": x["anomaly_score"],
              "Behavioral": x["behavioral_risk"], "Amount only": x["amount_inr"]}
    colors = ["#ef4444", "#f97316", "#a78bfa", "#fbbf24", "#34d399", "#64748b"]
    with c2:
        fig = go.Figure()
        for (name, s), col in zip(scores.items(), colors):
            fpr, tpr, _ = roc_curve(y, s)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{name} ({roc_auc_score(y, s):.3f})",
                                     line=dict(color=col, width=2.5 if name == "Final risk" else 1.5)))
        fig.update_xaxes(type="log", range=[-5, 0], title="false positive rate (log)")
        fig.update_yaxes(title="true positive rate")
        st.plotly_chart(style_fig(fig, height=330).update_layout(title="ROC curves (AUC)",
                        legend=dict(y=-0.35)), width="stretch")
    with c3:
        fig = go.Figure()
        for (name, s), col in zip(scores.items(), colors):
            p, r, _ = precision_recall_curve(y, s)
            fig.add_trace(go.Scatter(x=r, y=p, name=name, showlegend=False,
                                     line=dict(color=col, width=2.5 if name == "Final risk" else 1.5)))
        fig.update_xaxes(title="recall")
        fig.update_yaxes(title="precision", range=[0, 1.03])
        st.plotly_chart(style_fig(fig, height=330).update_layout(title="Precision–recall"),
                        width="stretch")

    st.markdown("### Every layer vs. the combined system")
    methods = {
        "Combined system (Final ≥ threshold)": x["final_risk"] >= thr,
        "Transaction risk only": x["transaction_risk"] >= thr,
        "Real-time rules only": x["realtime_risk"] >= 0.5,
        "ML model only (p ≥ 0.5)": x["ml_probability"] >= 0.5,
        "Isolation Forest only": x["is_anomaly"],
        "Behavioral deviation only": x["behavioral_risk"] >= thr,
        "Naive: flag the top 1% amounts": x["amount_inr"] >= x["amount_inr"].quantile(0.99),
    }
    rows = []
    for name, f in methods.items():
        r = _prf(np.asarray(f, bool), y)
        rows.append({"Method": name, "Precision": round(100 * r["precision"], 1),
                     "Recall": round(100 * r["recall"], 1), "F1": round(100 * r["f1"], 1),
                     "False alarms": r["fp"], "FPR %": round(100 * r["fpr"], 4)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("No single layer is enough: rules are precise but miss rings; the ML model "
               "catches more but raises false alarms; the network layer is what recovers "
               "coordinated fraud. Flagging big amounts catches nothing — big ≠ fraud.")

    c1, c2 = st.columns(2)
    with c1:
        if "pattern" in x:
            st.markdown("### Recall by fraud pattern")
            rp = x[y].groupby("pattern").apply(lambda g: (g["final_risk"] >= thr).mean(),
                                               include_groups=False)
            fig = go.Figure(go.Bar(x=rp.values * 100, y=rp.index, orientation="h",
                                   marker_color="#22c55e", text=[f"{v:.0%}" for v in rp.values],
                                   textposition="outside"))
            fig.update_xaxes(range=[0, 115])
            st.plotly_chart(style_fig(fig, height=330, legend=False), width="stretch")
    with c2:
        if "benign_tag" in x:
            st.markdown("### False alarms on look-alikes")
            b = x[~y & (x["benign_tag"] != "")]
            if len(b):
                fa = b.groupby("benign_tag").agg(transactions=("txn_id", "size"),
                                                 flagged=("final_risk", lambda s: int((s >= thr).sum())))
                st.dataframe(fa.reset_index().rename(columns={"benign_tag": "Legitimate look-alike"}),
                             hide_index=True, width="stretch")
                st.caption("Legitimate behaviour designed to look suspicious: premium customers "
                           "moving lakhs, businesses, travellers, shared household phones, rent "
                           "collected by landlords, occasional crypto buyers.")

    al = d.get("account_labels")
    a = state.acc()
    if al is not None and len(a):
        st.markdown("### Accounts and rings")
        ax = a.merge(al[["account_id", "is_fraud", "ring_id"]].rename(columns={"ring_id": "true_ring"}),
                     on="account_id")
        if len(ax):
            ay = ax["is_fraud"].to_numpy(bool)
            am = _prf(ax["account_risk"].to_numpy() >= 61, ay)
            truth = ax[ax["true_ring"] != ""].groupby("true_ring")["account_id"].apply(set)
            ring_rows = []
            for rid, members in truth.items():
                best = max(state.rings(), key=lambda r: len(members & set(r["members"])) /
                           len(members | set(r["members"])), default=None)
                jac = len(members & set(best["members"])) / len(members | set(best["members"])) \
                    if best else 0
                ring_rows.append({"Planted ring": rid, "Size": len(members),
                                  "Detected as": best["ring_id"] if best and jac else "—",
                                  "Overlap (Jaccard)": f"{jac:.0%}"})
            c1, c2 = st.columns([1, 1.3])
            with c1:
                kpi_row([("Fraud accounts caught", f"{am['tp']}/{am['tp'] + am['fn']}", ""),
                         ("Innocent accounts flagged", am["fp"], "", "#22c55e")])
            with c2:
                st.dataframe(pd.DataFrame(ring_rows), hide_index=True, width="stretch")

    rob = state.robustness()
    if rob:
        st.markdown("### Robustness on unseen datasets (no retuning)")
        st.dataframe(pd.DataFrame(rob).rename(columns={
            "seed": "Seed", "role": "Role", "transactions": "Transactions",
            "txn_precision": "Precision", "txn_recall": "Recall", "txn_f1": "F1",
            "txn_false_alarms": "False alarms", "txn_fpr": "FPR", "roc_auc": "ROC-AUC",
            "fraud_accounts_caught": "Fraud accounts", "innocent_accounts_flagged": "Innocent flagged",
            "rings_exact": "Rings exact", "lookalike_false_alarms": "Look-alike FPs"}),
            hide_index=True, width="stretch")
    card = state.model_card()
    if card:
        st.markdown("### ML model card")
        comp = pd.DataFrame(card.get("comparison", {})).T.reset_index().rename(columns={"index": "Model"})
        st.dataframe(comp, hide_index=True, width="stretch")
        st.caption(f"Selected: **{card.get('name')}** (best validation average precision) · trained "
                   f"on {card.get('train_rows', 0):,} transactions ({card.get('train_fraud', 0)} "
                   f"fraud) from seeds {card.get('train_seeds')} · validated on seed "
                   f"{card.get('train_seeds', [0])[-1]} before the final refit. "
                   f"Its weight in Transaction Risk is capped (0.6 × p), so it cannot block on "
                   f"its own.")
