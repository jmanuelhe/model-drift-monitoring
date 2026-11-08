"""Simula 16 semanas de producción y compara un modelo estático vs uno con política de reentrenamiento."""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from . import dashboard, plots
from .data import CATEGORICAL, FEATURES, NUMERIC, ROOT, TARGET, load
from .drift import drift_report, psi_numeric
from .model import build_model
from .policy import RetrainPolicy
from .simulate import WEEKS, simulate

REPORTS = ROOT / "reports"
WINDOW = 4  # semanas de datos etiquetados usadas al reentrenar


def main():
    (REPORTS / "figures").mkdir(parents=True, exist_ok=True)
    df = load()
    train, pool = train_test_split(df, test_size=0.4, stratify=df[TARGET], random_state=42)
    stream = simulate(pool.reset_index(drop=True))

    static = build_model().fit(train[FEATURES], train[TARGET])
    live, live_ref = static, train
    base_auc = roc_auc_score(pool[TARGET], static.predict_proba(pool[FEATURES])[:, 1])
    policy = RetrainPolicy(baseline_auc=base_auc)
    ref_scores = static.predict_proba(train[FEATURES])[:, 1]

    weekly, psi_rows = [], []
    for w in range(1, WEEKS + 1):
        b = stream[stream.week == w]
        y = b[TARGET]
        p_static = static.predict_proba(b[FEATURES])[:, 1]
        p_live = live.predict_proba(b[FEATURES])[:, 1]

        # Dos referencias: (1) el entrenamiento ORIGINAL, para ver cuánto cambió el mundo (heatmap);
        # (2) los datos con que se entrenó el modelo ACTUAL, para decidir si hay que reentrenar.
        rep = drift_report(train, b, NUMERIC, CATEGORICAL)
        rep_live = drift_report(live_ref, b, NUMERIC, CATEGORICAL)
        psi_rows += [{"week": w, **r} for r in rep.to_dict("records")]
        levels = dict(zip(rep_live.feature, rep_live.level))
        score_psi = psi_numeric(pd.Series(ref_scores), pd.Series(p_static))

        auc_live = roc_auc_score(y, p_live)
        retrain, reason = policy.decide(w, auc_live, levels)
        weekly.append({
            "week": w, "churn_rate": y.mean(),
            "auc_static": roc_auc_score(y, p_static), "auc_live": auc_live,
            "pr_auc_static": average_precision_score(y, p_static), "pr_auc_live": average_precision_score(y, p_live),
            "pred_rate_static": p_static.mean(), "pred_rate_live": p_live.mean(),
            "score_psi": score_psi, "max_psi": rep.psi.max(), "max_psi_vs_live_ref": rep_live.psi.max(),
            "n_critical": int((rep.level == "critical").sum()), "n_warning": int((rep.level == "warning").sum()),
            "retrain": retrain, "reason": reason,
        })
        if retrain:  # el modelo nuevo se usa desde la semana siguiente
            recent = stream[(stream.week > w - WINDOW) & (stream.week <= w)]
            live, live_ref = build_model().fit(recent[FEATURES], recent[TARGET]), recent

    weekly = pd.DataFrame(weekly)
    psi = pd.DataFrame(psi_rows)
    weekly.to_csv(REPORTS / "weekly_metrics.csv", index=False)
    psi.to_csv(REPORTS / "psi_by_week.csv", index=False)

    after = weekly.week > weekly.loc[weekly.retrain, "week"].min() if weekly.retrain.any() else weekly.week > 0
    summary = {
        "baseline_auc": base_auc,
        "retrains": policy.log,
        "auc_mean_static": weekly.auc_static.mean(), "auc_mean_live": weekly.auc_live.mean(),
        "auc_static_weeks_9_16": weekly[weekly.week >= 9].auc_static.mean(),
        "auc_live_weeks_9_16": weekly[weekly.week >= 9].auc_live.mean(),
        "calibration_gap_static_last": float(weekly.churn_rate.iloc[-1] - weekly.pred_rate_static.iloc[-1]),
        "calibration_gap_live_last": float(weekly.churn_rate.iloc[-1] - weekly.pred_rate_live.iloc[-1]),
        "first_warning_week": int(weekly.loc[weekly.n_warning + weekly.n_critical > 0, "week"].min()),
        "first_critical_week": int(weekly.loc[weekly.n_critical > 0, "week"].min()),
    }
    (REPORTS / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float))
    plots.performance(weekly, REPORTS / "figures/rendimiento.png")
    plots.psi_heatmap(psi, REPORTS / "figures/psi_heatmap.png")
    plots.calibration_drift(weekly, REPORTS / "figures/calibracion_semanal.png")
    dashboard.build(weekly, psi, summary, REPORTS / "dashboard.html")
    print(weekly.round(3).to_string())
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
