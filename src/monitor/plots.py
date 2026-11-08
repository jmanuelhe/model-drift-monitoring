import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .simulate import CONCEPT_DRIFT_START, DATA_DRIFT_START

INK, CORAL, TEAL, GOLD, SLATE = "#1B2430", "#E4572E", "#17A398", "#F2A541", "#6B7A8F"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 12, "axes.edgecolor": SLATE})


def _phases(ax):
    ax.axvspan(DATA_DRIFT_START - 0.5, CONCEPT_DRIFT_START - 0.5, color=GOLD, alpha=0.10)
    ax.axvspan(CONCEPT_DRIFT_START - 0.5, 16.5, color=CORAL, alpha=0.08)
    y = ax.get_ylim()[1]
    ax.text(DATA_DRIFT_START - 0.4, y, "data drift", va="top", fontsize=8, color=SLATE)
    ax.text(CONCEPT_DRIFT_START - 0.4, y, "+ concept drift", va="top", fontsize=8, color=SLATE)


def _save(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)


def performance(w, path):
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    ax.plot(w.week, w.auc_static, "o-", color=SLATE, label="modelo estático")
    ax.plot(w.week, w.auc_live, "o-", color=TEAL, lw=2.2, label="con política de reentrenamiento")
    for wk in w.loc[w.retrain, "week"]:
        ax.axvline(wk + 0.5, color=CORAL, ls="--", lw=1)
        ax.text(wk + 0.6, w.auc_static.min() + 0.01, "reentrena", color=CORAL, fontsize=8, rotation=90)
    ax.set_ylim(min(w.auc_static.min(), w.auc_live.min()) - 0.03, 0.92)
    _phases(ax)
    ax.set_xticks(range(1, 17)); ax.set_xlabel("semana"); ax.set_ylabel("ROC-AUC")
    ax.set_title("Rendimiento semanal en producción", loc="left"); ax.legend(frameon=False, loc="lower left")
    _save(fig, path)


def psi_heatmap(psi, path):
    p = psi.pivot(index="feature", columns="week", values="psi")
    p = p.loc[p.max(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    im = ax.imshow(p.values, cmap="YlOrRd", vmin=0, vmax=0.4, aspect="auto")
    ax.set_xticks(range(p.shape[1]), p.columns); ax.set_yticks(range(p.shape[0]), p.index)
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            v = p.values[i, j]
            if v >= 0.10:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.25 else INK)
    ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=ax, label="PSI (≥0.10 vigilar · ≥0.25 crítico)")
    ax.set_xlabel("semana"); ax.set_title("PSI por variable y semana (vs. entrenamiento)", loc="left")
    _save(fig, path)


def calibration_drift(w, path):
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    ax.plot(w.week, w.churn_rate * 100, "o-", color=INK, lw=2, label="churn real")
    ax.plot(w.week, w.pred_rate_static * 100, "s--", color=SLATE, label="predicho · estático")
    ax.plot(w.week, w.pred_rate_live * 100, "s--", color=TEAL, label="predicho · reentrenado")
    _phases(ax)
    ax.set_xticks(range(1, 17)); ax.set_xlabel("semana"); ax.set_ylabel("% de clientes")
    ax.set_title("Churn real vs. churn predicho, por semana", loc="left")
    ax.legend(frameon=False, loc="upper left")
    _save(fig, path)
