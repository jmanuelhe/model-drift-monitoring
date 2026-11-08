"""Métricas de drift.

PSI (Population Stability Index): compara la distribución de una variable en producción
contra la de referencia (entrenamiento).
    PSI = Σ (p_prod - p_ref) * ln(p_prod / p_ref)   sobre bins
Convención de la industria (riesgo crediticio):  < 0.10 estable · 0.10–0.25 vigilar · > 0.25 drift fuerte.

KS (Kolmogorov-Smirnov) y chi-cuadrado dan un p-valor. Con muestras grandes casi todo sale
"significativo", por eso la alerta se basa en el PSI (tamaño del efecto) y el p-valor es complementario.
"""
import numpy as np
import pandas as pd
from scipy import stats

EPS = 1e-4
LEVELS = [(0.10, "ok"), (0.25, "warning"), (np.inf, "critical")]


def _psi_from_props(ref: np.ndarray, cur: np.ndarray) -> float:
    ref, cur = np.clip(ref, EPS, None), np.clip(cur, EPS, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def psi_numeric(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    # Bins por cuantiles de la REFERENCIA, para que cada bin tenga ~10 % de los datos de entrenamiento.
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    return _psi_from_props(r, c)


def psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    cats = sorted(set(ref.unique()) | set(cur.unique()))
    r = ref.value_counts(normalize=True).reindex(cats, fill_value=0).values
    c = cur.value_counts(normalize=True).reindex(cats, fill_value=0).values
    return _psi_from_props(r, c)


def pvalue(ref: pd.Series, cur: pd.Series, numeric: bool) -> float:
    if numeric:
        return float(stats.ks_2samp(ref, cur).pvalue)
    cats = sorted(set(ref.unique()) | set(cur.unique()))
    table = np.array([ref.value_counts().reindex(cats, fill_value=0).values,
                      cur.value_counts().reindex(cats, fill_value=0).values])
    table = table[:, table.sum(axis=0) > 0]
    return float(stats.chi2_contingency(table)[1])


def level(psi: float) -> str:
    return next(name for limit, name in LEVELS if psi < limit)


def drift_report(ref: pd.DataFrame, cur: pd.DataFrame, numeric: list, categorical: list) -> pd.DataFrame:
    rows = []
    for col in numeric + categorical:
        is_num = col in numeric
        psi = psi_numeric(ref[col], cur[col]) if is_num else psi_categorical(ref[col], cur[col])
        rows.append({"feature": col, "psi": psi, "p_value": pvalue(ref[col], cur[col], is_num), "level": level(psi)})
    return pd.DataFrame(rows)
