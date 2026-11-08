"""Simulación de 16 semanas de producción con drift controlado.

Los datos reales no traen fechas, así que el drift se SIMULA de forma explícita y reproducible.
Así se puede verificar que el monitor detecta exactamente lo que se inyectó:

- Semanas 1–4   · estable: muestras de la misma distribución que el entrenamiento.
- Semanas 5–16  · DATA DRIFT gradual: campaña comercial que atrae más contratos mes a mes
                  y alza de tarifas (MonthlyCharges sube hasta +15 %).
- Semanas 9–16  · CONCEPT DRIFT: un competidor lanza fibra más barata y paga la penalidad
                  de cancelación. Clientes con FIBRA y contrato de 1 o 2 años (que el modelo
                  considera "seguros") ahora se van con probabilidad 50 %. La relación X → y cambia: el modelo
                  viejo no lo sabe, y ninguna feature cambia de distribución por esto.
"""
import numpy as np
import pandas as pd

WEEKS = 16
BATCH = 600
DATA_DRIFT_START, CONCEPT_DRIFT_START = 5, 9


def week_batch(pool: pd.DataFrame, week: int, rng: np.random.Generator) -> pd.DataFrame:
    weights = np.ones(len(pool))
    intensity = 0.0
    if week >= DATA_DRIFT_START:
        intensity = min(1.0, (week - DATA_DRIFT_START + 1) / 8)          # rampa de 8 semanas
        weights = np.where(pool["Contract"] == "Month-to-month", 1 + 2.0 * intensity, 1.0)
    idx = rng.choice(len(pool), size=BATCH, replace=True, p=weights / weights.sum())
    b = pool.iloc[idx].copy().reset_index(drop=True)

    if week >= DATA_DRIFT_START:
        b["MonthlyCharges"] = (b["MonthlyCharges"] * (1 + 0.15 * intensity)).round(2)
    if week >= CONCEPT_DRIFT_START:
        at_risk = (b["Contract"] != "Month-to-month") & (b["InternetService"] == "Fiber optic") & (b["churn"] == 0)
        flip = at_risk & (rng.random(len(b)) < 0.50)
        b.loc[flip, "churn"] = 1
    b["week"] = week
    return b


def simulate(pool: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.concat([week_batch(pool, w, rng) for w in range(1, WEEKS + 1)], ignore_index=True)
