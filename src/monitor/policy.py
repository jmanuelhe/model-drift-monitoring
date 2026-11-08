"""Política de reentrenamiento.

Se reentrena cuando se cumple CUALQUIERA de estas condiciones:
  A) Rendimiento: ROC-AUC semanal cae más de MAX_REL_DROP frente a la línea base.
  B) Datos: alguna de las features más importantes está en nivel 'critical' (PSI > 0.25)
     durante CONSECUTIVE semanas seguidas (evita reaccionar a ruido de una sola semana).
Se usa ROC-AUC y no PR-AUC porque PR-AUC depende de la proporción de positivos: si la tasa de
churn sube, PR-AUC puede SUBIR aunque el modelo empeore.
Tras reentrenar hay un enfriamiento de COOLDOWN semanas para no reentrenar en bucle.
"""
from dataclasses import dataclass, field

MAX_REL_DROP = 0.10
CONSECUTIVE = 2
COOLDOWN = 3
KEY_FEATURES = ["tenure", "Contract", "InternetService", "MonthlyCharges"]


@dataclass
class RetrainPolicy:
    baseline_auc: float
    critical_streak: int = 0
    last_retrain_week: int = -100
    log: list = field(default_factory=list)

    def decide(self, week: int, auc: float, drift_levels: dict) -> tuple[bool, str]:
        crit = any(drift_levels.get(f) == "critical" for f in KEY_FEATURES)
        self.critical_streak = self.critical_streak + 1 if crit else 0
        if week - self.last_retrain_week <= COOLDOWN:
            return False, "enfriamiento"
        if auc < self.baseline_auc * (1 - MAX_REL_DROP):
            return self._fire(week, f"ROC-AUC {auc:.3f} < {self.baseline_auc * (1 - MAX_REL_DROP):.3f}")
        if self.critical_streak >= CONSECUTIVE:
            return self._fire(week, f"PSI crítico {self.critical_streak} semanas seguidas")
        return False, ""

    def _fire(self, week, reason):
        self.last_retrain_week, self.critical_streak = week, 0
        self.log.append({"week": week, "reason": reason})
        return True, reason
