import numpy as np
import pandas as pd

from src.monitor.data import load
from src.monitor.drift import level, psi_categorical, psi_numeric
from src.monitor.policy import RetrainPolicy
from src.monitor.simulate import week_batch

rng = np.random.default_rng(0)


def test_psi_identical_distribution_is_near_zero():
    x = pd.Series(rng.normal(size=5000))
    assert psi_numeric(x, pd.Series(rng.normal(size=5000))) < 0.02


def test_psi_detects_shift():
    x = pd.Series(rng.normal(size=5000))
    assert psi_numeric(x, pd.Series(rng.normal(loc=1.0, size=5000))) > 0.25


def test_psi_categorical_handles_new_category():
    ref = pd.Series(["a"] * 50 + ["b"] * 50)
    cur = pd.Series(["a"] * 40 + ["b"] * 40 + ["c"] * 20)
    assert psi_categorical(ref, cur) > 0.25


def test_levels():
    assert level(0.05) == "ok" and level(0.15) == "warning" and level(0.4) == "critical"


def test_policy_triggers_on_performance_drop_and_respects_cooldown():
    p = RetrainPolicy(baseline_auc=0.85)
    assert p.decide(1, 0.84, {})[0] is False
    assert p.decide(2, 0.70, {})[0] is True          # caída > 10 %
    assert p.decide(3, 0.60, {}) == (False, "enfriamiento")


def test_policy_needs_consecutive_critical_weeks():
    p = RetrainPolicy(baseline_auc=0.85)
    crit = {"Contract": "critical"}
    assert p.decide(1, 0.85, crit)[0] is False        # 1 semana: puede ser ruido
    assert p.decide(2, 0.85, crit)[0] is True         # 2 seguidas: reentrena


def test_simulation_injects_drift():
    pool = load()
    stable = week_batch(pool, 1, np.random.default_rng(1))
    drifted = week_batch(pool, 16, np.random.default_rng(1))
    assert (drifted.Contract == "Month-to-month").mean() > (stable.Contract == "Month-to-month").mean() + 0.1
    assert drifted.MonthlyCharges.mean() > stable.MonthlyCharges.mean() * 1.08
