from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_current_rule_ablation_20260822 as mod


def test_build_cases_returns_replay_components(monkeypatch):
    score = np.array([[0.9, 0.8]], dtype=float)
    context = type(
        "Context",
        (),
        {
            "score": score,
            "protocol": {},
            "arrays": {"close_qfq": np.ones_like(score)},
        },
    )()
    monkeypatch.setattr(mod.regime, "strong_market_mask", lambda score, protocol: np.array([True]))
    monkeypatch.setattr(mod.confirmed, "confirmed_weak_mask", lambda strong, days: np.array([False]))
    monkeypatch.setattr(mod.age_boundary, "pressure_age_schedule", lambda strong, age: np.array([4]))
    monkeypatch.setattr(mod.defensive, "trailing_log_volatility", lambda close, window: np.ones_like(close))
    monkeypatch.setattr(mod.percentile, "cross_sectional_percent_rank", lambda value: value)
    monkeypatch.setattr(mod.priority, "sell_priority_matrix", lambda *args: score - 0.1)
    policy = {
        "score_sell_pressure_limit": 2,
        "maintenance_topup_requires_score": 0.8,
        "renewal_policy": "no_score_exit",
        "max_hold_renewal_score": 0.8,
        "portfolio_rebalance_interval_days": 20,
    }
    cases, extra_age, priority_matrix = mod.build_cases(context, policy)
    assert "full_current" in cases
    assert np.array_equal(extra_age, np.array([4]))
    assert np.array_equal(priority_matrix, score - 0.1)
