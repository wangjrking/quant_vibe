import sys
from inspect import signature
from pathlib import Path
from types import SimpleNamespace

import numpy as np


MAIN = Path(__file__).resolve().parents[1]
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))

from research_v260_runtime.fixed10_risk_event_v110 import pressure_sells_with_extra_min_age
import research_v260_fixed10_pressure_extra_age_guard_20260822 as subject
from research_v260_fixed10_pressure_extra_age_guard_20260822 import run_policy_at_cost


def test_extra_age_guard_keeps_base_exit_and_filters_extra_exit():
    selected = pressure_sells_with_extra_min_age(
        [1, 2, 3], 2, 1, {1: 9, 2: 7, 3: 2}, 10, 8
    )
    assert selected == [1, 3]


def test_extra_age_guard_none_preserves_original_order():
    selected = pressure_sells_with_extra_min_age(
        [1, 2, 3], 2, 1, {1: 9, 2: 7, 3: 2}, 10, None
    )
    assert selected == [1, 2]


def test_validation_end_date_is_an_explicit_optional_parameter():
    assert signature(run_policy_at_cost).parameters["end_date"].default is None


def test_validation_end_date_reaches_the_simulator_wrapper(monkeypatch):
    captured = {}

    def fake_run_fixed10(*args, **kwargs):
        captured["end_date"] = args[7]
        return "daily", "actions"

    monkeypatch.setattr(subject.round1, "run_fixed10", fake_run_fixed10)
    monkeypatch.setattr(
        subject.regime, "strong_market_mask", lambda score, protocol: np.array([True])
    )
    monkeypatch.setattr(
        subject.cadence, "rebalance_schedule", lambda length, interval: np.array([True])
    )
    monkeypatch.setattr(
        subject.trigger, "pressure_trigger_schedule", lambda strong: np.array([4])
    )
    context = SimpleNamespace(
        empty_block=np.zeros((1, 1), dtype=bool),
        score=np.array([[0.9]]),
        order=np.array([[0]]),
        arrays={"dates": np.array(["20260820"])},
        protocol={},
        harness=object(),
        definition={},
    )
    policy = {
        "maintenance_topup_requires_score": None,
        "portfolio_rebalance_interval_days": 20,
        "portfolio_rebalance_min_weight_deviation": 0.01,
        "score_sell_pressure_trigger": 4,
        "score_sell_pressure_limit": 2,
        "score_sell_pressure_confirmation_days": 1,
    }
    assert run_policy_at_cost(
        context, policy, 10, 0.003, end_date="20260820"
    ) == ("daily", "actions")
    assert captured["end_date"] == "20260820"
