from __future__ import annotations

from inspect import signature

from quant.main.research_v260_fixed10_price_trailing_exit_20260822 import (
    PRICE_PEAK_DRAWDOWN_THRESHOLDS,
    candidate_id,
    candidate_policy,
    selection_key,
)
from quant.main.research_v260_runtime import fixed10_price_trailing_v109


def test_price_trailing_budget_is_coarse_and_bounded() -> None:
    assert PRICE_PEAK_DRAWDOWN_THRESHOLDS == (None, 0.10, 0.15, 0.20)


def test_candidate_only_adds_price_peak_drawdown_threshold() -> None:
    off = candidate_policy(None)
    active = candidate_policy(0.15)
    changed = {key for key in active if active[key] != off.get(key)}
    assert changed == {"price_peak_drawdown_exit"}
    assert active["max_daily_score_sells"] == 1
    assert active["sell_score_below"] == 0.85
    assert active["replacement_advantage"] == 0.05
    assert active["max_hold_days"] == 25
    assert active["min_hold_mode"] == 4
    assert active["renewal_policy"] == "no_score_exit_rebalance"


def test_candidate_ids_are_stable() -> None:
    assert candidate_id(None) == "price_peak_drawdown_off"
    assert candidate_id(0.10) == "price_peak_drawdown_0.10"


def test_runtime_defaults_to_disabled_price_trailing_exit() -> None:
    parameter = signature(fixed10_price_trailing_v109.simulate).parameters[
        "price_peak_drawdown_exit_override"
    ]
    assert parameter.default is None


def test_selection_prioritizes_training_sharpe() -> None:
    def item(sharpe: float, drawdown: float) -> dict:
        return {
            "train_2022_2024": {
                "sharpe": sharpe,
                "cagr": 0.2,
                "max_drawdown": drawdown,
                "turnover_annualized": 10.0,
            },
        }

    assert selection_key(item(1.0, 0.4)) > selection_key(item(0.9, 0.2))
