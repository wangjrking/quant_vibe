from __future__ import annotations

import numpy as np

from quant.main.research_v260_fixed10_regime_exit_threshold_20260822 import (
    STRONG_MARKET_EXIT_THRESHOLD,
    WEAK_MARKET_EXIT_THRESHOLDS,
    candidate_policy,
    selection_key,
    threshold_schedule,
)


def test_regime_exit_budget_is_coarse() -> None:
    assert WEAK_MARKET_EXIT_THRESHOLDS == (0.80, 0.85, 0.90)
    assert STRONG_MARKET_EXIT_THRESHOLD == 0.85


def test_candidate_keeps_equalweight_policy_fixed() -> None:
    actual = candidate_policy(0.80)
    assert actual["weak_market_sell_score_below"] == 0.80
    assert actual["strong_market_sell_score_below"] == 0.85
    assert actual["max_daily_score_sells"] == 1
    assert actual["replacement_advantage"] == 0.05
    assert actual["renewal_policy"] == "no_score_exit_rebalance"


def test_threshold_schedule_uses_frozen_breadth_state() -> None:
    score = np.asarray([[0.95, 0.91], [0.95, 0.20]], dtype=float)
    protocol = {
        "breadth": {"score_threshold": 0.90},
        "breadth_state": {"breadth_count_threshold": 2, "high_when": "above"},
    }
    actual = threshold_schedule(score, protocol, 0.80)
    np.testing.assert_array_equal(actual, np.asarray([0.85, 0.80]))


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
