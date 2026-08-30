from __future__ import annotations

import numpy as np

from quant.main.research_v260_fixed10_weak_market_exit_capacity_20260822 import (
    STRONG_MARKET_EXIT_LIMIT,
    WEAK_MARKET_EXIT_LIMITS,
    candidate_policy,
    exit_limit_schedule,
    selection_key,
)


def test_weak_market_exit_budget_is_small() -> None:
    assert WEAK_MARKET_EXIT_LIMITS == (1, 2, 3)
    assert STRONG_MARKET_EXIT_LIMIT == 1


def test_candidate_keeps_current_holding_chain_fixed() -> None:
    actual = candidate_policy(2)
    assert actual["weak_market_max_daily_score_sells"] == 2
    assert actual["strong_market_max_daily_score_sells"] == 1
    assert actual["max_daily_score_sells"] == 1
    assert actual["sell_score_below"] == 0.85
    assert actual["max_hold_renewal_score"] == 0.80


def test_exit_limit_schedule_only_changes_weak_market() -> None:
    score = np.asarray([[0.95, 0.91], [0.95, 0.20]], dtype=float)
    protocol = {
        "breadth": {"score_threshold": 0.90},
        "breadth_state": {"breadth_count_threshold": 2, "high_when": "above"},
    }
    np.testing.assert_array_equal(
        exit_limit_schedule(score, protocol, 3), np.asarray([1, 3])
    )


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
