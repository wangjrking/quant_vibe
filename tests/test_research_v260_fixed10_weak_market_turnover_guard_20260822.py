from __future__ import annotations

import numpy as np

from quant.main.research_v260_fixed10_weak_market_turnover_guard_20260822 import (
    GUARD_CANDIDATES,
    candidate_policy,
    schedules,
    selection_key,
)


def test_guard_budget_has_only_three_structural_cases() -> None:
    assert [item["candidate_id"] for item in GUARD_CANDIDATES] == [
        "weak_turnover_guard_off",
        "weak_suppress_score_exit",
        "weak_suppress_score_and_renewal_exit",
    ]


def test_candidate_keeps_main_holding_chain_fixed() -> None:
    actual = candidate_policy(GUARD_CANDIDATES[1])
    assert actual["max_hold_days"] == 25
    assert actual["sell_score_below"] == 0.85
    assert actual["max_hold_renewal_score"] == 0.80
    assert actual["replacement_advantage"] == 0.05
    assert actual["max_daily_score_sells"] == 1


def test_schedules_only_change_weak_market_values() -> None:
    score = np.asarray([[0.95, 0.91], [0.95, 0.20]], dtype=float)
    protocol = {
        "breadth": {"score_threshold": 0.90},
        "breadth_state": {"breadth_count_threshold": 2, "high_when": "above"},
    }
    sell, renewal = schedules(score, protocol, GUARD_CANDIDATES[2])
    np.testing.assert_array_equal(sell, np.asarray([0.85, 0.00]))
    np.testing.assert_array_equal(renewal, np.asarray([0.80, 0.00]))


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
