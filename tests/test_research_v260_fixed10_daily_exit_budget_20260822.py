from __future__ import annotations

from quant.main.research_v260_fixed10_daily_exit_budget_20260822 import (
    DAILY_EXIT_LIMITS,
    candidate_policy,
    selection_key,
)


def test_daily_exit_budget_is_small() -> None:
    assert DAILY_EXIT_LIMITS == (1, 2, 3)


def test_candidate_only_changes_daily_exit_limit() -> None:
    actual = candidate_policy(2)
    assert actual["max_daily_score_sells"] == 2
    assert actual["sell_score_below"] == 0.85
    assert actual["replacement_advantage"] == 0.05
    assert actual["max_hold_days"] == 25
    assert actual["min_hold_mode"] == 4
    assert actual["renewal_policy"] == "no_score_exit_rebalance"


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
