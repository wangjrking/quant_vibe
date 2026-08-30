from __future__ import annotations

from quant.main.research_v260_fixed10_exit_confirmation_20260822 import (
    CONFIRMATION_DAYS,
    candidate_policy,
    selection_key,
)


def test_confirmation_budget_is_bounded() -> None:
    assert CONFIRMATION_DAYS == (1, 2, 3)


def test_candidate_only_changes_confirmation() -> None:
    actual = candidate_policy(2)
    assert actual["sell_confirmation_days"] == 2
    assert actual["renewal_policy"] == "no_score_exit_rebalance"
    assert actual["max_hold_days"] == 25
    assert actual["min_hold_mode"] == 4


def test_selection_prioritizes_training_sharpe() -> None:
    def item(sharpe: float, cagr: float) -> dict:
        return {
            "train_2022_2024": {
                "sharpe": sharpe,
                "cagr": cagr,
                "max_drawdown": 0.3,
                "turnover_annualized": 10.0,
            }
        }

    assert selection_key(item(1.0, 0.2)) > selection_key(item(0.9, 0.3))
