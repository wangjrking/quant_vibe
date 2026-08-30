from __future__ import annotations

from quant.main.research_v260_fixed10_reentry_cooldown_20260822 import (
    REENTRY_COOLDOWN_DAYS,
    candidate_policy,
    selection_key,
)


def test_reentry_budget_is_bounded() -> None:
    assert REENTRY_COOLDOWN_DAYS == (0, 2, 5)


def test_candidate_only_changes_reentry_cooldown() -> None:
    actual = candidate_policy(5)
    assert actual["reentry_cooldown_days"] == 5
    assert actual["min_hold_mode"] == 4
    assert actual["max_hold_days"] == 25
    assert actual["sell_score_below"] == 0.85


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
