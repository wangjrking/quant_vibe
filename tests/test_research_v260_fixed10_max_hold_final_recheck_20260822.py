from __future__ import annotations

from quant.main.research_v260_fixed10_max_hold_final_recheck_20260822 import (
    MAX_HOLD_DAYS,
    candidate_policy,
    selection_key,
)


def test_max_hold_budget_is_bounded() -> None:
    assert MAX_HOLD_DAYS == (20, 25, 30)


def test_candidate_only_changes_max_hold() -> None:
    actual = candidate_policy(30)
    assert actual["max_hold_days"] == 30
    assert actual["min_hold_mode"] == 4
    assert actual["age_bands"] is None
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
