from __future__ import annotations

from quant.main.research_v260_fixed10_sell_threshold_final_recheck_20260822 import (
    SELL_THRESHOLDS,
    candidate_policy,
    selection_key,
)


def test_threshold_budget_is_bounded() -> None:
    assert SELL_THRESHOLDS == (0.80, 0.85, 0.90)


def test_candidate_only_changes_threshold() -> None:
    actual = candidate_policy(0.80)
    assert actual["sell_score_below"] == 0.80
    assert actual["replacement_advantage"] == 0.05
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
