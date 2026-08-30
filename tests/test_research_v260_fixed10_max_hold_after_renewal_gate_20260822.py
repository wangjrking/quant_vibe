from __future__ import annotations

from quant.main.research_v260_fixed10_max_hold_after_renewal_gate_20260822 import (
    MAX_HOLD_DAYS,
    candidate_policy,
    selection_key,
)


def test_max_hold_budget_remains_coarse() -> None:
    assert MAX_HOLD_DAYS == (20, 25, 30)


def test_candidate_keeps_renewal_quality_gate_fixed() -> None:
    actual = candidate_policy(20)
    assert actual["max_hold_days"] == 20
    assert actual["max_hold_renewal_score"] == 0.80
    assert actual["max_daily_score_sells"] == 1
    assert actual["sell_score_below"] == 0.85
    assert actual["replacement_advantage"] == 0.05
    assert actual["min_hold_mode"] == 4


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
