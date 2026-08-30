from __future__ import annotations

from quant.main.research_v260_fixed10_replacement_margin_after_renewal_gate_20260822 import (
    REPLACEMENT_MARGINS,
    candidate_policy,
    selection_key,
)


def test_replacement_margin_budget_remains_coarse() -> None:
    assert REPLACEMENT_MARGINS == (0.03, 0.05, 0.08)


def test_candidate_keeps_renewal_gate_and_hold_limit_fixed() -> None:
    actual = candidate_policy(0.03)
    assert actual["replacement_advantage"] == 0.03
    assert actual["max_hold_renewal_score"] == 0.80
    assert actual["max_hold_days"] == 25
    assert actual["max_daily_score_sells"] == 1
    assert actual["sell_score_below"] == 0.85


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
