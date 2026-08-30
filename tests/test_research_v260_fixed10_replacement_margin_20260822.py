from __future__ import annotations

from quant.main.research_v260_fixed10_replacement_margin_20260822 import (
    REPLACEMENT_MARGINS,
    candidate_policy,
    selection_key,
)


def test_margin_budget_is_bounded_and_contains_production_value() -> None:
    assert REPLACEMENT_MARGINS == (0.03, 0.05, 0.08, 0.10)


def test_candidate_only_changes_replacement_margin() -> None:
    actual = candidate_policy(0.08)
    assert actual["replacement_advantage"] == 0.08
    assert actual["max_hold_days"] == 25
    assert actual["sell_score_below"] == 0.85
    assert actual["max_daily_score_sells"] == 1
    assert actual["renewal_policy"] == "no_score_exit_rebalance"


def test_selection_prioritizes_training_sharpe_over_turnover() -> None:
    def item(sharpe: float, turnover: float) -> dict:
        return {
            "train_2022_2024": {
                "sharpe": sharpe,
                "cagr": 0.2,
                "max_drawdown": 0.3,
                "turnover_annualized": turnover,
            },
        }

    assert selection_key(item(1.0, 20.0)) > selection_key(item(0.9, 10.0))
