from __future__ import annotations

from quant.main.research_v260_fixed10_replacement_margin_final_recheck_20260822 import (
    REPLACEMENT_MARGINS,
    candidate_policy,
    selection_key,
)


def test_margin_budget_is_bounded() -> None:
    assert REPLACEMENT_MARGINS == (0.03, 0.05, 0.08, 0.10)


def test_candidate_only_changes_replacement_margin() -> None:
    actual = candidate_policy(0.08)
    assert actual["replacement_advantage"] == 0.08
    assert actual["max_hold_days"] == 25
    assert actual["min_hold_mode"] == 4
    assert actual["age_bands"] is None


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
