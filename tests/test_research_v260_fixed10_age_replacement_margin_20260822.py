from __future__ import annotations

from quant.main.research_v260_fixed10_age_replacement_margin_20260822 import (
    AGE_MARGIN_POLICIES,
    candidate_policy,
    selection_key,
)


def test_age_margin_budget_is_small_and_monotone() -> None:
    assert len(AGE_MARGIN_POLICIES) == 3
    for bands in AGE_MARGIN_POLICIES.values():
        if bands is None:
            continue
        margins = [margin for _, margin in bands]
        assert margins == sorted(margins, reverse=True)


def test_candidate_only_changes_age_bands() -> None:
    bands = [(0, 0.08), (10, 0.05), (20, 0.03)]
    actual = candidate_policy(bands)
    assert actual["age_bands"] == bands
    assert actual["replacement_advantage"] == 0.05
    assert actual["sell_confirmation_days"] == 1
    assert actual["max_hold_days"] == 25


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
