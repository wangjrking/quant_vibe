from __future__ import annotations

from quant.main.research_v260_fixed10_renewal_policy_20260822 import (
    RENEWAL_POLICIES,
    candidate_policy,
    selection_key,
)


def test_renewal_budget_is_bounded() -> None:
    assert RENEWAL_POLICIES == (
        "none",
        "top1_rebalance",
        "no_score_exit_daily",
        "no_score_exit_rebalance",
    )


def test_candidate_only_changes_renewal_policy() -> None:
    actual = candidate_policy("top1_rebalance")
    assert actual["renewal_policy"] == "top1_rebalance"
    assert actual["min_hold_mode"] == 4
    assert actual["max_hold_days"] == 25
    assert actual["reentry_cooldown_days"] == 0


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
