from __future__ import annotations

from quant.main.research_v260_fixed10_renewal_maintenance_20260822 import (
    CANDIDATES,
    PORTFOLIO_REBALANCE_BAND,
    PORTFOLIO_REBALANCE_INTERVAL_DAYS,
    candidate_policy,
    robust_selection,
)


def test_candidate_budget_has_only_two_renewal_actions() -> None:
    assert CANDIDATES == {
        "renewal_reset_only": "no_score_exit",
        "renewal_reset_and_rebalance": "no_score_exit_rebalance",
    }


def test_candidates_change_only_renewal_policy() -> None:
    reset = candidate_policy("renewal_reset_only")
    rebalance = candidate_policy("renewal_reset_and_rebalance")
    changed = {key for key in reset if reset[key] != rebalance[key]}
    assert changed == {"renewal_policy"}


def test_current_controls_remain_frozen() -> None:
    policy = candidate_policy("renewal_reset_only")
    assert policy["max_hold_renewal_score"] == 0.80
    assert policy["sell_score_below"] == 0.85
    assert policy["replacement_advantage"] == 0.05
    assert PORTFOLIO_REBALANCE_INTERVAL_DAYS == 20
    assert PORTFOLIO_REBALANCE_BAND == 0.01


def test_robust_selection_accepts_consistent_fold_winner_within_band() -> None:
    def item(sharpe: float) -> dict:
        return {
            "train_2022_2024": {
                "sharpe": sharpe,
                "cagr": 0.30,
                "max_drawdown": 0.40,
                "turnover_annualized": 20.0,
            }
        }

    results = {
        "renewal_reset_only": item(0.895),
        "renewal_reset_and_rebalance": item(0.90),
    }
    walk_forward = [
        {"selected_policy_id": "renewal_reset_only"},
        {"selected_policy_id": "renewal_reset_only"},
    ]
    selected, reason = robust_selection(results, walk_forward)
    assert selected == "renewal_reset_only"
    assert reason == "walk_forward_consistent_within_sharpe_equivalence_band"
