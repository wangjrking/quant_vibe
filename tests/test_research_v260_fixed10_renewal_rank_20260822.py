from __future__ import annotations

from inspect import signature

from quant.main.research_v260_fixed10_renewal_rank_20260822 import (
    MAX_HOLD_RENEWAL_RANKS,
    PORTFOLIO_REBALANCE_BAND,
    PORTFOLIO_REBALANCE_INTERVAL_DAYS,
    candidate_id,
    candidate_policy,
    robust_selection,
)
from quant.main.research_v260_runtime import fixed10_renewal_rank_v109


def test_rank_budget_is_coarse_and_bounded() -> None:
    assert MAX_HOLD_RENEWAL_RANKS == (None, 10, 20, 30)


def test_rank_candidate_changes_only_the_renewal_gate() -> None:
    baseline = candidate_policy(None)
    active = candidate_policy(20)
    changed = {key for key in active if active[key] != baseline.get(key)}
    assert changed == {"max_hold_renewal_rank"}
    assert active["max_hold_renewal_score"] == 0.80
    assert active["portfolio_rebalance_interval_days"] == 20
    assert active["portfolio_rebalance_min_weight_deviation"] == 0.01


def test_candidate_ids_are_stable() -> None:
    assert candidate_id(None) == "renewal_score_080"
    assert candidate_id(10) == "renewal_rank_top10"
    assert candidate_id(30) == "renewal_rank_top30"


def test_runtime_rank_gate_defaults_to_disabled() -> None:
    parameter = signature(fixed10_renewal_rank_v109.simulate).parameters[
        "max_hold_renewal_rank_override"
    ]
    assert parameter.default is None


def test_current_equalweight_controls_are_frozen() -> None:
    assert PORTFOLIO_REBALANCE_INTERVAL_DAYS == 20
    assert PORTFOLIO_REBALANCE_BAND == 0.01


def test_robust_selection_prefers_consistent_fold_winner_within_band() -> None:
    def item(sharpe: float) -> dict:
        return {
            "train_2022_2024": {
                "sharpe": sharpe,
                "cagr": 0.30,
                "max_drawdown": 0.40,
                "turnover_annualized": 20.0,
            }
        }

    results = {"aggregate": item(0.90), "stable": item(0.895)}
    walk_forward = [
        {"selected_policy_id": "stable"},
        {"selected_policy_id": "stable"},
    ]
    selected, reason = robust_selection(results, results, walk_forward)
    assert selected == "stable"
    assert reason == "walk_forward_consistent_within_sharpe_equivalence_band"
