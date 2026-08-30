from __future__ import annotations

import numpy as np

from quant.main.research_v260_fixed10_portfolio_rebalance_cadence_20260822 import (
    PORTFOLIO_REBALANCE_INTERVALS,
    SHARPE_PRACTICAL_EQUIVALENCE,
    candidate_id,
    candidate_policy,
    rebalance_schedule,
    robust_selection,
    selection_key,
)


def test_portfolio_rebalance_budget_is_coarse() -> None:
    assert PORTFOLIO_REBALANCE_INTERVALS == (None, 20, 10, 5)
    assert SHARPE_PRACTICAL_EQUIVALENCE == 0.01


def test_candidate_only_adds_portfolio_rebalance_interval() -> None:
    off = candidate_policy(None)
    active = candidate_policy(10)
    changed = {key for key in active if active[key] != off.get(key)}
    assert changed == {"portfolio_rebalance_interval_days"}
    assert active["max_hold_renewal_score"] == 0.80
    assert active["max_daily_score_sells"] == 1
    assert active["sell_score_below"] == 0.85


def test_rebalance_schedule_has_fixed_cadence() -> None:
    np.testing.assert_array_equal(
        rebalance_schedule(7, 3),
        np.asarray([False, False, True, False, False, True, False]),
    )
    assert not rebalance_schedule(7, None).any()


def test_candidate_ids_are_stable() -> None:
    assert candidate_id(None) == "portfolio_rebalance_off"
    assert candidate_id(20) == "portfolio_rebalance_20d"


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


def test_robust_selection_prefers_consistent_near_best_candidate() -> None:
    results = {
        "fast": {
            "train_2022_2024": {
                "sharpe": 1.000,
                "cagr": 0.2,
                "max_drawdown": 0.3,
                "turnover_annualized": 20.0,
            }
        },
        "stable": {
            "train_2022_2024": {
                "sharpe": 0.995,
                "cagr": 0.2,
                "max_drawdown": 0.2,
                "turnover_annualized": 10.0,
            }
        },
    }
    walk_forward = [
        {"selected_policy_id": "stable"},
        {"selected_policy_id": "stable"},
    ]
    selected, reason = robust_selection(results, results, walk_forward)
    assert selected == "stable"
    assert reason == "walk_forward_consistent_within_sharpe_equivalence_band"
