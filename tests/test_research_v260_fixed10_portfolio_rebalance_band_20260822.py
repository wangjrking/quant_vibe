from __future__ import annotations

from quant.main.research_v260_fixed10_portfolio_rebalance_band_20260822 import (
    PORTFOLIO_REBALANCE_BANDS,
    PORTFOLIO_REBALANCE_INTERVAL_DAYS,
    SHARPE_PRACTICAL_EQUIVALENCE,
    candidate_id,
    candidate_policy,
    robust_selection,
)


def test_rebalance_band_budget_is_coarse() -> None:
    assert PORTFOLIO_REBALANCE_BANDS == (0.00, 0.01, 0.02, 0.03)
    assert PORTFOLIO_REBALANCE_INTERVAL_DAYS == 20


def test_candidate_only_adds_weight_deviation_band() -> None:
    zero = candidate_policy(0.00)
    active = candidate_policy(0.02)
    changed = {key for key in active if active[key] != zero.get(key)}
    assert changed == {"portfolio_rebalance_min_weight_deviation"}
    assert active["portfolio_rebalance_interval_days"] == 20
    assert active["max_hold_renewal_score"] == 0.80


def test_candidate_ids_are_stable() -> None:
    assert candidate_id(0.00) == "portfolio_rebalance_band_00pct"
    assert candidate_id(0.02) == "portfolio_rebalance_band_02pct"


def test_robust_selection_prefers_consistent_candidate_in_band() -> None:
    assert SHARPE_PRACTICAL_EQUIVALENCE == 0.01
    results = {
        "zero": {
            "train_2022_2024": {
                "sharpe": 1.0,
                "cagr": 0.2,
                "max_drawdown": 0.3,
                "turnover_annualized": 20.0,
            }
        },
        "band": {
            "train_2022_2024": {
                "sharpe": 0.995,
                "cagr": 0.2,
                "max_drawdown": 0.2,
                "turnover_annualized": 10.0,
            }
        },
    }
    selected, reason = robust_selection(
        results,
        results,
        [{"selected_policy_id": "band"}, {"selected_policy_id": "band"}],
    )
    assert selected == "band"
    assert reason == "walk_forward_consistent_within_sharpe_equivalence_band"
