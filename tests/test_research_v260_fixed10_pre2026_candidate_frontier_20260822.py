from __future__ import annotations

from quant.main.research_v260_fixed10_pre2026_candidate_frontier_20260822 import (
    dominates,
    frontier_eligible,
    utility,
)


def metric(cagr, sharpe, drawdown, turnover):
    return {
        "cumulative_return": 1.0,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": drawdown,
        "turnover_annualized": turnover,
        "average_invested_ratio": 1.0,
    }


def test_dominance_requires_no_worse_on_every_axis() -> None:
    better = metric(0.5, 1.5, 0.2, 10.0)
    worse = metric(0.4, 1.4, 0.3, 11.0)
    tradeoff = metric(0.6, 1.4, 0.3, 11.0)
    assert dominates(better, worse)
    assert not dominates(worse, better)
    assert not dominates(better, tradeoff)


def test_fixed_utility_rewards_return_and_penalizes_risk_and_turnover() -> None:
    reference = metric(0.4, 1.2, 0.3, 20.0)
    assert utility(metric(0.5, 1.2, 0.3, 20.0)) > utility(reference)
    assert utility(metric(0.4, 1.2, 0.4, 20.0)) < utility(reference)
    assert utility(metric(0.4, 1.2, 0.3, 30.0)) < utility(reference)


def test_frontier_excludes_explicit_diagnostic_only_payloads() -> None:
    assert not frontier_eligible({"frontier_eligible": False})
    assert not frontier_eligible({
        "event_overlay_role": "diagnostic_only_not_selection_candidate"
    })
    assert frontier_eligible({"frontier_eligible": True})
