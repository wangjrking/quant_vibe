from __future__ import annotations

import research_v260_fixed10_residual_cash_sweep_cost_frontier_20260823 as module


def item(delta: float) -> dict:
    return {"sweep_minus_current": {"cumulative_return": delta}}


def test_frontier_reports_highest_positive_and_first_nonpositive_cost() -> None:
    actual = module.summarize_frontier(
        {"0.0030": item(0.1), "0.0065": item(0.01), "0.0100": item(-0.01)}
    )
    assert actual["highest_tested_positive_cost"] == 0.0065
    assert actual["first_tested_nonpositive_cost"] == 0.01
    assert actual["all_tested_costs_positive"] is False


def test_frontier_reports_all_positive() -> None:
    actual = module.summarize_frontier(
        {"0.0030": item(0.1), "0.0065": item(0.01)}
    )
    assert actual["highest_tested_positive_cost"] == 0.0065
    assert actual["first_tested_nonpositive_cost"] is None
    assert actual["all_tested_costs_positive"] is True
