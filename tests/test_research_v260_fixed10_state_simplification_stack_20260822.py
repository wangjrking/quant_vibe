from __future__ import annotations

import research_v260_fixed10_state_simplification_stack_20260822 as target


def test_metrics_delta_preserves_drawdown_direction() -> None:
    keys = (
        "cumulative_return",
        "cagr",
        "sharpe",
        "max_drawdown",
        "turnover_annualized",
        "average_invested_ratio",
    )
    left = {key: 2.0 for key in keys}
    right = {key: 1.0 for key in keys}

    result = target.metrics_delta(left, right)

    assert result == {key: 1.0 for key in keys}
