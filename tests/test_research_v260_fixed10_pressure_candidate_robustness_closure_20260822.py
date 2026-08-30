from __future__ import annotations

import sys
from pathlib import Path


MAIN = Path(__file__).resolve().parents[1]
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))

from research_v260_fixed10_pressure_candidate_robustness_closure_20260822 import (
    COST_LEVELS,
    START_OFFSETS,
    metric_delta,
)


def test_robustness_budget_is_fixed() -> None:
    assert COST_LEVELS == (0.0030, 0.0040, 0.0065)
    assert START_OFFSETS == (0, 5, 20, 60)


def test_metric_delta_preserves_drawdown_sign() -> None:
    candidate = {
        "cumulative_return": 2.0,
        "cagr": 0.4,
        "sharpe": 1.2,
        "max_drawdown": 0.3,
        "turnover_annualized": 20.0,
        "average_invested_ratio": 0.97,
    }
    reference = {
        "cumulative_return": 1.0,
        "cagr": 0.3,
        "sharpe": 1.0,
        "max_drawdown": 0.4,
        "turnover_annualized": 19.0,
        "average_invested_ratio": 0.90,
    }

    actual = metric_delta(candidate, reference)

    assert actual["max_drawdown"] < 0.0
    assert actual["cumulative_return"] > 0.0
