import sys
from pathlib import Path


MAIN = Path(__file__).resolve().parents[1]
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))

from research_v260_fixed10_regime_pressure_robustness_checkpoint_20260822 import (
    acceptance_gates,
)


def test_simplified_acceptance_uses_whole_period_not_each_year_outperformance():
    current = {
        "cumulative_return": 3.0,
        "cagr": 0.5,
        "sharpe": 1.3,
        "max_drawdown": 0.4,
    }
    candidate = {
        "cumulative_return": 3.1,
        "cagr": 0.51,
        "sharpe": 1.4,
        "max_drawdown": 0.35,
        "annual_returns": {"2022": 0.1, "2023": 0.2, "2024": 0.3, "2025": 0.1},
        "full_10_position_ratio": 1.0,
    }
    costs = {"low": {"cumulative_return": 3.1}, "high": {"cumulative_return": 2.0}}
    starts = {"0": {"cumulative_return": 3.1}, "60": {"cumulative_return": 1.0}}
    assert all(acceptance_gates(candidate, current, costs, starts).values())
