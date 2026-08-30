import sys
from pathlib import Path


MAIN = Path(__file__).resolve().parents[1]
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))

from research_v260_fixed10_pressure_size_guard_combination_20260822 import (
    CANDIDATES,
    selection_gates,
)


def test_combination_budget_has_one_coarse_candidate_only():
    assert CANDIDATES == {
        "current_pressure_exit": 0.0,
        "pressure_exit_plus_weak_market_bottom10_size_guard": 0.10,
    }


def test_selection_requires_training_forward_stress_and_exact10():
    baseline = {
        "train_2022_2024": {"sharpe": 1.0, "cagr": 0.4, "max_drawdown": 0.3},
        "holdout_2025": {"cumulative_return": 0.8},
        "metrics_0_65pct": {"cumulative_return": 1.0},
        "metrics_0_30pct": {
            "annual_returns": {"2022": 0.1, "2023": 0.1},
            "full_10_position_ratio": 1.0,
        },
    }
    candidate = {
        "train_2022_2024": {"sharpe": 1.1, "cagr": 0.5, "max_drawdown": 0.2},
        "holdout_2025": {"cumulative_return": 0.9},
        "metrics_0_65pct": {"cumulative_return": 1.1},
        "metrics_0_30pct": {
            "annual_returns": {"2022": 0.1, "2023": 0.1},
            "full_10_position_ratio": 1.0,
        },
    }
    assert all(selection_gates(candidate, baseline).values())
