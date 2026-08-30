import sys
from pathlib import Path

import pytest


MAIN = Path(__file__).resolve().parents[1]
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))

from research_v260_fixed10_pressure_cooldown_recheck_20260822 import (
    selection_gates,
)
from research_v260_runtime.fixed10_risk_event_v110 import apply_pressure_cooldown


def test_pressure_cooldown_only_suppresses_extra_limit_inside_window():
    assert apply_pressure_cooldown(2, 1, 10, 8, 3) == 1
    assert apply_pressure_cooldown(2, 1, 12, 8, 3) == 2
    assert apply_pressure_cooldown(2, 1, 10, 8, 0) == 2


def test_selection_gates_keep_forward_and_stress_confirmation():
    baseline = {
        "train_2022_2024": {"sharpe": 1.0, "cagr": 0.4, "max_drawdown": 0.3},
        "holdout_2025": {"cumulative_return": 0.8},
        "metrics_0_65pct": {"cumulative_return": 1.0},
        "metrics_0_30pct": {"full_10_position_ratio": 1.0},
    }
    candidate = {
        "train_2022_2024": {"sharpe": 1.1, "cagr": 0.5, "max_drawdown": 0.2},
        "holdout_2025": {"cumulative_return": 0.7},
        "metrics_0_65pct": {"cumulative_return": 1.1},
        "metrics_0_30pct": {"full_10_position_ratio": 1.0},
    }
    gates = selection_gates(candidate, baseline)
    assert gates["train_sharpe_improved"]
    assert not gates["holdout_2025_not_worse"]
