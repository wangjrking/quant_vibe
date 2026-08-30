import sys
from pathlib import Path

import numpy as np


MAIN = Path(__file__).resolve().parents[1]
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))

from research_v260_fixed10_pressure_regime_trigger_20260822 import (
    pressure_trigger_schedule,
    selection_gates,
)


def test_pressure_trigger_schedule_is_four_strong_five_weak():
    actual = pressure_trigger_schedule([True, False, True])
    assert actual.dtype == np.int16
    assert actual.tolist() == [4, 5, 4]


def test_selection_rejects_forward_regression():
    baseline = {
        "train_2022_2024": {"sharpe": 1.0, "cagr": 0.4, "max_drawdown": 0.3},
        "holdout_2025": {"cumulative_return": 0.8},
        "metrics_0_65pct": {"cumulative_return": 1.0},
        "metrics_0_30pct": {"full_10_position_ratio": 1.0},
    }
    candidate = {
        "train_2022_2024": {"sharpe": 1.1, "cagr": 0.4, "max_drawdown": 0.2},
        "holdout_2025": {"cumulative_return": 0.7},
        "metrics_0_65pct": {"cumulative_return": 1.1},
        "metrics_0_30pct": {"full_10_position_ratio": 1.0},
    }
    gates = selection_gates(candidate, baseline)
    assert gates["train_sharpe_improved"]
    assert not gates["holdout_2025_not_worse"]
