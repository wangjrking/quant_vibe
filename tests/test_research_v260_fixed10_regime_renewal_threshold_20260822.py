import sys
from pathlib import Path

import numpy as np


MAIN = Path(__file__).resolve().parents[1]
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))

from research_v260_fixed10_regime_renewal_threshold_20260822 import (
    renewal_threshold_schedule,
    selection_gates,
)


def test_renewal_threshold_schedule_is_080_strong_085_weak():
    actual = renewal_threshold_schedule([True, False, True])
    assert actual.dtype == np.float64
    assert actual.tolist() == [0.80, 0.85, 0.80]


def test_selection_rejects_drawdown_regression():
    baseline = {
        "metrics_0_30pct": {
            "cumulative_return": 3.0,
            "cagr": 0.5,
            "sharpe": 1.3,
            "max_drawdown": 0.35,
            "annual_returns": {"2022": 0.1, "2023": 0.2},
            "full_10_position_ratio": 1.0,
        },
        "metrics_0_65pct": {"cumulative_return": 2.0},
    }
    candidate = {
        "metrics_0_30pct": {
            "cumulative_return": 3.1,
            "cagr": 0.51,
            "sharpe": 1.31,
            "max_drawdown": 0.36,
            "annual_returns": {"2022": 0.1, "2023": 0.2},
            "full_10_position_ratio": 1.0,
        },
        "metrics_0_65pct": {"cumulative_return": 2.1},
    }
    gates = selection_gates(candidate, baseline)
    assert not gates["drawdown_not_worse"]
