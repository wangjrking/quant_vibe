import sys
from pathlib import Path


MAIN = Path(__file__).resolve().parents[1]
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))

from research_v260_fixed10_current_defensive_sleeve_20260822 import dominates


def test_sleeve_cannot_trade_return_for_worse_drawdown():
    baseline = {
        "metrics_0_30pct": {
            "cumulative_return": 3.0, "sharpe": 1.3, "max_drawdown": 0.35,
            "annual_returns": {"2022": 0.1}, "full_10_position_ratio": 1.0,
        },
        "metrics_0_65pct": {"cumulative_return": 2.0},
    }
    candidate = {
        "metrics_0_30pct": {
            "cumulative_return": 3.1, "sharpe": 1.31, "max_drawdown": 0.36,
            "annual_returns": {"2022": 0.1}, "full_10_position_ratio": 1.0,
        },
        "metrics_0_65pct": {"cumulative_return": 2.1},
    }
    assert not dominates(candidate, baseline)["drawdown_not_worse"]
