from __future__ import annotations

import sys
from pathlib import Path


MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_soft_width_profit_optimization_20260822 as mod


def sample(return_value, stress, sharpe=1.0, drawdown=0.2):
    return {
        "metrics_0_30pct": {
            "cumulative_return": return_value,
            "sharpe": sharpe,
            "max_drawdown": drawdown,
        },
        "metrics_0_65pct": {"cumulative_return": stress},
        "profit_eligible": bool(
            return_value > 0
            and stress > 0
            and sharpe > 0
            and drawdown <= mod.MAX_DRAWDOWN
        ),
    }


def test_choose_width_prioritizes_return_not_exactly10():
    results = {
        "9": sample(3.8, 2.4),
        "10": sample(4.0, 2.6),
        "11": sample(4.3, 2.7),
    }
    assert mod.choose_width(results) == "11"


def test_exactly10_and_investment_are_not_hard_gates():
    baseline = {
        "cumulative_return": 1.0,
        "sharpe": 0.8,
        "max_drawdown": 0.3,
        "average_invested_ratio": 0.75,
        "full_target_position_ratio": 0.0,
    }
    stress = {"cumulative_return": 0.4}
    assert mod.profit_eligible(baseline, stress)


def test_basic_risk_floor_still_rejects_excess_drawdown():
    baseline = {
        "cumulative_return": 2.0,
        "sharpe": 1.2,
        "max_drawdown": mod.MAX_DRAWDOWN + 0.001,
    }
    assert not mod.profit_eligible(baseline, {"cumulative_return": 1.0})
