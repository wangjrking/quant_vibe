import sys
from pathlib import Path


MAIN = Path(__file__).resolve().parents[1]
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))

from research_v260_fixed10_pressure_coordinate_recheck_20260822 import (
    MARGINS,
    MAX_HOLDS,
    confirmation_gates,
    training_key,
)


def _item(cagr=0.3, sharpe=1.0, mdd=0.3, turnover=10, holdout=0.5, stress=1.0):
    return {
        "train_2022_2024": {
            "cagr": cagr,
            "sharpe": sharpe,
            "max_drawdown": mdd,
            "turnover_annualized": turnover,
        },
        "holdout_2025": {"cumulative_return": holdout},
        "metrics_0_65pct": {"cumulative_return": stress},
        "metrics_0_30pct": {
            "full_10_position_ratio": 1.0,
            "annual_returns": {"2022": 0.1, "2023": 0.2},
        },
    }


def test_coordinate_budgets_are_coarse():
    assert MARGINS == (0.03, 0.05, 0.07)
    assert MAX_HOLDS == (20, 25, 30)


def test_training_key_prioritizes_sharpe():
    assert training_key(_item(sharpe=1.1, cagr=0.2)) > training_key(
        _item(sharpe=1.0, cagr=0.4)
    )


def test_confirmation_requires_forward_and_stress_support():
    baseline = _item(holdout=0.5, stress=1.0)
    candidate = _item(holdout=0.4, stress=1.1)
    gates = confirmation_gates(candidate, baseline)
    assert not gates["forward_2025_not_worse"]
    assert gates["stress_not_worse"]
