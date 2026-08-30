import sys
from pathlib import Path


MAIN = Path(__file__).resolve().parents[1]
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))

from research_v260_fixed10_pressure_hold_renewal_recheck_20260822 import (
    MIN_HOLDS,
    RENEWAL_SCORES,
    select_coordinate,
)


def _item(sharpe, holdout=0.5, stress=1.0):
    return {
        "train_2022_2024": {
            "sharpe": sharpe,
            "cagr": 0.3,
            "max_drawdown": 0.3,
            "turnover_annualized": 10.0,
        },
        "holdout_2025": {"cumulative_return": holdout},
        "metrics_0_65pct": {"cumulative_return": stress},
        "metrics_0_30pct": {
            "full_10_position_ratio": 1.0,
            "annual_returns": {"2022": 0.1, "2023": 0.2},
        },
    }


def test_budgets_are_coarse():
    assert MIN_HOLDS == (3, 4, 5)
    assert RENEWAL_SCORES == (0.75, 0.80, 0.85)


def test_coordinate_accepts_train_winner_with_confirmation():
    results = {"base": _item(1.0), "candidate": _item(1.1, 0.6, 1.1)}
    ranked, selected, gates = select_coordinate(results, "base")
    assert ranked == selected == "candidate"
    assert all(gates.values())


def test_coordinate_falls_back_when_forward_confirmation_fails():
    results = {"base": _item(1.0), "candidate": _item(1.1, 0.4, 1.1)}
    ranked, selected, gates = select_coordinate(results, "base")
    assert ranked == "candidate"
    assert selected == "base"
    assert not gates["forward_2025_not_worse"]


def test_coordinate_keeps_baseline_on_exact_tie():
    results = {"candidate": _item(1.0), "base": _item(1.0)}
    ranked, selected, gates = select_coordinate(results, "base")
    assert ranked == selected == "base"
    assert all(gates.values())
