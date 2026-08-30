from __future__ import annotations

import sys
from pathlib import Path


MAIN = Path(__file__).resolve().parents[1]
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))

from research_v260_fixed10_pairwise_replacement_guard_recheck_20260822 import (
    CANDIDATES,
    select_candidate,
)


def item(train_sharpe, train_cagr, holdout, stress):
    return {
        "train_2022_2024": {
            "sharpe": train_sharpe,
            "cagr": train_cagr,
            "max_drawdown": 0.20,
            "turnover_annualized": 1.0,
        },
        "holdout_2025": {"cumulative_return": holdout},
        "metrics_0_65pct": {"cumulative_return": stress},
        "metrics_0_30pct": {
            "full_10_position_ratio": 1.0,
            "annual_returns": {"2022": 0.1, "2023": 0.1},
        },
    }


def test_budget_is_binary_and_bounded() -> None:
    assert CANDIDATES == (False, True)


def test_guard_is_selected_only_when_training_and_confirmation_pass() -> None:
    selected, gates = select_candidate(
        {
            "guard_off": item(1.0, 0.2, 0.3, 0.2),
            "guard_on": item(1.1, 0.2, 0.31, 0.21),
        }
    )

    assert selected == "guard_on"
    assert all(gates.values())


def test_guard_is_rejected_when_forward_confirmation_fails() -> None:
    selected, gates = select_candidate(
        {
            "guard_off": item(1.0, 0.2, 0.3, 0.2),
            "guard_on": item(1.1, 0.2, 0.29, 0.21),
        }
    )

    assert selected == "guard_off"
    assert not gates["forward_2025_not_worse"]
