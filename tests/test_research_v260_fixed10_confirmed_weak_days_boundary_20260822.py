from __future__ import annotations

import sys
from pathlib import Path


MAIN_ROOT = Path(r"D:/work/quant/quant_mcp/quant/main")
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_confirmed_weak_days_boundary_20260822 as subject


def metrics(cumulative: float, sharpe: float, drawdown: float) -> dict:
    return {
        "metrics_0_30pct": {
            "cumulative_return": cumulative,
            "cagr": cumulative,
            "sharpe": sharpe,
            "max_drawdown": drawdown,
            "annual_returns": {"2022": 0.1},
            "full_10_position_ratio": 1.0,
        },
        "metrics_0_65pct": {"cumulative_return": cumulative / 2},
    }


def test_tiny_drawdown_difference_is_not_a_false_block() -> None:
    baseline = metrics(1.0, 1.0, 0.30)
    candidate = metrics(1.1, 1.1, 0.3005)

    gates = subject.practical_selection_gates(candidate, baseline)

    assert all(gates.values())


def test_material_drawdown_worsening_still_rejects() -> None:
    baseline = metrics(1.0, 1.0, 0.30)
    candidate = metrics(1.1, 1.1, 0.302)

    gates = subject.practical_selection_gates(candidate, baseline)

    assert gates["drawdown_within_10bp_tolerance"] is False
