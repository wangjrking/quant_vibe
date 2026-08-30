import sys
from pathlib import Path


MAIN = Path(__file__).resolve().parents[1]
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))

from research_v260_fixed10_pressure_replacement_quality_gate_20260822 import (
    selection_gates,
)
from research_v260_runtime.fixed10_risk_event_v110 import (
    apply_pressure_replacement_quality_gate,
)


def test_pressure_quality_gate_requires_a_refill_for_every_allowed_sale():
    assert apply_pressure_replacement_quality_gate(2, 1, 2) == 2
    assert apply_pressure_replacement_quality_gate(2, 1, 1) == 1
    assert apply_pressure_replacement_quality_gate(1, 1, 0) == 1


def test_selection_gates_reject_forward_or_stress_regression():
    baseline = {
        "train_2022_2024": {"sharpe": 1.0, "cagr": 0.4, "max_drawdown": 0.3},
        "holdout_2025": {"cumulative_return": 0.8},
        "metrics_0_65pct": {"cumulative_return": 1.0},
        "metrics_0_30pct": {"full_10_position_ratio": 1.0},
    }
    candidate = {
        "train_2022_2024": {"sharpe": 1.1, "cagr": 0.5, "max_drawdown": 0.2},
        "holdout_2025": {"cumulative_return": 0.7},
        "metrics_0_65pct": {"cumulative_return": 0.9},
        "metrics_0_30pct": {"full_10_position_ratio": 1.0},
    }
    gates = selection_gates(candidate, baseline)
    assert gates["train_sharpe_improved"]
    assert not gates["holdout_2025_not_worse"]
    assert not gates["stress_not_worse"]
