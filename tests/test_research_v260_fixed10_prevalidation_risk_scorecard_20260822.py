from __future__ import annotations

from research_v260_fixed10_prevalidation_risk_scorecard_20260822 import (
    classify_validation_result,
)


def metrics(return_value: float, sharpe: float, drawdown: float) -> dict:
    return {
        "cumulative_return": return_value,
        "sharpe": sharpe,
        "max_drawdown": drawdown,
    }


def test_rejects_when_frozen_gate_fails() -> None:
    assert classify_validation_result(
        metrics(0.20, 1.2, 0.10),
        metrics(0.10, 1.0, 0.20),
        frozen_gates_passed=False,
    ) == "reject"


def test_labels_return_only_upgrade_separately() -> None:
    assert classify_validation_result(
        metrics(0.20, 0.9, 0.25),
        metrics(0.10, 1.0, 0.20),
        frozen_gates_passed=True,
    ) == "return_upgrade_with_risk_tradeoff"


def test_labels_full_risk_adjusted_upgrade() -> None:
    assert classify_validation_result(
        metrics(0.20, 1.1, 0.19),
        metrics(0.10, 1.0, 0.20),
        frozen_gates_passed=True,
    ) == "risk_adjusted_upgrade"
