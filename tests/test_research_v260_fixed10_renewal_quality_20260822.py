from __future__ import annotations

from inspect import signature

from quant.main.research_v260_fixed10_renewal_quality_20260822 import (
    MAX_HOLD_RENEWAL_SCORE_THRESHOLDS,
    candidate_id,
    candidate_policy,
    selection_key,
)
from quant.main.research_v260_runtime import fixed10_renewal_quality_v109


def test_renewal_quality_budget_is_coarse_and_bounded() -> None:
    assert MAX_HOLD_RENEWAL_SCORE_THRESHOLDS == (None, 0.80, 0.85, 0.90)


def test_candidate_only_adds_max_hold_renewal_score() -> None:
    off = candidate_policy(None)
    active = candidate_policy(0.85)
    changed = {key for key in active if active[key] != off.get(key)}
    assert changed == {"max_hold_renewal_score"}
    assert active["max_daily_score_sells"] == 1
    assert active["sell_score_below"] == 0.85
    assert active["replacement_advantage"] == 0.05
    assert active["max_hold_days"] == 25
    assert active["renewal_policy"] == "no_score_exit_rebalance"


def test_candidate_ids_are_stable() -> None:
    assert candidate_id(None) == "renewal_score_gate_off"
    assert candidate_id(0.85) == "renewal_score_gate_0.85"


def test_runtime_defaults_to_disabled_renewal_quality_gate() -> None:
    parameter = signature(fixed10_renewal_quality_v109.simulate).parameters[
        "max_hold_renewal_score_override"
    ]
    assert parameter.default is None


def test_selection_prioritizes_training_sharpe() -> None:
    def item(sharpe: float, drawdown: float) -> dict:
        return {
            "train_2022_2024": {
                "sharpe": sharpe,
                "cagr": 0.2,
                "max_drawdown": drawdown,
                "turnover_annualized": 10.0,
            },
        }

    assert selection_key(item(1.0, 0.4)) > selection_key(item(0.9, 0.2))
