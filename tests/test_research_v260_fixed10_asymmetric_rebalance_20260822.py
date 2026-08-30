from __future__ import annotations

from inspect import signature

from quant.main.research_v260_fixed10_asymmetric_rebalance_20260822 import (
    OVERWEIGHT_BANDS,
    UNDERWEIGHT_BAND,
    candidate_id,
)
from quant.main.research_v260_fixed10_renewal_maintenance_20260822 import selection_key
from quant.main.research_v260_runtime import fixed10_asymmetric_rebalance_v111


def test_candidate_budget_is_coarse() -> None:
    assert OVERWEIGHT_BANDS == (0.01, 0.02, 0.03, 10.0)
    assert UNDERWEIGHT_BAND == 0.01


def test_candidate_ids_express_absolute_trim_level() -> None:
    assert candidate_id(0.01) == "trim_above_11pct"
    assert candidate_id(0.02) == "trim_above_12pct"
    assert candidate_id(0.03) == "trim_above_13pct"
    assert candidate_id(10.0) == "never_trim_overweight"


def test_runtime_asymmetric_controls_default_to_disabled() -> None:
    parameters = signature(fixed10_asymmetric_rebalance_v111.simulate).parameters
    assert parameters["portfolio_rebalance_overweight_deviation_override"].default is None
    assert parameters["portfolio_rebalance_underweight_deviation_override"].default is None


def test_aggregate_training_key_does_not_promote_worse_train_metrics() -> None:
    current = {"train_2022_2024": {"sharpe": 0.90, "cagr": 0.31, "max_drawdown": 0.39, "turnover_annualized": 22.0}}
    later_holdout_winner = {"train_2022_2024": {"sharpe": 0.89, "cagr": 0.30, "max_drawdown": 0.40, "turnover_annualized": 21.0}}
    assert selection_key(current) > selection_key(later_holdout_winner)
