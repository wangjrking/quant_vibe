from __future__ import annotations

import numpy as np
import pandas as pd

from quant.main.research_v260_fixed10_hold_optimization_20260822 import (
    HOLD_POLICIES,
    action_metrics,
    selection_utility,
)


def test_hold_policy_budget_is_small_and_keeps_production_score_threshold() -> None:
    assert len(HOLD_POLICIES) == 6
    assert all(policy["sell_score_below"] == 0.85 for policy in HOLD_POLICIES.values())
    assert all(policy["min_hold_mode"] == "production_dynamic" for policy in HOLD_POLICIES.values())


def test_action_metrics_respects_validation_window() -> None:
    actions = pd.DataFrame(
        {
            "buy_date": ["20251231", "20260105", "20260106"],
            "action": ["BUY", "BUY", "SELL"],
        }
    )
    assert action_metrics(actions, "20260101", "20261231") == {
        "buy_count": 1,
        "sell_count": 1,
        "round_trips": 1,
    }


def test_selection_utility_rewards_return_and_penalizes_drawdown() -> None:
    base = {
        "cagr": 0.10,
        "sharpe": 1.0,
        "max_drawdown": 0.20,
        "turnover_annualized": 10.0,
        "annual_returns": {"2022": 0.10, "2023": 0.10},
    }
    stress = {**base, "cagr": 0.05, "sharpe": 0.5}
    better = {**base, "cagr": 0.15, "max_drawdown": 0.15}
    assert selection_utility(better, stress) > selection_utility(base, stress)


def test_all_renewal_policies_are_explicit() -> None:
    allowed = {"none", "no_score_exit"}
    assert {policy["renewal_policy"] for policy in HOLD_POLICIES.values()} <= allowed
