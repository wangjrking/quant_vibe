from __future__ import annotations

import inspect

import pandas as pd

from quant.main.research_v260_runtime import fixed10_safe_v109
from quant.main.strategy_library.production.prod_v260_10d_regime_warmup_all4key_v20260724.production_code.v260_all4key_runtime import (
    production_v260_active_l4_breadth_exit_v109_20260722,
)
from quant.main.research_v260_fixed10_equalweight_maintenance_20260822 import (
    CANDIDATES,
    candidate_policy,
    maintenance_action_metrics,
    selection_key,
)


def test_candidate_budget_is_small_and_economically_distinct() -> None:
    assert len(CANDIDATES) == 4
    assert {value[0] for value in CANDIDATES.values()} == {20, 25}
    assert {value[1] for value in CANDIDATES.values()} == {
        "no_score_exit",
        "no_score_exit_rebalance",
    }


def test_candidate_policy_preserves_fixed_exit_rule() -> None:
    actual = candidate_policy(20, "no_score_exit_rebalance")
    assert actual["sell_score_below"] == 0.85
    assert actual["replacement_advantage"] == 0.05
    assert actual["max_daily_score_sells"] == 1
    assert actual["renewal_policy"] == "no_score_exit_rebalance"


def test_maintenance_actions_are_not_counted_as_position_exits() -> None:
    actions = pd.DataFrame(
        [
            {"action": "SELL", "target_pct": 0.0},
            {"action": "SELL", "target_pct": 0.1},
            {"action": "BUY", "target_pct": 0.1},
        ]
    )
    assert maintenance_action_metrics(actions) == {
        "position_exit_count": 1,
        "maintenance_sell_count": 1,
        "all_sell_action_count": 2,
    }


def test_selection_prefers_training_sharpe_before_investment() -> None:
    def item(sharpe: float, invested: float) -> dict:
        return {
            "train_2022_2024": {
                "average_invested_ratio": invested,
                "sharpe": sharpe,
                "cagr": 0.2,
                "max_drawdown": 0.3,
                "turnover_annualized": 10.0,
            },
        }

    assert selection_key(item(1.0, 0.90)) > selection_key(item(0.9, 0.99))


def test_safe_runtime_diff_is_only_invalid_open_maintenance_guard() -> None:
    original = inspect.getsource(
        production_v260_active_l4_breadth_exit_v109_20260722.simulate
    )
    safe = inspect.getsource(fixed10_safe_v109.simulate)
    guard = "            if not valid_open[idx]:\n                continue\n"
    assert guard in safe
    assert safe.replace(guard, "") == original
