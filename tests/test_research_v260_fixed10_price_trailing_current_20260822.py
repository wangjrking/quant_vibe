from __future__ import annotations

import inspect
import sys
from pathlib import Path


MAIN_ROOT = Path(r"D:/work/quant/quant_mcp/quant/main")
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_price_trailing_current_20260822 as module
from research_v260_runtime import fixed10_risk_event_v110 as runtime


def test_candidate_changes_only_price_trailing_rule() -> None:
    base = {"sell_score_below": 0.85, "price_peak_drawdown_exit": None}
    candidate = module.candidate_policy(base, 0.15)
    assert module.policy_difference(base, candidate) == {
        "price_peak_drawdown_exit"
    }
    assert candidate["sell_score_below"] == 0.85


def test_runtime_exposes_optional_price_trailing_override() -> None:
    parameter = inspect.signature(runtime.simulate).parameters[
        "price_peak_drawdown_exit_override"
    ]
    assert parameter.default is None


def test_utility_rewards_return_and_penalizes_drawdown_and_turnover() -> None:
    base = {
        "cagr": 0.20,
        "sharpe": 1.0,
        "max_drawdown": 0.30,
        "turnover_annualized": 20.0,
    }
    better = {**base, "cagr": 0.21, "max_drawdown": 0.29}
    assert module.utility(better) > module.utility(base)
