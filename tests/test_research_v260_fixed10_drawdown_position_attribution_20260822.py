from __future__ import annotations

import inspect
import sys
from pathlib import Path


MAIN_ROOT = Path(r"D:/work/quant/quant_mcp/quant/main")
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_drawdown_position_attribution_20260822 as module
import research_v260_fixed10_hold_optimization_20260822 as round1
from research_v260_runtime import fixed10_risk_event_v110 as runtime


def observation(date: str, records: list[dict]) -> dict:
    mark = sum(item["mark_pnl"] for item in records)
    return {
        "buy_date": date,
        "previous_equity": 100.0,
        "equity_before_trades": 100.0 + mark,
        "equity_after_trades": 100.0 + mark,
        "mark_records": records,
    }


def record(code: str, pnl: float) -> dict:
    return {
        "stock_code": code,
        "shares_before": 100.0,
        "prior_price": 10.0,
        "current_price": 10.0 + pnl / 100.0,
        "mark_pnl": pnl,
        "current_open_available": True,
    }


def test_aggregate_episode_reconciles_and_detects_concentration() -> None:
    observations = [
        observation("20240102", [record("A", -60.0), record("B", -10.0)]),
        observation("20240103", [record("A", -30.0), record("C", 20.0)]),
    ]
    actual = module.aggregate_episode(observations, "20240101", "20240103")
    assert actual["classification"] == "concentrated_position_drawdown"
    assert actual["negative_stock_count"] == 2
    assert actual["positive_stock_count"] == 1
    assert actual["mark_accounting_difference"] == 0.0
    assert actual["worst_stock_mark_contributors"][0]["stock_code"] == "A"


def test_position_observer_defaults_to_disabled() -> None:
    assert inspect.signature(runtime.simulate).parameters["position_observer"].default is None
    assert inspect.signature(round1.run_fixed10).parameters["position_observer"].default is None
