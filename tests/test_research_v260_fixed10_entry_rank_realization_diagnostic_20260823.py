from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_entry_rank_realization_diagnostic_20260823 as mod


def test_implied_entry_rank_maps_linear_schedule_and_rejects_refill() -> None:
    schedule = np.linspace(1.10, 0.90, 10) / 10.0
    assert mod.implied_entry_rank(schedule[0]) == 1
    assert mod.implied_entry_rank(schedule[-1]) == 10
    assert mod.implied_entry_rank(0.10) is None


def test_completed_lifecycle_compounds_marks_until_sell() -> None:
    actions = pd.DataFrame(
        [
            {"buy_date": "20240102", "action": "BUY", "stock_code": "A", "target_pct": 0.11},
            {"buy_date": "20240104", "action": "SELL", "stock_code": "A", "target_pct": 0.0},
        ]
    )
    records = [
        {"buy_date": "20240102", "mark_records": []},
        {"buy_date": "20240103", "mark_records": [{"stock_code": "A", "shares_before": 100, "prior_price": 10.0, "mark_pnl": 100.0}]},
        {"buy_date": "20240104", "mark_records": [{"stock_code": "A", "shares_before": 100, "prior_price": 11.0, "mark_pnl": -55.0}]},
    ]
    frame, open_positions = mod.completed_position_lifecycles(records, actions)
    assert open_positions == 0
    assert frame.iloc[0]["entry_rank"] == 1
    assert frame.iloc[0]["holding_days"] == 2
    assert np.isclose(frame.iloc[0]["lifecycle_return"], 0.045)


def test_open_lifecycle_is_counted_but_not_reported_as_completed() -> None:
    actions = pd.DataFrame(
        [{"buy_date": "20240102", "action": "BUY", "stock_code": "A", "target_pct": 0.10}]
    )
    frame, open_positions = mod.completed_position_lifecycles(
        [{"buy_date": "20240102", "mark_records": []}], actions
    )
    assert frame.empty
    assert open_positions == 1
