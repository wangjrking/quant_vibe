from __future__ import annotations

import json
import sys
from pathlib import Path


MAIN_DIR = Path(__file__).resolve().parents[1]
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

import run_all_a_raw_update as raw_update


RISK_EVENT_TABLES = {"stk_shock", "stk_high_shock", "stk_alert"}
WORKFLOW_PATH = MAIN_DIR / "workflows" / "standard_incremental_trading_signal_l1_l8.json"


def test_l1_daily_default_includes_all_risk_event_tables() -> None:
    configured = {
        item.strip() for item in raw_update.DEFAULT_TABLES.split(",") if item.strip()
    }

    assert RISK_EVENT_TABLES <= configured


def test_l2_daily_workflow_requires_risk_event_companion() -> None:
    workflow = json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))
    l2_stage = next(
        stage
        for stage in workflow["execution_route"]["stages"]
        if stage["layer"] == "L2"
    )

    assert "integrate_l2_stock_risk_events_daily.py" in l2_stage.get(
        "companion_entrypoints", []
    )
