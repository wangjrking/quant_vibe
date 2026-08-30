from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


MAIN_ROOT = Path(r"D:/work/quant/quant_mcp/quant/main")
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_turnover_source_attribution_20260822 as subject


@pytest.mark.parametrize(
    ("action", "before", "after", "expected"),
    [
        ("SELL", 100.0, 0.0, "replacement_exit"),
        ("SELL", 200.0, 100.0, "equalweight_trim"),
        ("BUY", 0.0, 100.0, "new_entry"),
        ("BUY", 100.0, 200.0, "equalweight_topup"),
    ],
)
def test_classify_trade(action, before, after, expected):
    assert subject.classify_trade(action, 0.1, before, after) == expected


def test_classify_trade_rejects_non_moving_action():
    with pytest.raises(ValueError, match="must reduce shares"):
        subject.classify_trade("SELL", 0.1, 100.0, 100.0)


def test_attribute_turnover_reconciles_all_categories():
    daily = pd.DataFrame(
        [{"date": "20220105", "turnover": 0.4}]
    )
    actions = pd.DataFrame(
        [
            {"buy_date": "20220105", "action": "SELL", "stock_code": "A", "target_pct": 0.0, "execution_open_raw": 10.0},
            {"buy_date": "20220105", "action": "SELL", "stock_code": "B", "target_pct": 0.1, "execution_open_raw": 10.0},
            {"buy_date": "20220105", "action": "BUY", "stock_code": "C", "target_pct": 0.1, "execution_open_raw": 10.0},
            {"buy_date": "20220105", "action": "BUY", "stock_code": "D", "target_pct": 0.1, "execution_open_raw": 10.0},
        ]
    )
    observations = [
        {
            "buy_date": "20220105",
            "equity_before_trades": 10_000.0,
            "mark_records": [
                {"stock_code": "A", "shares_before": 100.0},
                {"stock_code": "B", "shares_before": 200.0},
                {"stock_code": "D", "shares_before": 100.0},
            ],
            "positions_after_trades": {
                "B": {"shares": 100.0},
                "C": {"shares": 100.0},
                "D": {"shares": 200.0},
            },
        }
    ]
    result = subject.attribute_turnover(daily, actions, observations)
    assert result["reconciliation"]["passed"] is True
    assert result["replacement_turnover_share"] == pytest.approx(0.5)
    assert result["equalweight_maintenance_turnover_share"] == pytest.approx(0.5)
    assert all(
        result["by_category"][category]["action_count"] == 1
        for category in subject.CATEGORIES
    )


def test_attribute_turnover_rejects_duplicate_same_day_stock_actions():
    daily = pd.DataFrame([{"date": "20220105", "turnover": 0.2}])
    actions = pd.DataFrame(
        [
            {"buy_date": "20220105", "action": "BUY", "stock_code": "A", "target_pct": 0.1, "execution_open_raw": 10.0},
            {"buy_date": "20220105", "action": "BUY", "stock_code": "A", "target_pct": 0.1, "execution_open_raw": 10.0},
        ]
    )
    observations = [
        {
            "buy_date": "20220105",
            "equity_before_trades": 10_000.0,
            "mark_records": [],
            "positions_after_trades": {"A": {"shares": 200.0}},
        }
    ]
    with pytest.raises(ValueError, match="multiple same-day actions"):
        subject.attribute_turnover(daily, actions, observations)
