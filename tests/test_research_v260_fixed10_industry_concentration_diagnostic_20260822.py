from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


MAIN_ROOT = Path(r"D:/work/quant/quant_mcp/quant/main")
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_industry_concentration_diagnostic_20260822 as module


def test_daily_concentration_counts_largest_group() -> None:
    frame = pd.DataFrame(
        [
            {"buy_date": "20220104", "stock_code": "A", "industry": "bank"},
            {"buy_date": "20220104", "stock_code": "B", "industry": "bank"},
            {"buy_date": "20220104", "stock_code": "C", "industry": "tech"},
        ]
    )
    result = module.daily_concentration(frame).iloc[0]
    assert result["position_count"] == 3
    assert result["industry_count"] == 2
    assert result["max_industry_positions"] == 2
    assert result["max_industry_share"] == 2 / 3


def test_summary_reports_threshold_days() -> None:
    frame = pd.DataFrame(
        {
            "industry_count": [8, 7, 6],
            "max_industry_positions": [2, 3, 4],
        }
    )
    result = module.summary(frame)
    assert result["days_max_industry_at_least_3"] == 2
    assert result["days_max_industry_at_least_4"] == 1
    assert result["maximum_industry_positions"] == 4
