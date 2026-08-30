from __future__ import annotations

import pandas as pd
import pytest

from research_v260_fixed10_drawdown_position_age_attribution_20260822 import (
    age_band,
    reconstruct_mark_ages,
    summarize_episode,
)


def test_age_band_boundaries_are_explicit() -> None:
    assert age_band(1) == "age_1_4"
    assert age_band(4) == "age_1_4"
    assert age_band(5) == "age_5_9"
    assert age_band(10) == "age_10_19"
    assert age_band(20) == "age_20_plus"
    with pytest.raises(ValueError):
        age_band(-1)


def test_reconstruction_keeps_entry_age_across_topup_and_trim() -> None:
    observations = [
        {
            "buy_date": "20220103",
            "mark_records": [],
            "positions_after_trades": {"000001.SZ": {}},
        },
        {
            "buy_date": "20220104",
            "mark_records": [
                {"stock_code": "000001.SZ", "mark_pnl": 1.0, "current_open_available": True}
            ],
            "positions_after_trades": {"000001.SZ": {}},
        },
        {
            "buy_date": "20220105",
            "mark_records": [
                {"stock_code": "000001.SZ", "mark_pnl": -2.0, "current_open_available": True}
            ],
            "positions_after_trades": {"000001.SZ": {}},
        },
    ]
    actions = pd.DataFrame(
        [
            {"buy_date": "20220103", "action": "BUY", "stock_code": "000001.SZ", "target_pct": 0.1},
            {"buy_date": "20220104", "action": "BUY", "stock_code": "000001.SZ", "target_pct": 0.1},
            {"buy_date": "20220105", "action": "SELL", "stock_code": "000001.SZ", "target_pct": 0.1},
        ]
    )
    rows = reconstruct_mark_ages(observations, actions)
    assert [row["age_sessions"] for row in rows] == [1, 2]
    assert {row["entry_date"] for row in rows} == {"20220103"}


def test_full_exit_removes_position_and_summary_reconciles_negative_share() -> None:
    observations = [
        {
            "buy_date": "20220103",
            "mark_records": [],
            "positions_after_trades": {"000001.SZ": {}},
        },
        {
            "buy_date": "20220104",
            "mark_records": [
                {"stock_code": "000001.SZ", "mark_pnl": -3.0, "current_open_available": True}
            ],
            "positions_after_trades": {},
        },
    ]
    actions = pd.DataFrame(
        [
            {"buy_date": "20220103", "action": "BUY", "stock_code": "000001.SZ", "target_pct": 0.1},
            {"buy_date": "20220104", "action": "SELL", "stock_code": "000001.SZ", "target_pct": 0.0},
        ]
    )
    rows = reconstruct_mark_ages(observations, actions)
    summary = summarize_episode(rows, "20220103", "20220104")
    assert summary["gross_negative_mark_pnl"] == -3.0
    assert summary["age_bands"]["age_1_4"]["share_of_episode_gross_negative"] == 1.0
