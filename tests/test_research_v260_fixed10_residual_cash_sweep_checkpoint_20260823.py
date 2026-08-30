from __future__ import annotations

import pandas as pd

import research_v260_fixed10_residual_cash_sweep_checkpoint_20260823 as module


def test_action_integrity_accepts_unique_real_actions() -> None:
    frame = pd.DataFrame(
        [
            {
                "signal_date": "20220104",
                "buy_date": "20220105",
                "action": "BUY",
                "stock_code": "000001.SZ",
                "execution_open_raw": 10.0,
            }
        ]
    )
    assert all(module.action_integrity(frame).values())


def test_action_integrity_rejects_duplicate_order_key() -> None:
    row = {
        "signal_date": "20220104",
        "buy_date": "20220105",
        "action": "BUY",
        "stock_code": "000001.SZ",
        "execution_open_raw": 10.0,
    }
    actual = module.action_integrity(pd.DataFrame([row, row]))
    assert actual["unique_action_keys"] is False
