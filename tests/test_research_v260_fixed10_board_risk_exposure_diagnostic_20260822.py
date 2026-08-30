from __future__ import annotations

import sys
from pathlib import Path


MAIN_ROOT = Path(r"D:/work/quant/quant_mcp/quant/main")
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_board_risk_exposure_diagnostic_20260822 as module


def test_build_exposure_counts_twenty_percent_boards() -> None:
    observations = [
        {
            "signal_date": "20240102",
            "buy_date": "20240103",
            "positions_after_trades": {
                "000001.SZ": {},
                "300001.SZ": {},
                "688001.SH": {},
            },
        }
    ]
    row = module.build_exposure(observations).iloc[0]
    assert row["high_limit_board_positions"] == 2
    assert row["high_limit_board_share"] == 2 / 3
