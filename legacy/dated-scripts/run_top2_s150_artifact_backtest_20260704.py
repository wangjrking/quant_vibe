from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


ROOT = Path("D:/work/quant/quant_mcp")
REPORT_DIR = ROOT / "quant/data_file/reports/strategy_agent_active_l4_prod_repro_grid_20260703"
OUT_DIR = REPORT_DIR / "production_only_juejin_20260704"
STRATEGY_DIR = OUT_DIR / "code_snapshot_sell_available_safe"
SIGNAL_FILE = OUT_DIR / "executable_top2_open_gap_market_signals/top2_s150_cap82_w62.csv"
ARTIFACT_DIR = OUT_DIR / "top2_s150_cap82_w62_artifacts"
LOG_FILE = ARTIFACT_DIR / "run.log"
SCORE_DB = REPORT_DIR / "score_current.duckdb"
MARKET_DB = ROOT / "quant/data_file/production_assets/duckdb/l2_stock_daily_data.duckdb"


def infer_window() -> tuple[str, str]:
    df = pd.read_csv(SIGNAL_FILE, dtype={"buy_date": str})
    dates = [datetime.strptime(v, "%Y%m%d") for v in df["buy_date"].dropna().astype(str)]
    start = min(dates).strftime("%Y-%m-%d 09:00:00")
    end = (max(dates) + timedelta(days=12)).strftime("%Y-%m-%d 15:30:00")
    return start, end


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    for old in ["execution_reports.jsonl", "order_status.jsonl", "backtest_trade_artifacts.json", "run.log"]:
        path = ARTIFACT_DIR / old
        if path.exists():
            path.unlink()
    start, end = infer_window()
    env = os.environ.copy()
    env.update(
        {
            "GM_SIGNAL_FILE": str(SIGNAL_FILE),
            "GM_BACKTEST_ARTIFACT_DIR": str(ARTIFACT_DIR),
            "GM_BACKTEST_START": start,
            "GM_BACKTEST_END": end,
            "GM_MAX_POSITIONS": "2",
            "GM_HOLDING_DAYS": "2",
            "GM_TARGET_POSITION_PCT": "0.82",
            "GM_SCORE_DB": str(SCORE_DB),
            "GM_SCORE_TABLE": "score",
            "GM_MARKET_DB": str(MARKET_DB),
            "GM_SCORE_EXIT_ENTRY_RATIO": "0.900",
            "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "2",
            "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.000",
            "GM_MAX_HOLDING_DAYS": "4",
            "GM_LIGHT_STOP_LOSS_PCT": "0.080",
            "GM_MIN_HOLDING_DAYS_BEFORE_LIGHT_STOP": "1",
            "GM_BACKTEST_ADJUST": "none",
            "GM_BACKTEST_SLIPPAGE_RATIO": "0.0015",
        }
    )
    with LOG_FILE.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            [sys.executable, "main.py"],
            cwd=str(STRATEGY_DIR),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    result = {
        "returncode": proc.returncode,
        "signal_file": str(SIGNAL_FILE),
        "artifact_dir": str(ARTIFACT_DIR),
        "log_file": str(LOG_FILE),
        "backtest_start": start,
        "backtest_end": end,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
