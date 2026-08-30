from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import os
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA = ROOT / "quant" / "data_file"
REPORT_DIR = DATA / "reports" / "strategy_agent_adaptive70w_fouryear_fullwindow_20260629"
OUT_DIR = DATA / "reports" / "strategy_agent_adaptive70w_fouryear_pricecap150_admission_20260629"
STRATEGY_DIR = DATA / "reports" / "strategy_agent_dynamic_slippage_70w_20260628" / "code_snapshot_dynamic_slippage"
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")
MARKET_DB = DATA / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
SCORE_DB = REPORT_DIR / "scores" / "grid_scores.duckdb"
CASE_NAME = "w72_23_05_amt150_mv30__hold3m5_c097_e096_ddtight_pos65_pricecap150"
SIGNAL_FILE = REPORT_DIR / "signals" / f"{CASE_NAME}.csv"
SCORE_TABLE = f"score_{CASE_NAME}"

SLICES = [
    ("full", "2022-06-07 09:00:00", "2026-06-26 15:30:00"),
    ("recent120", "2025-12-24 09:00:00", "2026-06-26 15:30:00"),
    ("recent60", "2026-03-25 09:00:00", "2026-06-26 15:30:00"),
    ("ytd2026", "2026-01-05 09:00:00", "2026-06-26 15:30:00"),
]

BASE_ENV = {
    "GM_OPEN_DAILY_SCORE_EXIT": "1",
    "GM_MAX_DAILY_SELLS": "0",
    "GM_LIGHT_STOP_LOSS_PCT": "none",
    "GM_LOG_EXPOSURE": "1",
    "GM_SKIP_OPEN_LIMIT_UP_BUY": "1",
    "GM_SYNC_POSITIONS": "1",
    "GM_CASH_BUFFER": "0.99",
    "GM_VERBOSE_TRADES": "1",
    "GM_FORCE_SELL_MARKET_ORDER": "0",
    "GM_FORCE_BUY_MARKET_ORDER": "0",
    "GM_INTRADAY_RISK_MODE": "1",
    "GM_INTRADAY_REPLACE_BUY": "0",
    "GM_INTRADAY_RISK_TIMES": "10:00:00,11:00:00,14:30:00",
    "GM_EQUITY_DD_RISK_MODE": "1",
    "GM_EQUITY_DD_RESIZE_EXISTING": "0",
    "GM_EQUITY_DD_SOFT_TRIGGER": "0.06",
    "GM_EQUITY_DD_HARD_TRIGGER": "0.10",
    "GM_EQUITY_DD_RECOVER_TRIGGER": "0.03",
    "GM_EQUITY_DD_SOFT_SCALE": "0.65",
    "GM_EQUITY_DD_HARD_SCALE": "0.45",
    "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "1",
    "GM_SCORE_CONTINUE_ENTRY_RATIO": "0.97",
    "GM_SCORE_EXIT_ENTRY_RATIO": "0.96",
    "GM_ADAPTIVE_LIQUIDITY_SLIPPAGE": "1",
    "GM_ADAPTIVE_SLIPPAGE_ORDER_VALUE": "700000",
    "GM_ADAPTIVE_SLIPPAGE_BASE": "0.0015",
    "GM_ADAPTIVE_SLIPPAGE_IMPACT_COEF": "0.0200",
    "GM_ADAPTIVE_SLIPPAGE_CAP": "0.0065",
    "GM_ADAPTIVE_SLIPPAGE_AMOUNT_UNIT": "1000",
    "GM_ADAPTIVE_BUY_SLIPPAGE_MULT": "1.0",
    "GM_ADAPTIVE_SELL_SLIPPAGE_MULT": "1.0",
}


def _extract_indicator(log_file: Path) -> dict[str, Any] | None:
    if not log_file.exists():
        return None
    for line in reversed(log_file.read_text(encoding="utf-8", errors="ignore").splitlines()):
        if "GM_BACKTEST_INDICATOR:" not in line:
            continue
        payload = line.split("GM_BACKTEST_INDICATOR:", 1)[1].strip()
        try:
            return ast.literal_eval(payload)
        except Exception:
            return eval(payload, {"__builtins__": {}}, {"datetime": datetime_module})
    return None


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0].keys()) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _run(tag: str, start: str, end: str) -> dict[str, Any]:
    log_file = OUT_DIR / "logs" / f"{CASE_NAME}__adaptive70w_buy_sell__{tag}.log"
    indicator = _extract_indicator(log_file)
    returncode = 0
    if indicator is None:
        env = os.environ.copy()
        env.update(BASE_ENV)
        cmd = [
            str(JUEJIN_PYTHON),
            str(MAIN / "run_juejin_signal_backtest.py"),
            "--strategy-dir", str(STRATEGY_DIR),
            "--signal-file", str(SIGNAL_FILE),
            "--log-file", str(log_file),
            "--max-positions", "1",
            "--holding-days", "3",
            "--max-holding-days", "5",
            "--target-position-pct", "0.65",
            "--score-db", str(SCORE_DB),
            "--score-table", SCORE_TABLE,
            "--market-db", str(MARKET_DB),
            "--backtest-start", start,
            "--backtest-end", end,
            "--backtest-adjust", "none",
            "--backtest-initial-cash", "600000",
            "--backtest-slippage-ratio", "0",
            "--stop-loss-pct", "0.05",
            "--take-profit-pct", "0.08",
        ]
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, cwd=str(MAIN), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        returncode = proc.returncode
        indicator = _extract_indicator(log_file)
    return {
        "tag": tag,
        "returncode": returncode,
        "annual": indicator.get("pnl_ratio_annual") if indicator else None,
        "pnl_ratio": indicator.get("pnl_ratio") if indicator else None,
        "sharpe": indicator.get("sharp_ratio") if indicator else None,
        "max_drawdown": indicator.get("max_drawdown") if indicator else None,
        "win_ratio": indicator.get("win_ratio") if indicator else None,
        "open_count": indicator.get("open_count") if indicator else None,
        "close_count": indicator.get("close_count") if indicator else None,
        "log_file": str(log_file),
    }


def main() -> None:
    rows = [_run(tag, start, end) for tag, start, end in SLICES]
    _write_rows(OUT_DIR / "strict_buy_sell_slippage_summary.csv", rows)
    print(rows)


if __name__ == "__main__":
    main()

