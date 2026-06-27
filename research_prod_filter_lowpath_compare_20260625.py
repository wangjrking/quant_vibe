from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import os
import re
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA = ROOT / "quant" / "data_file"
REPORT_DIR = DATA / "reports" / "strategy_agent_prod_filter_lowpath_compare_20260625"
STRATEGY_DIR = MAIN / "strategy_library" / "production" / "prod_dynamic_h2m3_pos8975_scale7555_v20260625" / "code_snapshot"
SCORE_DB = DATA / "reports" / "strategy_agent_goal_high_annual_20260623" / "top1_no_delist_rerun_20260624" / "diversification_tune_20260624" / "scores_diversification.db"
SCORE_TABLE = "score_div_top1_w90_5d10_h5_e099"
MARKET_DB = DATA / "STOCK_DAILY_DATA.db"
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")

BASE_ENV = {
    "GM_OPEN_DAILY_SCORE_EXIT": "1",
    "GM_MAX_DAILY_SELLS": "1",
    "GM_LIGHT_STOP_LOSS_PCT": "none",
    "GM_LOG_EXPOSURE": "1",
    "GM_SKIP_OPEN_LIMIT_UP_BUY": "1",
    "GM_SYNC_POSITIONS": "1",
    "GM_CASH_BUFFER": "0.99",
    "GM_VERBOSE_TRADES": "0",
    "GM_FORCE_SELL_MARKET_ORDER": "1",
    "GM_INTRADAY_RISK_MODE": "1",
    "GM_INTRADAY_REPLACE_BUY": "0",
    "GM_INTRADAY_RISK_TIMES": "10:00:00,11:00:00,14:30:00",
    "GM_EQUITY_DD_RISK_MODE": "1",
    "GM_EQUITY_DD_SOFT_TRIGGER": "0.07",
    "GM_EQUITY_DD_HARD_TRIGGER": "0.115",
    "GM_EQUITY_DD_RECOVER_TRIGGER": "0.03",
    "GM_EQUITY_DD_SOFT_SCALE": "0.75",
    "GM_EQUITY_DD_HARD_SCALE": "0.55",
    "GM_EQUITY_DD_RESIZE_EXISTING": "0",
    "GM_SCORE_EXIT_ENTRY_RATIO": "0.97",
    "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "1",
    "GM_SCORE_CONTINUE_ENTRY_RATIO": "0.975",
}

SIGNALS = {
    "base_amt10w_mv20w": DATA / "reports" / "strategy_agent_prod_filter_narrow_grid_20260625" / "signals" / "prod_filter_base_amt10w_mv20w.csv",
    "amt8w_mv20w": DATA / "reports" / "strategy_agent_prod_filter_narrow_grid_20260625" / "signals" / "prod_filter_amt8w_mv20w.csv",
}

ANCHORS = {
    "20250701": ["2025-06-26 09:00:00", "2025-06-27 09:00:00", "2025-06-30 09:00:00", "2025-07-01 09:00:00", "2025-07-02 09:00:00", "2025-07-03 09:00:00", "2025-07-04 09:00:00"],
    "20251009": ["2025-09-26 09:00:00", "2025-09-29 09:00:00", "2025-09-30 09:00:00", "2025-10-08 09:00:00", "2025-10-09 09:00:00", "2025-10-10 09:00:00", "2025-10-13 09:00:00"],
    "20260105": ["2025-12-29 09:00:00", "2025-12-30 09:00:00", "2025-12-31 09:00:00", "2026-01-05 09:00:00", "2026-01-06 09:00:00", "2026-01-07 09:00:00", "2026-01-08 09:00:00"],
}


def _extract_indicator(log_file: Path) -> dict | None:
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


def _run(signal_name: str, signal_file: Path, anchor: str, start: str) -> dict:
    safe = start.split(" ")[0].replace("-", "")
    log_file = REPORT_DIR / "logs" / f"{signal_name}_{anchor}_{safe}.log"
    env = os.environ.copy()
    env.update(BASE_ENV)
    command = [
        str(JUEJIN_PYTHON),
        str(MAIN / "run_juejin_signal_backtest.py"),
        "--strategy-dir",
        str(STRATEGY_DIR),
        "--signal-file",
        str(signal_file),
        "--log-file",
        str(log_file),
        "--max-positions",
        "1",
        "--holding-days",
        "2",
        "--max-holding-days",
        "3",
        "--target-position-pct",
        "0.8975",
        "--score-db",
        str(SCORE_DB),
        "--score-table",
        SCORE_TABLE,
        "--market-db",
        str(MARKET_DB),
        "--backtest-start",
        start,
        "--backtest-end",
        "2026-06-23 15:30:00",
        "--backtest-adjust",
        "none",
        "--backtest-initial-cash",
        "600000",
        "--backtest-slippage-ratio",
        "0.0015",
        "--stop-loss-pct",
        "0.06",
        "--take-profit-pct",
        "0.07",
    ]
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as log:
        proc = subprocess.run(command, cwd=str(MAIN), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    indicator = _extract_indicator(log_file)
    return {
        "signal_name": signal_name,
        "anchor": anchor,
        "start": start,
        "returncode": proc.returncode,
        "annual": indicator.get("pnl_ratio_annual") if indicator else None,
        "pnl_ratio": indicator.get("pnl_ratio") if indicator else None,
        "sharpe": indicator.get("sharp_ratio") if indicator else None,
        "max_drawdown": indicator.get("max_drawdown") if indicator else None,
        "win_ratio": indicator.get("win_ratio") if indicator else None,
        "open_count": indicator.get("open_count") if indicator else None,
        "log_file": str(log_file),
    }


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for signal_name, signal_file in SIGNALS.items():
        for anchor, starts in ANCHORS.items():
            for start in starts:
                rows.append(_run(signal_name, signal_file, anchor, start))
    _write_rows(REPORT_DIR / "detail.csv", rows)
    summary = []
    for signal_name in SIGNALS:
        for anchor in ANCHORS:
            items = [row for row in rows if row["signal_name"] == signal_name and row["anchor"] == anchor and row["annual"] is not None]
            annuals = [float(row["annual"]) for row in items]
            sharpes = [float(row["sharpe"]) for row in items if row["sharpe"] is not None]
            drawdowns = [float(row["max_drawdown"]) for row in items if row["max_drawdown"] is not None]
            summary.append(
                {
                    "signal_name": signal_name,
                    "anchor": anchor,
                    "starts": len(items),
                    "annual_min": min(annuals) if annuals else None,
                    "annual_median": float(pd.Series(annuals).median()) if annuals else None,
                    "annual_max": max(annuals) if annuals else None,
                    "sharpe_min": min(sharpes) if sharpes else None,
                    "sharpe_median": float(pd.Series(sharpes).median()) if sharpes else None,
                    "max_drawdown_max": max(drawdowns) if drawdowns else None,
                }
            )
    _write_rows(REPORT_DIR / "summary.csv", summary)
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report_dir": str(REPORT_DIR), "rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
