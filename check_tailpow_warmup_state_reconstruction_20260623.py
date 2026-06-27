from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import math
import os
import re
import subprocess
from pathlib import Path


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_ROOT = DATA_DIR / "reports" / "strategy_agent_goal_high_annual_20260623"
OUT_DIR = REPORT_ROOT / "current_formal_top1keep_tailpow_warmup_state"
MARKET_DB = DATA_DIR / "STOCK_DAILY_DATA.db"
STRATEGY_DIR = Path(r"D:\work\dfcf\juejin\strategy\ce327750-634e-11f1-b8d7-10ffe0295517")
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")

BASE_SIGNAL = REPORT_ROOT / "current_formal_top1keep_tailpow_peak" / "signals" / "tp_peak_g1p38_o0p942.csv"
SCORE_DB = REPORT_ROOT / "current_formal_top1keep_tailpow_peak" / "scores_top1keep_tailpow_peak.db"
SCORE_TABLE = "score_tp_peak_g1p38_o0p942"
BACKTEST_END = "2026-06-23 15:30:00"

TARGET_STARTS = [
    ("target_20250701", "20250701", "2025-07-01 09:00:00"),
    ("target_20251008", "20251008", "2025-10-08 09:00:00"),
    ("target_20260105", "20260105", "2026-01-05 09:00:00"),
]

WARMUP_SIGNAL_DAYS = [0, 20, 40, 60, 80]

BASE_ENV = {
    "GM_OPEN_DAILY_SCORE_EXIT": "1",
    "GM_SCORE_EXIT_ENTRY_RATIO": "0.942",
    "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "2",
    "GM_MAX_DAILY_SELLS": "4",
    "GM_STOP_LOSS_PCT": "0.08",
    "GM_TAKE_PROFIT_PCT": "none",
    "GM_LIGHT_STOP_LOSS_PCT": "none",
    "GM_LOG_EXPOSURE": "1",
    "GM_SKIP_OPEN_LIMIT_UP_BUY": "1",
    "GM_SYNC_POSITIONS": "1",
    "GM_CASH_BUFFER": "0.99",
    "GM_VERBOSE_TRADES": "0",
    "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.02",
    "GM_EQUITY_DD_RISK_MODE": "0",
}

EXPOSURE_RE = re.compile(
    r"EXPOSURE\s+(?P<date>\d{8})\s+post_buy\s+"
    r"invested_pct=(?P<invested>[0-9.]+)\s+"
    r"active_positions=(?P<positions>\d+)\s+"
    r"market_value=(?P<market_value>[0-9.]+)\s+"
    r"nav=(?P<nav>[0-9.]+)\s+"
    r"cash=(?P<cash>[0-9.]+)"
)


def _load_buy_dates(path: Path) -> list[str]:
    dates: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            value = str(row.get("buy_date") or "").strip()
            if value:
                dates.add(value)
    return sorted(dates)


def _warmup_start_date(buy_dates: list[str], target_date: str, warmup_days: int) -> str:
    before = [date for date in buy_dates if date <= target_date]
    if not before:
        raise ValueError(f"No buy date available before target {target_date}")
    if warmup_days <= 0:
        return target_date
    index = max(0, len(before) - warmup_days - 1)
    return before[index]


def _to_backtest_start(yyyymmdd: str) -> str:
    return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:]} 09:00:00"


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


def _parse_exposures(log_file: Path) -> list[dict]:
    rows: list[dict] = []
    if not log_file.exists():
        return rows
    for line in log_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = EXPOSURE_RE.search(line)
        if not match:
            continue
        item = match.groupdict()
        rows.append(
            {
                "date": item["date"],
                "invested_pct": float(item["invested"]),
                "active_positions": int(item["positions"]),
                "market_value": float(item["market_value"]),
                "nav": float(item["nav"]),
                "cash": float(item["cash"]),
            }
        )
    return rows


def _annualize(cumulative_return: float, trading_days: int) -> float | None:
    if trading_days <= 0 or cumulative_return <= -1.0:
        return None
    return math.pow(1.0 + cumulative_return, 252.0 / trading_days) - 1.0


def _max_drawdown(nav_values: list[float]) -> float | None:
    if not nav_values:
        return None
    peak = nav_values[0]
    max_dd = 0.0
    for value in nav_values:
        peak = max(peak, value)
        if peak > 0:
            max_dd = max(max_dd, 1.0 - value / peak)
    return max_dd


def _run_case(target_tag: str, target_date: str, target_start: str, warmup_days: int, buy_dates: list[str]) -> dict:
    warmup_date = _warmup_start_date(buy_dates, target_date, warmup_days)
    backtest_start = _to_backtest_start(warmup_date)
    log_file = OUT_DIR / "logs" / f"{target_tag}_warmup{warmup_days}_{warmup_date}.log"
    indicator = _extract_indicator(log_file)
    if indicator is None:
        env = os.environ.copy()
        env.update(BASE_ENV)
        command = [
            str(JUEJIN_PYTHON),
            str(MAIN / "run_juejin_signal_backtest.py"),
            "--strategy-dir",
            str(STRATEGY_DIR),
            "--signal-file",
            str(BASE_SIGNAL),
            "--log-file",
            str(log_file),
            "--max-positions",
            "4",
            "--holding-days",
            "5",
            "--max-holding-days",
            "5",
            "--target-position-pct",
            "0.5",
            "--score-db",
            str(SCORE_DB),
            "--score-table",
            SCORE_TABLE,
            "--market-db",
            str(MARKET_DB),
            "--backtest-start",
            backtest_start,
            "--backtest-end",
            BACKTEST_END,
            "--backtest-adjust",
            "none",
            "--backtest-initial-cash",
            "600000",
            "--backtest-slippage-ratio",
            "0.0015",
        ]
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w", encoding="utf-8") as log:
            proc = subprocess.run(command, cwd=str(MAIN), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        indicator = _extract_indicator(log_file)
        returncode = proc.returncode
    else:
        returncode = 0

    exposures = _parse_exposures(log_file)
    after = [row for row in exposures if row["date"] >= target_date]
    start_row = after[0] if after else None
    end_row = after[-1] if after else None
    nav_values = [row["nav"] for row in after]
    cumulative = (end_row["nav"] / start_row["nav"] - 1.0) if start_row and end_row and start_row["nav"] else None
    trading_days = len(after)
    return {
        "target": target_tag,
        "target_date": target_date,
        "warmup_signal_days": warmup_days,
        "warmup_backtest_start_date": warmup_date,
        "backtest_start": backtest_start,
        "backtest_end": BACKTEST_END,
        "returncode": returncode,
        "juejin_full_run_annual": indicator.get("pnl_ratio_annual") if indicator else None,
        "juejin_full_run_pnl": indicator.get("pnl_ratio") if indicator else None,
        "juejin_full_run_sharpe": indicator.get("sharp_ratio") if indicator else None,
        "juejin_full_run_max_drawdown": indicator.get("max_drawdown") if indicator else None,
        "post_target_nav_start": start_row["nav"] if start_row else None,
        "post_target_nav_end": end_row["nav"] if end_row else None,
        "post_target_cumulative_return": cumulative,
        "post_target_annualized_return": _annualize(cumulative, trading_days) if cumulative is not None else None,
        "post_target_max_drawdown": _max_drawdown(nav_values),
        "post_target_trading_days": trading_days,
        "target_start_invested_pct": start_row["invested_pct"] if start_row else None,
        "target_start_active_positions": start_row["active_positions"] if start_row else None,
        "target_end_invested_pct": end_row["invested_pct"] if end_row else None,
        "target_end_active_positions": end_row["active_positions"] if end_row else None,
        "log_file": str(log_file),
    }


def main() -> int:
    buy_dates = _load_buy_dates(BASE_SIGNAL)
    rows = []
    for target_tag, target_date, target_start in TARGET_STARTS:
        for warmup_days in WARMUP_SIGNAL_DAYS:
            row = _run_case(target_tag, target_date, target_start, warmup_days, buy_dates)
            rows.append(row)
            print(
                f"{target_tag} warmup={warmup_days} start={row['warmup_backtest_start_date']} "
                f"post_annual={row['post_target_annualized_return']} "
                f"post_pnl={row['post_target_cumulative_return']} "
                f"pos={row['target_start_active_positions']} invested={row['target_start_invested_pct']}",
                flush=True,
            )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = OUT_DIR / "warmup_state_summary.csv"
    with summary.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (OUT_DIR / "warmup_state_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
