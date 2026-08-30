from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import os
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA = ROOT / "quant" / "data_file"
SIGNAL_ROOT = DATA / "reports" / "strategy_agent_adaptive70w_fouryear_momentum_cooldown_focused_20260629"
OUT_DIR = DATA / "reports" / "strategy_agent_adaptive70w_fouryear_pos48_sell_micro_20260629"
STRATEGY_DIR = DATA / "reports" / "strategy_agent_dynamic_slippage_70w_20260628" / "code_snapshot_dynamic_slippage"
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")
MARKET_DB = DATA / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
SCORE_DB = DATA / "reports" / "strategy_agent_adaptive70w_fouryear_fullwindow_20260629" / "scores" / "grid_scores.duckdb"

CASE_KEY = "w72_23_05_amt150_mv30__hold3m5_c097_e096_ddtight_pos65__amt150000_mv300000_pc150_pos48_cool2d20"
SIGNAL_FILE = SIGNAL_ROOT / "signals" / f"{CASE_KEY}.csv"
SCORE_TABLE = f"score_{CASE_KEY}"
SIGNAL_COLUMNS_TO_OVERRIDE = (
    "score_exit_entry_ratio",
    "signal_score_exit_entry_ratio",
    "score_continue_entry_ratio",
    "signal_score_continue_entry_ratio",
    "min_holding_days_before_score_exit",
    "signal_min_holding_days_before_score_exit",
)

SLICES = [
    ("full", "2022-06-07 09:00:00", "2026-06-26 15:30:00"),
    ("recent120", "2025-12-24 09:00:00", "2026-06-26 15:30:00"),
    ("recent60", "2026-03-25 09:00:00", "2026-06-26 15:30:00"),
    ("ytd2026", "2026-01-05 09:00:00", "2026-06-26 15:30:00"),
]

DD_ENV = {
    "GM_EQUITY_DD_RISK_MODE": "1",
    "GM_EQUITY_DD_RESIZE_EXISTING": "0",
    "GM_EQUITY_DD_SOFT_TRIGGER": "0.07",
    "GM_EQUITY_DD_HARD_TRIGGER": "0.12",
    "GM_EQUITY_DD_RECOVER_TRIGGER": "0.035",
    "GM_EQUITY_DD_SOFT_SCALE": "0.75",
    "GM_EQUITY_DD_HARD_SCALE": "0.55",
    "GM_EQUITY_DD_STRICT_WHEN_DRAWDOWN": "1",
    "GM_EQUITY_DD_STRICT_TRIGGER": "0.08",
    "GM_EQUITY_DD_STRICT_SOFT_TRIGGER": "0.06",
    "GM_EQUITY_DD_STRICT_HARD_TRIGGER": "0.10",
    "GM_EQUITY_DD_STRICT_RECOVER_TRIGGER": "0.03",
    "GM_EQUITY_DD_STRICT_SOFT_SCALE": "0.65",
    "GM_EQUITY_DD_STRICT_HARD_SCALE": "0.45",
}

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
    "GM_ADAPTIVE_SELL_SLIPPAGE_MULT": "0.25",
    **DD_ENV,
}

SELL_CASES = [
    {
        "name": "baseline",
        "score_continue_ratio": 0.97,
        "score_exit_ratio": 0.96,
        "min_score_exit_days": 1,
        "day_drop_ratio": None,
    },
    {
        "name": "daydrop97",
        "score_continue_ratio": 0.97,
        "score_exit_ratio": 0.96,
        "min_score_exit_days": 1,
        "day_drop_ratio": 0.97,
    },
    {
        "name": "daydrop96",
        "score_continue_ratio": 0.97,
        "score_exit_ratio": 0.96,
        "min_score_exit_days": 1,
        "day_drop_ratio": 0.96,
    },
    {
        "name": "daydrop95",
        "score_continue_ratio": 0.97,
        "score_exit_ratio": 0.96,
        "min_score_exit_days": 1,
        "day_drop_ratio": 0.95,
    },
    {
        "name": "daydrop94",
        "score_continue_ratio": 0.97,
        "score_exit_ratio": 0.96,
        "min_score_exit_days": 1,
        "day_drop_ratio": 0.94,
    },
    {
        "name": "e095_daydrop95",
        "score_continue_ratio": 0.97,
        "score_exit_ratio": 0.95,
        "min_score_exit_days": 1,
        "day_drop_ratio": 0.95,
    },
]


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fields} for row in rows])


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


def _signal_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig") as file:
        return max(sum(1 for _ in file) - 1, 0)


def _load_signal_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _ensure_case_signal(case: dict[str, Any], base_rows: list[dict[str, Any]]) -> Path:
    signal_path = OUT_DIR / "signals" / f"{CASE_KEY}__{case['name']}.csv"
    if signal_path.exists():
        return signal_path
    rows: list[dict[str, Any]] = []
    for row in base_rows:
        item = dict(row)
        for key in SIGNAL_COLUMNS_TO_OVERRIDE:
            item.pop(key, None)
        item["score_exit_entry_ratio"] = f"{case['score_exit_ratio']:.5f}"
        item["signal_score_exit_entry_ratio"] = f"{case['score_exit_ratio']:.5f}"
        item["score_continue_entry_ratio"] = f"{case['score_continue_ratio']:.5f}"
        item["signal_score_continue_entry_ratio"] = f"{case['score_continue_ratio']:.5f}"
        item["min_holding_days_before_score_exit"] = str(case["min_score_exit_days"])
        item["signal_min_holding_days_before_score_exit"] = str(case["min_score_exit_days"])
        rows.append(item)
    _write_rows(signal_path, rows)
    return signal_path


def _env_for_case(case: dict[str, Any]) -> dict[str, str]:
    env = dict(BASE_ENV)
    env["GM_SCORE_CONTINUE_ENTRY_RATIO"] = str(case["score_continue_ratio"])
    env["GM_SCORE_EXIT_ENTRY_RATIO"] = str(case["score_exit_ratio"])
    env["GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT"] = str(case["min_score_exit_days"])
    env["GM_SCORE_STOP_LOSS_DAY_DROP_RATIO"] = "none" if case["day_drop_ratio"] is None else str(case["day_drop_ratio"])
    return env


def _run_case(case: dict[str, Any], signal_file: Path, tag: str, start: str, end: str, signal_rows: int) -> dict[str, Any]:
    log_file = OUT_DIR / "logs" / f"{CASE_KEY}__{case['name']}__{tag}.log"
    indicator = _extract_indicator(log_file)
    returncode = 0
    if indicator is None:
        env = os.environ.copy()
        env.update(_env_for_case(case))
        cmd = [
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
            "3",
            "--max-holding-days",
            "5",
            "--target-position-pct",
            "0.48",
            "--score-db",
            str(SCORE_DB),
            "--score-table",
            SCORE_TABLE,
            "--market-db",
            str(MARKET_DB),
            "--backtest-start",
            start,
            "--backtest-end",
            end,
            "--backtest-adjust",
            "none",
            "--backtest-initial-cash",
            "600000",
            "--backtest-slippage-ratio",
            "0",
            "--stop-loss-pct",
            "0.05",
            "--take-profit-pct",
            "0.08",
        ]
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, cwd=str(MAIN), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        returncode = proc.returncode
        indicator = _extract_indicator(log_file)
    return {
        "case_name": case["name"],
        "slice": tag,
        "returncode": returncode,
        "signal_rows": signal_rows,
        "signal_file": str(signal_file),
        "score_continue_ratio": case["score_continue_ratio"],
        "score_exit_ratio": case["score_exit_ratio"],
        "min_score_exit_days": case["min_score_exit_days"],
        "day_drop_ratio": case["day_drop_ratio"],
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
    if not SIGNAL_FILE.exists():
        raise FileNotFoundError(f"signal file not found: {SIGNAL_FILE}")
    signal_rows = _signal_rows(SIGNAL_FILE)
    base_rows = _load_signal_rows(SIGNAL_FILE)
    detail: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for case in SELL_CASES:
        case_signal_file = _ensure_case_signal(case, base_rows)
        rows = [_run_case(case, case_signal_file, tag, start, end, signal_rows) for tag, start, end in SLICES]
        detail.extend(rows)
        by_slice = {row["slice"]: row for row in rows}
        out = {
            "case_name": case["name"],
            "signal_rows": signal_rows,
            "signal_file": str(case_signal_file),
            "score_continue_ratio": case["score_continue_ratio"],
            "score_exit_ratio": case["score_exit_ratio"],
            "min_score_exit_days": case["min_score_exit_days"],
            "day_drop_ratio": case["day_drop_ratio"],
        }
        for slice_name in ["full", "recent120", "recent60", "ytd2026"]:
            row = by_slice[slice_name]
            for key in ["annual", "pnl_ratio", "sharpe", "max_drawdown", "win_ratio", "open_count", "close_count", "log_file"]:
                out[f"{slice_name}_{key}"] = row.get(key)
        summary.append(out)
        print(json.dumps(out, ensure_ascii=False), flush=True)
        _write_rows(OUT_DIR / "sell_micro_detail.csv", detail)
        _write_rows(OUT_DIR / "sell_micro_summary.csv", summary)


if __name__ == "__main__":
    main()

