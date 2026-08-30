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
OUT_DIR = DATA / "reports" / "strategy_agent_adaptive70w_fouryear_pos60_daydrop96_target_grid_20260629"
STRATEGY_DIR = DATA / "reports" / "strategy_agent_dynamic_slippage_70w_20260628" / "code_snapshot_dynamic_slippage"
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")
MARKET_DB = DATA / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
SCORE_DB = DATA / "reports" / "strategy_agent_adaptive70w_fouryear_fullwindow_20260629" / "scores" / "grid_scores.duckdb"

CASE_KEY = "w72_23_05_amt150_mv30__hold3m5_c097_e096_ddtight_pos65__strict_amt150000_mv300000_pc150_pos60"
BASE_SIGNAL_FILE = DATA / "reports" / "strategy_agent_adaptive70w_fouryear_pos60_sell_rule_focus_20260629" / "signals" / f"{CASE_KEY}__daydrop96.csv"
SCORE_TABLE = f"score_{CASE_KEY}"

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
    "GM_ADAPTIVE_SELL_SLIPPAGE_MULT": "0.0",
}

VARIANTS = [
    {
        "name": "dd_base",
        "soft_trigger": "0.06",
        "hard_trigger": "0.10",
        "recover_trigger": "0.03",
        "soft_scale": "0.65",
        "hard_scale": "0.45",
        "targets": [0.60, 0.65, 0.70, 0.75],
    },
    {
        "name": "dd_mid_a",
        "soft_trigger": "0.07",
        "hard_trigger": "0.12",
        "recover_trigger": "0.035",
        "soft_scale": "0.75",
        "hard_scale": "0.55",
        "targets": [0.60, 0.65, 0.70],
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


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _make_signal_file(target_pct: float) -> Path:
    target_tag = str(int(round(target_pct * 100)))
    signal_file = OUT_DIR / "signals" / f"{CASE_KEY}__daydrop96__pos{target_tag}.csv"
    if signal_file.exists():
        return signal_file
    rows = _load_rows(BASE_SIGNAL_FILE)
    for row in rows:
        row["target_pct"] = f"{target_pct:.5f}"
    _write_rows(signal_file, rows)
    return signal_file


def _env_for_case(case: dict[str, Any]) -> dict[str, str]:
    env = dict(BASE_ENV)
    env.update(
        {
            "GM_EQUITY_DD_RISK_MODE": "1",
            "GM_EQUITY_DD_RESIZE_EXISTING": "0",
            "GM_EQUITY_DD_SOFT_TRIGGER": case["soft_trigger"],
            "GM_EQUITY_DD_HARD_TRIGGER": case["hard_trigger"],
            "GM_EQUITY_DD_RECOVER_TRIGGER": case["recover_trigger"],
            "GM_EQUITY_DD_SOFT_SCALE": case["soft_scale"],
            "GM_EQUITY_DD_HARD_SCALE": case["hard_scale"],
            "GM_EQUITY_DD_STRICT_WHEN_DRAWDOWN": "0",
        }
    )
    return env


def _run_case(case: dict[str, Any], target_pct: float, signal_file: Path, tag: str, start: str, end: str) -> dict[str, Any]:
    target_tag = str(int(round(target_pct * 100)))
    log_file = OUT_DIR / "logs" / f"{CASE_KEY}__daydrop96__{case['name']}__pos{target_tag}__{tag}.log"
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
            f"{target_pct:.2f}",
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
        "variant": case["name"],
        "target_pct": target_pct,
        "slice": tag,
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
    if not BASE_SIGNAL_FILE.exists():
        raise FileNotFoundError(f"signal file not found: {BASE_SIGNAL_FILE}")
    detail: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for case in VARIANTS:
        for target_pct in case["targets"]:
            signal_file = _make_signal_file(target_pct)
            rows = [_run_case(case, target_pct, signal_file, tag, start, end) for tag, start, end in SLICES]
            detail.extend(rows)
            by_slice = {row["slice"]: row for row in rows}
            out: dict[str, Any] = {"variant": case["name"], "target_pct": target_pct, "signal_file": str(signal_file)}
            for slice_name in ["full", "recent120", "recent60", "ytd2026"]:
                row = by_slice[slice_name]
                for key in ["annual", "pnl_ratio", "sharpe", "max_drawdown", "win_ratio", "open_count", "close_count", "log_file"]:
                    out[f"{slice_name}_{key}"] = row.get(key)
            summary.append(out)
            print(json.dumps(out, ensure_ascii=False), flush=True)
            _write_rows(OUT_DIR / "target_grid_detail.csv", detail)
            _write_rows(OUT_DIR / "target_grid_summary.csv", summary)


if __name__ == "__main__":
    main()

