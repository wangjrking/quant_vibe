from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import importlib.util
import json
import os
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA = ROOT / "quant" / "data_file"
TUNE_MODULE_PATH = MAIN / "tune_current_prod_four_year_formal_20260628.py"
REPORT_DIR = DATA / "reports" / "strategy_agent_adaptive70w_fouryear_fullwindow_20260629"
STRATEGY_DIR = DATA / "reports" / "strategy_agent_dynamic_slippage_70w_20260628" / "code_snapshot_dynamic_slippage"
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")
MARKET_DB = DATA / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
SCORE_DB = REPORT_DIR / "scores" / "grid_scores.duckdb"

ENTRIES = [
    {"name": "w72_23_05_amt90_mv20", "w10d": 0.72, "w5d": 0.23, "w3d": 0.05, "amount_min": 90000.0, "total_mv_min": 200000.0},
    {"name": "w75_20_05_amt90_mv20", "w10d": 0.75, "w5d": 0.20, "w3d": 0.05, "amount_min": 90000.0, "total_mv_min": 200000.0},
    {"name": "w78_17_05_amt90_mv20", "w10d": 0.78, "w5d": 0.17, "w3d": 0.05, "amount_min": 90000.0, "total_mv_min": 200000.0},
    {"name": "w80_15_05_amt90_mv20", "w10d": 0.80, "w5d": 0.15, "w3d": 0.05, "amount_min": 90000.0, "total_mv_min": 200000.0},
]

EXECS = [
    {
        "name": "hold4m6_c096_e094_ddtight_pos70",
        "holding_days": 4,
        "max_holding_days": 6,
        "continue_ratio": 0.96,
        "exit_ratio": 0.94,
        "target_pct": 0.70,
        "stop_loss": 0.05,
        "take_profit": 0.09,
        "dd_soft": 0.06,
        "dd_hard": 0.10,
        "dd_soft_scale": 0.65,
        "dd_hard_scale": 0.45,
    },
    {
        "name": "hold4m6_c096_e094_ddtight_pos75",
        "holding_days": 4,
        "max_holding_days": 6,
        "continue_ratio": 0.96,
        "exit_ratio": 0.94,
        "target_pct": 0.75,
        "stop_loss": 0.05,
        "take_profit": 0.09,
        "dd_soft": 0.06,
        "dd_hard": 0.10,
        "dd_soft_scale": 0.65,
        "dd_hard_scale": 0.45,
    },
    {
        "name": "hold4m6_c096_e094_ddtight_pos80",
        "holding_days": 4,
        "max_holding_days": 6,
        "continue_ratio": 0.96,
        "exit_ratio": 0.94,
        "target_pct": 0.80,
        "stop_loss": 0.05,
        "take_profit": 0.09,
        "dd_soft": 0.06,
        "dd_hard": 0.10,
        "dd_soft_scale": 0.65,
        "dd_hard_scale": 0.45,
    },
]

SLICES = [
    ("full", "2022-06-07 09:00:00", "2026-06-26 15:30:00"),
    ("recent120", "2025-12-24 09:00:00", "2026-06-26 15:30:00"),
    ("recent60", "2026-03-25 09:00:00", "2026-06-26 15:30:00"),
    ("ytd2026", "2026-01-05 09:00:00", "2026-06-26 15:30:00"),
]

BASE_ENV = {
    "GM_OPEN_DAILY_SCORE_EXIT": "1",
    "GM_MAX_DAILY_SELLS": "1",
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
    "GM_EQUITY_DD_RECOVER_TRIGGER": "0.03",
    "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "1",
    "GM_ADAPTIVE_LIQUIDITY_SLIPPAGE": "1",
    "GM_ADAPTIVE_SLIPPAGE_ORDER_VALUE": "700000",
    "GM_ADAPTIVE_SLIPPAGE_BASE": "0.0015",
    "GM_ADAPTIVE_SLIPPAGE_IMPACT_COEF": "0.0200",
    "GM_ADAPTIVE_SLIPPAGE_CAP": "0.0065",
    "GM_ADAPTIVE_SLIPPAGE_AMOUNT_UNIT": "1000",
    "GM_ADAPTIVE_BUY_SLIPPAGE_MULT": "1.0",
    "GM_ADAPTIVE_SELL_SLIPPAGE_MULT": "0.0",
}


def _load_tune_module():
    spec = importlib.util.spec_from_file_location("tune_mod", TUNE_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {TUNE_MODULE_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.REPORT_DIR = REPORT_DIR
    mod.SCORE_DB = SCORE_DB
    return mod


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


def _run(case: dict[str, Any], exe: dict[str, Any], tag: str, start: str, end: str) -> dict[str, Any]:
    case_key = case["case_key"]
    log_file = REPORT_DIR / "logs" / f"{case_key}__adaptive70w_buyonly__{tag}.log"
    indicator = _extract_indicator(log_file)
    returncode = 0
    if indicator is None:
        env = os.environ.copy()
        env.update(BASE_ENV)
        env.update(
            {
                "GM_EQUITY_DD_SOFT_TRIGGER": str(exe["dd_soft"]),
                "GM_EQUITY_DD_HARD_TRIGGER": str(exe["dd_hard"]),
                "GM_EQUITY_DD_SOFT_SCALE": str(exe["dd_soft_scale"]),
                "GM_EQUITY_DD_HARD_SCALE": str(exe["dd_hard_scale"]),
                "GM_SCORE_EXIT_ENTRY_RATIO": str(exe["exit_ratio"]),
                "GM_SCORE_CONTINUE_ENTRY_RATIO": str(exe["continue_ratio"]),
            }
        )
        cmd = [
            str(JUEJIN_PYTHON),
            str(MAIN / "run_juejin_signal_backtest.py"),
            "--strategy-dir",
            str(STRATEGY_DIR),
            "--signal-file",
            str(case["signal_file"]),
            "--log-file",
            str(log_file),
            "--max-positions",
            "1",
            "--holding-days",
            str(exe["holding_days"]),
            "--max-holding-days",
            str(exe["max_holding_days"]),
            "--target-position-pct",
            str(exe["target_pct"]),
            "--score-db",
            str(SCORE_DB),
            "--score-table",
            str(case["score_table"]),
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
            str(exe["stop_loss"]),
            "--take-profit-pct",
            str(exe["take_profit"]),
        ]
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, cwd=str(MAIN), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        returncode = proc.returncode
        indicator = _extract_indicator(log_file)
    return {
        "case_key": case_key,
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


def _summary(entry: dict[str, Any], exe: dict[str, Any], case: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_slice = {row["slice"]: row for row in rows}
    out = {
        "case_key": case["case_key"],
        "entry": entry["name"],
        "exe": exe["name"],
        "w10d": entry["w10d"],
        "w5d": entry["w5d"],
        "w3d": entry["w3d"],
        "target_pct": exe["target_pct"],
        "signal_rows": case["signal_rows"],
        "signal_file": str(case["signal_file"]),
        "score_table": case["score_table"],
    }
    for slice_name in ["full", "recent120", "recent60", "ytd2026"]:
        row = by_slice[slice_name]
        for key in ["annual", "pnl_ratio", "sharpe", "max_drawdown", "win_ratio", "open_count", "close_count"]:
            out[f"{slice_name}_{key}"] = row.get(key)
    return out


def main() -> None:
    mod = _load_tune_module()
    manifest, _, sources = mod._load_strategy(mod.STRATEGY_DIR)
    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for entry in ENTRIES:
        for exe in EXECS:
            case = mod._build_case_assets(manifest, sources, entry, exe)
            rows = [_run(case, exe, tag, start, end) for tag, start, end in SLICES]
            detail_rows.extend(rows)
            summary_rows.append(_summary(entry, exe, case, rows))
            _write_rows(REPORT_DIR / "adaptive70w_fouryear_detail.csv", detail_rows)
            _write_rows(REPORT_DIR / "adaptive70w_fouryear_summary.csv", summary_rows)
            print(json.dumps(summary_rows[-1], ensure_ascii=False), flush=True)
    (REPORT_DIR / "adaptive70w_fouryear_summary.json").write_text(
        json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

