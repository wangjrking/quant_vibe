from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import os
import re
import sqlite3
import subprocess
from pathlib import Path

import research_10d_with_new5d_confirm_probe_20260621 as base
from gm_signal_module import build_gm_signal_rows, load_market_rows_by_trade_date, write_gm_signals_csv
from research_10d_new5d_3d_confirm_grid_20260621 import TABLE_3D
from research_list_age_current_best_probe_20260620 import JUEJIN_PYTHON, MAIN, MARKET_DB, PRED_DB, STRATEGY_DIR, TABLE_10D
from selection_module import SelectionConfig


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "new5d_diagnostic_filter_grid_20260621"
)


GRID = [
    {"name": "mv120_200_h7_mp6", "min_total_mv": 120000.0, "max_total_mv": 200000.0, "holding_days": 7, "max_positions": 6},
    {"name": "mv120_200_h7_mp7", "min_total_mv": 120000.0, "max_total_mv": 200000.0, "holding_days": 7, "max_positions": 7},
    {"name": "r3_0p3_0p7_h7_mp6", "min_rank3d": 0.30, "max_rank3d": 0.70, "holding_days": 7, "max_positions": 6},
    {"name": "r3_0p3_0p7_h7_mp7", "min_rank3d": 0.30, "max_rank3d": 0.70, "holding_days": 7, "max_positions": 7},
    {
        "name": "mv120_200_r3_0p3_0p7_h7_mp6",
        "min_total_mv": 120000.0,
        "max_total_mv": 200000.0,
        "min_rank3d": 0.30,
        "max_rank3d": 0.70,
        "holding_days": 7,
        "max_positions": 6,
    },
    {
        "name": "mv120_200_r3_0p3_0p7_h7_mp7",
        "min_total_mv": 120000.0,
        "max_total_mv": 200000.0,
        "min_rank3d": 0.30,
        "max_rank3d": 0.70,
        "holding_days": 7,
        "max_positions": 7,
    },
    {"name": "r5_0p5_0p95_h7_mp6", "min_rank5d": 0.50, "max_rank5d": 0.95, "holding_days": 7, "max_positions": 6},
    {"name": "r5_0p5_0p95_h7_mp7", "min_rank5d": 0.50, "max_rank5d": 0.95, "holding_days": 7, "max_positions": 7},
    {
        "name": "mv120_200_r5_0p5_0p95_h7_mp6",
        "min_total_mv": 120000.0,
        "max_total_mv": 200000.0,
        "min_rank5d": 0.50,
        "max_rank5d": 0.95,
        "holding_days": 7,
        "max_positions": 6,
    },
    {
        "name": "mv120_200_r5_0p5_0p95_h7_mp7",
        "min_total_mv": 120000.0,
        "max_total_mv": 200000.0,
        "min_rank5d": 0.50,
        "max_rank5d": 0.95,
        "holding_days": 7,
        "max_positions": 7,
    },
    {
        "name": "mv120_200_r3_0p3_0p7_r5_0p5_0p95_h7_mp6",
        "min_total_mv": 120000.0,
        "max_total_mv": 200000.0,
        "min_rank3d": 0.30,
        "max_rank3d": 0.70,
        "min_rank5d": 0.50,
        "max_rank5d": 0.95,
        "holding_days": 7,
        "max_positions": 6,
    },
    {
        "name": "mv120_200_r3_0p3_0p7_r5_0p5_0p95_h7_mp7",
        "min_total_mv": 120000.0,
        "max_total_mv": 200000.0,
        "min_rank3d": 0.30,
        "max_rank3d": 0.70,
        "min_rank5d": 0.50,
        "max_rank5d": 0.95,
        "holding_days": 7,
        "max_positions": 7,
    },
]


def _connect_readonly(path: Path) -> sqlite3.Connection:
    uri = path.resolve().as_uri() + "?mode=ro"
    return sqlite3.connect(uri, uri=True, timeout=30)


def _to_float(value):
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rank_by_day(table: str) -> dict[tuple[str, str], float]:
    conn = _connect_readonly(PRED_DB)
    conn.row_factory = sqlite3.Row
    try:
        rows = list(
            conn.execute(
                f"""
                SELECT trade_date, stock_code, pred_prob
                FROM "{table}"
                WHERE trade_date >= ? AND trade_date <= ? AND pred_prob IS NOT NULL
                ORDER BY trade_date, pred_prob DESC, stock_code
                """,
                (base.START_DATE, base.END_DATE),
            )
        )
    finally:
        conn.close()
    grouped: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        grouped.setdefault(str(row["trade_date"]), []).append(row)
    ranks = {}
    for trade_date, day_rows in grouped.items():
        denom = max(len(day_rows) - 1, 1)
        for idx, row in enumerate(day_rows):
            ranks[(trade_date, str(row["stock_code"]))] = 1.0 - (idx / denom)
    return ranks


def _load_rows() -> list[dict]:
    rows = base._load_10d_rows_with_new5d_rank()
    rank3d = _rank_by_day(TABLE_3D)
    for row in rows:
        row["rank3d"] = rank3d.get((str(row["trade_date"]), str(row["stock_code"])))
    return rows


def _passes(row: dict, params: dict) -> bool:
    close_rate = _to_float(row.get("close_rate"))
    if close_rate is None or close_rate < 0.955 or close_rate > 1.085:
        return False
    total_mv = _to_float(row.get("total_mv"))
    if total_mv is None:
        return False
    if params.get("min_total_mv") is not None and total_mv < float(params["min_total_mv"]):
        return False
    if params.get("max_total_mv") is not None and total_mv > float(params["max_total_mv"]):
        return False
    rank3d = _to_float(row.get("rank3d"))
    if params.get("min_rank3d") is not None and (rank3d is None or rank3d < float(params["min_rank3d"])):
        return False
    if params.get("max_rank3d") is not None and (rank3d is None or rank3d > float(params["max_rank3d"])):
        return False
    rank5d = _to_float(row.get("rank5d_new"))
    if params.get("min_rank5d") is not None and (rank5d is None or rank5d < float(params["min_rank5d"])):
        return False
    if params.get("max_rank5d") is not None and (rank5d is None or rank5d > float(params["max_rank5d"])):
        return False
    return True


def _write_signal(rows: list[dict], signal_file: Path, params: dict) -> None:
    market_rows = load_market_rows_by_trade_date(MARKET_DB, base.START_DATE, base.END_DATE)
    config = SelectionConfig(
        top_k=5,
        pred_col="pred_prob",
        min_pred_prob=None,
        max_atr_ratio=None,
        min_amount=None,
        max_total_mv=params.get("max_total_mv"),
        max_per_industry=999,
        exclude_bj=True,
        exclude_st=True,
        exclude_delisting=True,
        exclude_current_limit=True,
    )
    signals = build_gm_signal_rows(
        [row for row in rows if _passes(row, params)],
        config=config,
        market_rows_by_trade_date=market_rows,
        holding_days=int(params["holding_days"]),
        max_positions=int(params["max_positions"]),
        weight_mode="equal",
        target_total_pct=0.98,
    )
    by_key = {(str(row["trade_date"]), str(row["stock_code"])): row for row in rows}
    for signal in signals:
        source = by_key.get((str(signal["signal_date"]), str(signal["stock_code"])), {})
        signal["rank3d"] = source.get("rank3d")
        signal["rank5d_new"] = source.get("rank5d_new")
        signal["close_rate"] = source.get("close_rate")
        signal["total_mv"] = source.get("total_mv")
        signal["list_age_days"] = source.get("list_age_days")
    write_gm_signals_csv(signals, signal_file)


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


def _exposure_stats(log_file: Path) -> dict:
    values = []
    active = []
    pattern = re.compile(r"EXPOSURE .* invested_pct=([0-9.]+).*active_positions=(\d+)")
    if not log_file.exists():
        return {"avg_invested_pct": None, "ge80_ratio": None, "max_active_positions": None, "exposure_points": 0}
    for match in pattern.finditer(log_file.read_text(encoding="utf-8", errors="ignore")):
        values.append(float(match.group(1)))
        active.append(int(match.group(2)))
    if not values:
        return {"avg_invested_pct": None, "ge80_ratio": None, "max_active_positions": None, "exposure_points": 0}
    return {
        "avg_invested_pct": sum(values) / len(values),
        "ge80_ratio": sum(1 for value in values if value >= 0.80) / len(values),
        "max_active_positions": max(active) if active else None,
        "exposure_points": len(values),
    }


def _signal_stats(signal_file: Path) -> dict:
    with signal_file.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    return {
        "signal_count": len(rows),
        "buy_days": len({row.get("buy_date") for row in rows if row.get("buy_date")}),
    }


def _run(signal_file: Path, log_file: Path, params: dict) -> int:
    if log_file.exists() and _extract_indicator(log_file):
        return 0
    env = os.environ.copy()
    env.update(
        {
            "GM_OPEN_DAILY_SCORE_EXIT": "0",
            "GM_MAX_DAILY_SELLS": "1",
            "GM_LOG_EXPOSURE": "1",
            "GM_SKIP_OPEN_LIMIT_UP_BUY": "1",
            "GM_FORCE_MARKET_ORDER": "0",
            "GM_FORCE_BUY_MARKET_ORDER": "0",
            "GM_SYNC_POSITIONS": "0",
            "GM_CASH_BUFFER": "0.995",
            "GM_VERBOSE_TRADES": "0",
            "GM_SCORE_CONTINUE_ENTRY_RATIO": "999",
        }
    )
    max_positions = int(params["max_positions"])
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
        str(max_positions),
        "--holding-days",
        "5",
        "--max-holding-days",
        "8",
        "--target-position-pct",
        str(0.98 / max_positions),
        "--score-db",
        str(PRED_DB),
        "--score-table",
        TABLE_10D,
        "--market-db",
        str(MARKET_DB),
        "--score-continue-entry-ratio",
        "999",
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
    return proc.returncode


def main() -> None:
    REPORT_DIR.joinpath("signals").mkdir(parents=True, exist_ok=True)
    REPORT_DIR.joinpath("logs").mkdir(parents=True, exist_ok=True)
    rows = _load_rows()
    results = []
    for params in GRID:
        signal_file = REPORT_DIR / "signals" / f"{params['name']}.csv"
        log_file = REPORT_DIR / "logs" / f"{params['name']}.log"
        if not signal_file.exists():
            _write_signal(rows, signal_file, params)
        returncode = _run(signal_file, log_file, params)
        indicator = _extract_indicator(log_file) or {}
        result = {
            **params,
            "returncode": returncode,
            "signal_file": str(signal_file),
            "log_file": str(log_file),
            "annual": indicator.get("pnl_ratio_annual"),
            "sharpe": indicator.get("sharp_ratio"),
            "max_drawdown": indicator.get("max_drawdown"),
            "win_ratio": indicator.get("win_ratio"),
            **_signal_stats(signal_file),
            **_exposure_stats(log_file),
        }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False, default=str), flush=True)
    fieldnames = sorted({key for row in results for key in row.keys()})
    outputs = {
        "summary.csv": results,
        "summary_by_sharpe.csv": sorted(results, key=lambda row: float(row.get("sharpe") or -999), reverse=True),
        "summary_by_annual.csv": sorted(results, key=lambda row: float(row.get("annual") or -999), reverse=True),
        "summary_avg80_by_sharpe.csv": [
            row
            for row in sorted(results, key=lambda item: float(item.get("sharpe") or -999), reverse=True)
            if float(row.get("avg_invested_pct") or -999) >= 0.80
        ],
        "summary_qualified_target_by_sharpe.csv": [
            row
            for row in sorted(results, key=lambda item: float(item.get("sharpe") or -999), reverse=True)
            if float(row.get("annual") or -999) >= 2.0
            and float(row.get("sharpe") or -999) >= 3.0
            and float(row.get("avg_invested_pct") or -999) >= 0.80
        ],
    }
    for filename, data in outputs.items():
        with (REPORT_DIR / filename).open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(data)


if __name__ == "__main__":
    main()
