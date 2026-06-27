from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import os
import re
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path

from gm_signal_module import build_gm_signal_rows, load_market_rows_by_trade_date, write_gm_signals_csv
from research_list_age_current_best_probe_20260620 import (
    END_DATE,
    JUEJIN_PYTHON,
    MAIN,
    MARKET_DB,
    PRED_DB,
    START_DATE,
    STRATEGY_DIR,
    TABLE_10D,
    TABLE_5D,
)
from selection_module import SelectionConfig


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "primary5d_confirm10d_probe_20260621"
)

GRID = []
for confirm_rank10d_min in (0.45, 0.50, 0.55, 0.60):
    for min_close_rate in (0.955, 0.965):
        GRID.append(
            {
                "confirm_rank10d_min": confirm_rank10d_min,
                "min_close_rate": min_close_rate,
                "max_close_rate": 1.085,
                "max_total_mv": 200000.0,
                "top_k": 5,
                "holding_days": 5,
                "target_total_pct": 0.98,
            }
        )


def _connect_readonly(path: Path) -> sqlite3.Connection:
    uri = path.resolve().as_uri() + "?mode=ro"
    return sqlite3.connect(uri, uri=True, timeout=30)


def _safe(value) -> str:
    return str(value).replace(".", "p").replace("-", "neg").replace("None", "none")


def _to_float(value):
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _date_days(start: str | None, end: str | None) -> int | None:
    if not start or not end:
        return None
    try:
        return (datetime.strptime(str(end), "%Y%m%d") - datetime.strptime(str(start), "%Y%m%d")).days
    except ValueError:
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
                ORDER BY trade_date, pred_prob DESC
                """,
                (START_DATE, END_DATE),
            )
        )
    finally:
        conn.close()
    grouped: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        grouped.setdefault(str(row["trade_date"]), []).append(row)
    ranks: dict[tuple[str, str], float] = {}
    for trade_date, day_rows in grouped.items():
        denom = max(len(day_rows) - 1, 1)
        for idx, row in enumerate(day_rows):
            ranks[(trade_date, str(row["stock_code"]))] = 1.0 - (idx / denom)
    return ranks


def _load_market_meta() -> dict[tuple[str, str], dict]:
    conn = _connect_readonly(MARKET_DB)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT trade_date, stock_code, name, list_date, amount,
                   turnover_rate, total_mv, limit_times
            FROM STOCK_DAILY_DATA
            WHERE trade_date >= ? AND trade_date <= ?
            """,
            (START_DATE, END_DATE),
        ).fetchall()
    finally:
        conn.close()
    return {(str(row["trade_date"]), str(row["stock_code"])): dict(row) for row in rows}


def _load_5d_rows() -> list[dict]:
    rank10d = _rank_by_day(TABLE_10D)
    market_meta = _load_market_meta()
    conn = _connect_readonly(PRED_DB)
    conn.row_factory = sqlite3.Row
    try:
        base_rows = conn.execute(
            f"""
            SELECT *
            FROM "{TABLE_5D}"
            WHERE trade_date >= ? AND trade_date <= ? AND pred_prob IS NOT NULL
            """,
            (START_DATE, END_DATE),
        ).fetchall()
    finally:
        conn.close()
    rows = []
    for row in base_rows:
        enriched = dict(row)
        key = (str(enriched["trade_date"]), str(enriched["stock_code"]))
        meta = market_meta.get(key, {})
        for field in ("name", "list_date", "amount", "turnover_rate", "total_mv", "limit_times"):
            if meta.get(field) not in (None, ""):
                enriched[field] = meta[field]
        enriched["rank10d"] = rank10d.get(key)
        enriched["list_age_days"] = _date_days(enriched.get("list_date"), enriched.get("trade_date"))
        rows.append(enriched)
    return rows


def _filtered_rows(rows: list[dict], params: dict) -> list[dict]:
    selected = []
    for row in rows:
        rank10d = _to_float(row.get("rank10d"))
        if rank10d is None or rank10d < float(params["confirm_rank10d_min"]):
            continue
        close_rate = _to_float(row.get("close_rate"))
        if close_rate is None or close_rate < float(params["min_close_rate"]) or close_rate > float(params["max_close_rate"]):
            continue
        selected.append(row)
    return selected


def _write_signal(rows: list[dict], signal_file: Path, *, params: dict) -> None:
    market_rows = load_market_rows_by_trade_date(MARKET_DB, START_DATE, END_DATE)
    config = SelectionConfig(
        top_k=int(params["top_k"]),
        pred_col="pred_prob",
        min_pred_prob=None,
        max_atr_ratio=None,
        max_total_mv=float(params["max_total_mv"]),
        max_per_industry=999,
        exclude_bj=True,
        exclude_st=True,
        exclude_delisting=True,
        exclude_current_limit=True,
    )
    signals = build_gm_signal_rows(
        rows,
        config=config,
        market_rows_by_trade_date=market_rows,
        holding_days=int(params["holding_days"]),
        max_positions=5,
        weight_mode="equal",
        target_total_pct=float(params["target_total_pct"]),
    )
    by_key = {(str(row["trade_date"]), str(row["stock_code"])): row for row in rows}
    for signal in signals:
        source = by_key.get((str(signal["signal_date"]), str(signal["stock_code"])), {})
        signal["rank10d"] = source.get("rank10d")
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


def _run_backtest(signal_file: Path, log_file: Path) -> int:
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
        }
    )
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
        "5",
        "--holding-days",
        "5",
        "--max-holding-days",
        "5",
        "--target-position-pct",
        "0.196",
        "--score-db",
        str(PRED_DB),
        "--score-table",
        TABLE_5D,
        "--market-db",
        str(MARKET_DB),
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
    rows = _load_5d_rows()
    results = []
    for params in GRID:
        label = (
            f"p5d_c10d{_safe(params['confirm_rank10d_min'])}"
            f"_cr{_safe(params['min_close_rate'])}_{_safe(params['max_close_rate'])}"
        )
        signal_file = REPORT_DIR / "signals" / f"{label}.csv"
        log_file = REPORT_DIR / "logs" / f"{label}.log"
        if not signal_file.exists():
            _write_signal(_filtered_rows(rows, params), signal_file, params=params)
        returncode = _run_backtest(signal_file, log_file)
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
    with (REPORT_DIR / "summary.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    with (REPORT_DIR / "summary_by_sharpe.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(sorted(results, key=lambda row: float(row.get("sharpe") or -999), reverse=True))


if __name__ == "__main__":
    main()
