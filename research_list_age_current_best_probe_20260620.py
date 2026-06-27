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
from selection_module import SelectionConfig


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
PRED_DB = ROOT / "quant" / "data_file" / "model_predictions" / "MODEL_PREDICTIONS.db"
MARKET_DB = ROOT / "quant" / "data_file" / "STOCK_DAILY_DATA.db"
REPORT_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_model_application_20260620"
    / "latest_10d_current_best_list_age_probe_20260620"
)
STRATEGY_DIR = Path(r"D:\work\dfcf\juejin\strategy\ce327750-634e-11f1-b8d7-10ffe0295517")
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")

TABLE_10D = "stock_predict_data_model_agent_toprank_latestfactor_20260618_executable_10d_open_return_score_20240604_20260618"
TABLE_5D = "stock_predict_data_model_agent_toprank_latestfactor_20260618_executable_5d_open_return_score_20240604_20260618"
START_DATE = "20240604"
END_DATE = "20260618"


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


def _rank5d_by_day() -> dict[tuple[str, str], float]:
    conn = _connect_readonly(PRED_DB)
    conn.row_factory = sqlite3.Row
    try:
        rows = list(
            conn.execute(
                f"""
                SELECT trade_date, stock_code, pred_prob
                FROM "{TABLE_5D}"
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


def _load_rows() -> list[dict]:
    rank5d = _rank5d_by_day()
    market_meta = _load_market_meta()

    conn = _connect_readonly(PRED_DB)
    conn.row_factory = sqlite3.Row
    try:
        base_rows = conn.execute(
            f"""
            SELECT *
            FROM "{TABLE_10D}"
            WHERE trade_date >= ? AND trade_date <= ? AND pred_prob IS NOT NULL
            """,
            (START_DATE, END_DATE),
        ).fetchall()
    finally:
        conn.close()

    rows: list[dict] = []
    for row in base_rows:
        enriched = dict(row)
        key = (str(enriched["trade_date"]), str(enriched["stock_code"]))
        meta = market_meta.get(key, {})
        for field in ("name", "list_date", "amount", "turnover_rate", "total_mv", "limit_times"):
            if meta.get(field) not in (None, ""):
                enriched[field] = meta[field]
        enriched["rank5d"] = rank5d.get(key)
        enriched["list_age_days"] = _date_days(enriched.get("list_date"), enriched.get("trade_date"))
        rows.append(enriched)
    return rows


def _filtered_rows(rows: list[dict], min_list_days: int | None) -> list[dict]:
    selected = []
    for row in rows:
        rank5d = _to_float(row.get("rank5d"))
        if rank5d is None or rank5d < 0.50:
            continue
        close_rate = _to_float(row.get("close_rate"))
        if close_rate is None or close_rate < 0.965 or close_rate > 1.085:
            continue
        if min_list_days is not None:
            age = row.get("list_age_days")
            if age is None or int(age) < int(min_list_days):
                continue
        selected.append(row)
    return selected


def _extract_indicator(log_file: Path) -> dict | None:
    marker = "GM_BACKTEST_INDICATOR:"
    if not log_file.exists():
        return None
    for line in reversed(log_file.read_text(encoding="utf-8", errors="ignore").splitlines()):
        if marker not in line:
            continue
        payload = line.split(marker, 1)[1].strip()
        try:
            return ast.literal_eval(payload)
        except Exception:
            try:
                return eval(payload, {"__builtins__": {}}, {"datetime": datetime_module})
            except Exception:
                return None
    return None


def _exposure_stats(log_file: Path) -> dict:
    values = []
    active = []
    pattern = re.compile(r"invested_pct=([0-9.]+).*active_positions=(\d+)")
    if not log_file.exists():
        return {"avg_invested_pct": None, "ge80_ratio": None, "max_active_positions": None, "exposure_points": 0}
    for line in log_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if not match:
            continue
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
    ages = [int(float(row["list_age_days"])) for row in rows if row.get("list_age_days") not in (None, "")]
    return {
        "signal_count": len(rows),
        "buy_days": len({row.get("buy_date") for row in rows if row.get("buy_date")}),
        "min_signal_list_age_days": min(ages) if ages else None,
        "median_signal_list_age_days": sorted(ages)[len(ages) // 2] if ages else None,
    }


def _write_signal(rows: list[dict], signal_file: Path) -> None:
    market_rows = load_market_rows_by_trade_date(MARKET_DB, START_DATE, END_DATE)
    config = SelectionConfig(
        top_k=5,
        pred_col="pred_prob",
        min_pred_prob=None,
        max_atr_ratio=None,
        max_total_mv=200000.0,
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
        holding_days=5,
        max_positions=5,
        weight_mode="equal",
        target_total_pct=0.98,
    )
    by_key = {(str(row["trade_date"]), str(row["stock_code"])): row for row in rows}
    for signal in signals:
        source = by_key.get((str(signal["signal_date"]), str(signal["stock_code"])), {})
        signal["list_date"] = source.get("list_date")
        signal["list_age_days"] = source.get("list_age_days")
        signal["rank5d"] = source.get("rank5d")
        signal["close_rate"] = source.get("close_rate")
    write_gm_signals_csv(signals, signal_file)


def _run_backtest(signal_file: Path, log_file: Path) -> int:
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
        "0.98",
        "--score-db",
        str(PRED_DB),
        "--score-table",
        TABLE_10D,
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
    proc = subprocess.run(command, cwd=str(MAIN), env=env, capture_output=True, text=True)
    if proc.stdout.strip():
        try:
            payload = json.loads(proc.stdout.strip().splitlines()[-1])
            return int(payload.get("returncode", proc.returncode))
        except Exception:
            pass
    return proc.returncode


def main() -> None:
    REPORT_DIR.joinpath("signals").mkdir(parents=True, exist_ok=True)
    REPORT_DIR.joinpath("logs").mkdir(parents=True, exist_ok=True)
    rows = _load_rows()
    results = []
    for min_list_days in [None, 20, 60, 120, 180, 250, 365, 500, 750]:
        label = f"list{_safe(min_list_days)}"
        filtered = _filtered_rows(rows, min_list_days)
        signal_file = REPORT_DIR / "signals" / f"current_best_{label}.csv"
        log_file = REPORT_DIR / "logs" / f"current_best_{label}.log"
        _write_signal(filtered, signal_file)
        returncode = _run_backtest(signal_file, log_file)
        indicator = _extract_indicator(log_file)
        row = {
            "min_list_days": min_list_days,
            "returncode": returncode,
            "signal_file": str(signal_file),
            "log_file": str(log_file),
            **_signal_stats(signal_file),
            **_exposure_stats(log_file),
        }
        if indicator:
            row.update(
                {
                    "annual": indicator.get("pnl_ratio_annual"),
                    "sharpe": indicator.get("sharp_ratio"),
                    "max_drawdown": indicator.get("max_drawdown"),
                    "open_count": indicator.get("open_count"),
                    "close_count": indicator.get("close_count"),
                    "win_ratio": indicator.get("win_ratio"),
                    "calmar_ratio": indicator.get("calmar_ratio"),
                }
            )
        results.append(row)
        print(json.dumps(row, ensure_ascii=False, default=str))

    fieldnames = list(results[0].keys())
    for name, sorted_rows in {
        "summary.csv": results,
        "summary_by_sharpe.csv": sorted(results, key=lambda row: float(row.get("sharpe") or -999), reverse=True),
        "summary_by_annual.csv": sorted(results, key=lambda row: float(row.get("annual") or -999), reverse=True),
    }.items():
        with (REPORT_DIR / name).open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sorted_rows)


if __name__ == "__main__":
    main()
