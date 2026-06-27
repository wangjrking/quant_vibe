from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

from gm_signal_module import build_gm_signal_rows, load_market_rows_by_trade_date, write_gm_signals_csv
from research_confirm_mv_boundary_refine_20260620 import _result_row, _run_backtest
from research_list_age_current_best_probe_20260620 import (
    END_DATE,
    MARKET_DB,
    PRED_DB,
    START_DATE,
    _connect_readonly,
    _load_market_meta,
    _safe,
    _to_float,
)
from selection_module import SelectionConfig


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_model_application_20260620"
    / "latest_10d_tiered_confirm_fill_probe_20260620"
)
TABLE_10D = "stock_predict_data_model_agent_toprank_latestfactor_20260618_executable_10d_open_return_score_20240604_20260618"
TABLE_5D = "stock_predict_data_model_agent_toprank_latestfactor_20260618_executable_5d_open_return_score_20240604_20260618"
TABLE_3D = "stock_predict_data_model_agent_toprank_latestfactor_20260618_executable_3d_open_return_score_20240604_20260618"


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
    ranks = {}
    for trade_date, day_rows in grouped.items():
        denom = max(len(day_rows) - 1, 1)
        for idx, row in enumerate(day_rows):
            ranks[(trade_date, str(row["stock_code"]))] = 1.0 - (idx / denom)
    return ranks


def _date_days(start: str | None, end: str | None) -> int | None:
    if not start or not end:
        return None
    from datetime import datetime

    try:
        return (datetime.strptime(str(end), "%Y%m%d") - datetime.strptime(str(start), "%Y%m%d")).days
    except ValueError:
        return None


def _load_rows() -> list[dict]:
    rank5d = _rank_by_day(TABLE_5D)
    rank3d = _rank_by_day(TABLE_3D)
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
    rows = []
    for row in base_rows:
        enriched = dict(row)
        key = (str(enriched["trade_date"]), str(enriched["stock_code"]))
        meta = market_meta.get(key, {})
        for field in ("name", "list_date", "amount", "turnover_rate", "total_mv", "limit_times"):
            if meta.get(field) not in (None, ""):
                enriched[field] = meta[field]
        enriched["rank5d"] = rank5d.get(key)
        enriched["rank3d"] = rank3d.get(key)
        enriched["list_age_days"] = _date_days(enriched.get("list_date"), enriched.get("trade_date"))
        rows.append(enriched)
    return rows


def _filtered_rows(
    rows: list[dict],
    *,
    primary_rank3d_min: float,
    fallback_rank5d_min: float,
    min_close_rate: float,
    max_close_rate: float,
    tier_bonus: float,
) -> list[dict]:
    selected = []
    for row in rows:
        rank5d = _to_float(row.get("rank5d"))
        if rank5d is None or rank5d < fallback_rank5d_min:
            continue
        close_rate = _to_float(row.get("close_rate"))
        if close_rate is None or close_rate < min_close_rate or close_rate > max_close_rate:
            continue
        rank3d = _to_float(row.get("rank3d"))
        is_primary = rank3d is not None and rank3d >= primary_rank3d_min
        enriched = dict(row)
        enriched["is_primary3d"] = 1 if is_primary else 0
        enriched["tier_score"] = float(row.get("pred_prob") or 0.0) + (tier_bonus if is_primary else 0.0)
        selected.append(enriched)
    return selected


def _write_signal(rows: list[dict], signal_file: Path) -> None:
    market_rows = load_market_rows_by_trade_date(MARKET_DB, START_DATE, END_DATE)
    config = SelectionConfig(
        top_k=5,
        pred_col="tier_score",
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
        signal["raw_pred_prob"] = source.get("pred_prob")
        signal["rank3d"] = source.get("rank3d")
        signal["rank5d"] = source.get("rank5d")
        signal["is_primary3d"] = source.get("is_primary3d")
        signal["close_rate"] = source.get("close_rate")
        signal["total_mv"] = source.get("total_mv")
    write_gm_signals_csv(signals, signal_file)


def _primary_share(signal_file: Path) -> dict:
    if not signal_file.exists():
        return {"primary_signal_count": 0, "primary_signal_share": None}
    with signal_file.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return {"primary_signal_count": 0, "primary_signal_share": None}
    primary = sum(1 for row in rows if str(row.get("is_primary3d")) in {"1", "1.0"})
    return {"primary_signal_count": primary, "primary_signal_share": primary / len(rows)}


def main() -> None:
    REPORT_DIR.joinpath("signals").mkdir(parents=True, exist_ok=True)
    REPORT_DIR.joinpath("logs").mkdir(parents=True, exist_ok=True)
    rows = _load_rows()
    grid = [
        {
            "primary_rank3d_min": primary,
            "fallback_rank5d_min": fallback,
            "min_close_rate": 0.965,
            "max_close_rate": max_close,
            "tier_bonus": bonus,
        }
        for primary in [0.70, 0.75, 0.80]
        for fallback in [0.45, 0.50]
        for max_close in [1.085, 1.095]
        for bonus in [1.0, 10.0]
    ]
    results = []
    for params in grid:
        label = (
            f"p3d{_safe(params['primary_rank3d_min'])}"
            f"_f5d{_safe(params['fallback_rank5d_min'])}"
            f"_cr{_safe(params['min_close_rate'])}_{_safe(params['max_close_rate'])}"
            f"_b{_safe(params['tier_bonus'])}"
        )
        signal_file = REPORT_DIR / "signals" / f"{label}.csv"
        log_file = REPORT_DIR / "logs" / f"{label}.log"
        if not log_file.exists():
            filtered = _filtered_rows(rows, **params)
            _write_signal(filtered, signal_file)
            returncode = _run_backtest(signal_file, log_file)
        else:
            returncode = 0
        row = {
            **params,
            "top_k": 5,
            "holding_days": 5,
            "max_total_mv": 200000.0,
            "target_total_pct": 0.98,
            "open_daily_score_exit": 0,
            "max_daily_sells": 1,
            **_primary_share(signal_file),
            **_result_row(params, signal_file, log_file, returncode),
        }
        results.append(row)
        print(json.dumps(row, ensure_ascii=False, default=str))
    fieldnames = []
    for row in results:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    outputs = {
        "summary.csv": results,
        "summary_by_sharpe.csv": sorted(results, key=lambda row: float(row.get("sharpe") or -999), reverse=True),
        "summary_by_annual.csv": sorted(results, key=lambda row: float(row.get("annual") or -999), reverse=True),
        "summary_qualified_annual2_avg80_by_sharpe.csv": [
            row
            for row in sorted(results, key=lambda item: float(item.get("sharpe") or -999), reverse=True)
            if float(row.get("annual") or -999) >= 2.0 and float(row.get("avg_invested_pct") or -999) >= 0.80
        ],
    }
    for name, data in outputs.items():
        with (REPORT_DIR / name).open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)


if __name__ == "__main__":
    main()
