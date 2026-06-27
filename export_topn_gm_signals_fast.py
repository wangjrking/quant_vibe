from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path
from typing import Any

from gm_signal_module import build_gm_signal_rows, load_market_rows_by_trade_date
from selection_module import SelectionConfig


def _quote(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _parse_none_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "null"}:
        return None
    return float(text)


def _read_top_rows(db_path: Path, table: str, start: str, end: str, top_n_per_day: int) -> list[dict[str, Any]]:
    sql = f"""
        SELECT *
        FROM (
            SELECT p.*,
                   ROW_NUMBER() OVER (PARTITION BY trade_date ORDER BY pred_prob DESC) AS rn
            FROM {_quote(table)} p
            WHERE trade_date >= ? AND trade_date <= ?
        )
        WHERE rn <= ?
        ORDER BY trade_date, pred_prob DESC
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = [dict(row) for row in conn.execute(sql, (start, end, int(top_n_per_day))).fetchall()]
    finally:
        conn.close()
    for row in rows:
        row.pop("rn", None)
    return rows


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        raise ValueError("No signals to write.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Fast SQL prefiltered export of next-day gm signals.")
    parser.add_argument("--db", required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--top-n-per-day", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--min-pred", default="none")
    parser.add_argument("--min-pred-quantile", type=float)
    parser.add_argument("--max-atr-ratio", default="none")
    parser.add_argument("--min-amount", type=float)
    parser.add_argument("--min-turnover-rate", type=float)
    parser.add_argument("--max-total-mv", type=float)
    parser.add_argument("--max-per-industry", type=int, default=999)
    parser.add_argument("--holding-days", type=int)
    parser.add_argument("--max-positions", type=int)
    parser.add_argument("--target-total-pct", type=float)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    db_path = Path(args.db)
    rows = _read_top_rows(db_path, args.table, args.start, args.end, args.top_n_per_day)
    market_rows = load_market_rows_by_trade_date(db_path, args.start, args.end)
    signals = build_gm_signal_rows(
        rows,
        SelectionConfig(
            top_k=args.top_k,
            min_pred_prob=_parse_none_float(args.min_pred),
            min_pred_quantile=args.min_pred_quantile,
            max_atr_ratio=_parse_none_float(args.max_atr_ratio),
            min_amount=args.min_amount,
            min_turnover_rate=args.min_turnover_rate,
            max_total_mv=args.max_total_mv,
            max_per_industry=args.max_per_industry,
        ),
        market_rows_by_trade_date=market_rows,
        holding_days=args.holding_days,
        max_positions=args.max_positions,
        target_total_pct=args.target_total_pct,
    )
    _write_csv(signals, Path(args.output))
    buy_dates = sorted({str(row["buy_date"]) for row in signals})
    print(f"rows_loaded={len(rows)} signals={len(signals)} buy_days={len(buy_dates)} output={args.output}")
    if buy_dates:
        print(f"buy_date_range={buy_dates[0]}-{buy_dates[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
