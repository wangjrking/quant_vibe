from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path

from gm_signal_module import to_gm_symbol


def _to_int(value, default=999999):
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


def _to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _trade_dates(conn, start: str, end: str) -> list[str]:
    rows = conn.execute(
        'SELECT DISTINCT trade_date FROM "daily_data" WHERE trade_date >= ? AND trade_date <= ? ORDER BY trade_date',
        (start, end),
    ).fetchall()
    return [row[0] for row in rows]


def _limit_rows(conn, start: str, end: str) -> list[dict]:
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT *
        FROM "limit_list_data"
        WHERE trade_date >= ?
          AND trade_date <= ?
          AND "limit" = 'U'
          AND name NOT LIKE 'ST%'
          AND name NOT LIKE '*ST%'
        ORDER BY trade_date, first_time, open_times, fd_amount DESC
        """,
        (start, end),
    ).fetchall()
    return [dict(row) for row in rows]


def build_limit_signals(db_path: str | Path, start: str, end: str, top_k: int) -> list[dict]:
    conn = sqlite3.connect(db_path)
    dates = _trade_dates(conn, start, end)
    next_date = {date: dates[idx + 1] for idx, date in enumerate(dates[:-1])}
    rows = _limit_rows(conn, start, end)
    conn.close()

    grouped = {}
    for row in rows:
        grouped.setdefault(str(row["trade_date"]), []).append(row)

    signals = []
    for signal_date in sorted(grouped):
        buy_date = next_date.get(signal_date)
        if not buy_date:
            continue
        day_rows = sorted(
            grouped[signal_date],
            key=lambda row: (
                _to_int(row.get("open_times")),
                _to_int(row.get("first_time")),
                -_to_float(row.get("fd_amount")),
                -_to_float(row.get("amount")),
            ),
        )[:top_k]
        for rank, row in enumerate(day_rows, 1):
            signals.append(
                {
                    "signal_date": signal_date,
                    "buy_date": buy_date,
                    "symbol": to_gm_symbol(row.get("ts_code")),
                    "stock_code": row.get("ts_code"),
                    "name": row.get("name"),
                    "rank": rank,
                    "pct_chg": row.get("pct_chg"),
                    "first_time": row.get("first_time"),
                    "open_times": row.get("open_times"),
                    "fd_amount": row.get("fd_amount"),
                    "amount": row.get("amount"),
                    "limit_times": row.get("limit_times"),
                }
            )
    return signals


def write_csv(signals: list[dict], output_path: str | Path) -> None:
    if not signals:
        raise ValueError("No limit signals to write.")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(signals[0].keys()))
        writer.writeheader()
        writer.writerows(signals)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Export gm.api signals from limit-up stock list.")
    parser.add_argument("--db", default="data_file/odb.db")
    parser.add_argument("--start", default="20240604")
    parser.add_argument("--end", default="20260604")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", default="data_file/gm_signals_limit_board.csv")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    signals = build_limit_signals(args.db, args.start, args.end, args.top_k)
    write_csv(signals, args.output)
    dates = sorted({row["buy_date"] for row in signals})
    print(f"signals: {len(signals)}")
    print(f"buy_days: {len(dates)}")
    if dates:
        print(f"buy_date_range: {dates[0]}-{dates[-1]}")
    print(f"output: {Path(args.output)}")


if __name__ == "__main__":
    main()
