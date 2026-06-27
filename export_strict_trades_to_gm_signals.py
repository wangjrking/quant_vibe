from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path

from gm_signal_module import to_gm_symbol


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Convert strict walk-forward trade records to gm.api signal CSV.")
    parser.add_argument("--trades", required=True, help="CSV exported by strict/local portfolio backtest trades.")
    parser.add_argument("--output", required=True, help="Output gm signals CSV path.")
    parser.add_argument("--db", help="SQLite db used to map signal dates to next executable trade dates.")
    parser.add_argument("--target-total-pct", type=float, default=0.98, help="Total target portfolio percent to scale capital weights.")
    return parser.parse_args(argv)


def load_trade_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def load_next_trade_dates(db_path: str | None) -> dict[str, str]:
    if not db_path:
        return {}
    conn = sqlite3.connect(str(Path(db_path)))
    try:
        dates = [
            str(row[0])
            for row in conn.execute("SELECT DISTINCT trade_date FROM STOCK_DAILY_DATA ORDER BY trade_date")
            if row[0]
        ]
    finally:
        conn.close()
    return {date: dates[idx + 1] for idx, date in enumerate(dates[:-1])}


def build_signal_rows(trades: list[dict], target_total_pct: float, next_trade_dates: dict[str, str] | None = None) -> list[dict]:
    next_trade_dates = next_trade_dates or {}
    grouped: dict[str, list[dict]] = {}
    for row in trades:
        signal_date = str(row.get("entry_signal_date") or "").strip()
        buy_date = next_trade_dates.get(signal_date, signal_date)
        stock_code = str(row.get("stock_code") or "").strip()
        if not buy_date or not stock_code:
            continue
        grouped.setdefault(buy_date, []).append(row)

    signals: list[dict] = []
    for buy_date in sorted(grouped):
        rows = grouped[buy_date]
        rows.sort(
            key=lambda item: (
                -float(item.get("capital_weight") or 0.0),
                -float(item.get("pred_prob") or 0.0),
                str(item.get("stock_code") or ""),
            )
        )
        for rank, row in enumerate(rows, start=1):
            capital_weight = float(row.get("capital_weight") or 0.0)
            signal_date = str(row.get("entry_signal_date") or "").strip()
            signals.append(
                {
                    "signal_date": signal_date,
                    "buy_date": buy_date,
                    "symbol": to_gm_symbol(row["stock_code"]),
                    "stock_code": row["stock_code"],
                    "name": row.get("name"),
                    "rank": rank,
                    "pred_prob": row.get("pred_prob"),
                    "atr_ratio": "",
                    "target_pct": round(capital_weight * target_total_pct, 10),
                }
            )
    return signals


def write_signal_rows(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    args = parse_args(argv)
    trades = load_trade_rows(Path(args.trades))
    signals = build_signal_rows(trades, args.target_total_pct, load_next_trade_dates(args.db))
    if not signals:
        raise SystemExit("No valid signal rows were built from trade records.")
    output = Path(args.output)
    write_signal_rows(signals, output)
    print(f"signals: {len(signals)}")
    print(f"buy_days: {len({row['buy_date'] for row in signals})}")
    print(f"output: {output}")


if __name__ == "__main__":
    main()
