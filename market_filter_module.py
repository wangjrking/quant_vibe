"""Market regime filters for strategy backtests."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd


def build_market_filter(index_data: pd.DataFrame, index_code: str = "932000.CSI", ma_window: int = 20) -> pd.DataFrame:
    frame = index_data.copy()
    frame["trade_date"] = frame["trade_date"].astype(str)
    if "ts_code" in frame.columns:
        frame = frame[frame["ts_code"] == index_code]
    frame = frame.sort_values("trade_date")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame["market_ma"] = frame["close"].rolling(ma_window, min_periods=ma_window).mean()
    frame["market_ok"] = (frame["close"] >= frame["market_ma"]).astype(int)
    return frame[["trade_date", "market_ok", "market_ma"]]


def build_multi_index_market_filter(
    index_data: pd.DataFrame,
    index_codes: list[str] | tuple[str, ...],
    ma_window: int = 20,
    mode: str = "all",
) -> pd.DataFrame:
    """Combine multiple index moving-average filters into one market_ok flag."""
    filters = []
    for index_code in index_codes:
        item = build_market_filter(index_data, index_code=index_code, ma_window=ma_window)
        item = item.rename(
            columns={
                "market_ok": f"market_ok_{index_code}",
                "market_ma": f"market_ma_{index_code}",
            }
        )
        filters.append(item)
    if not filters:
        raise ValueError("index_codes must not be empty")

    merged = filters[0]
    for item in filters[1:]:
        merged = merged.merge(item, on="trade_date", how="outer")
    ok_cols = [col for col in merged.columns if col.startswith("market_ok_")]
    merged[ok_cols] = merged[ok_cols].fillna(0).astype(int)
    if mode == "all":
        merged["market_ok"] = (merged[ok_cols].sum(axis=1) == len(ok_cols)).astype(int)
    elif mode == "any":
        merged["market_ok"] = (merged[ok_cols].sum(axis=1) > 0).astype(int)
    else:
        raise ValueError("mode must be all or any")
    return merged.sort_values("trade_date").reset_index(drop=True)


def merge_market_filter(rows: list[dict], market_filter: pd.DataFrame) -> list[dict]:
    lookup = market_filter.set_index("trade_date")["market_ok"].to_dict()
    merged = []
    for row in rows:
        item = dict(row)
        item["market_ok"] = lookup.get(str(item.get("trade_date")), 0)
        merged.append(item)
    return merged


def load_index_data(data_dir: str | Path) -> pd.DataFrame:
    data_dir = Path(data_dir)
    parquet_path = data_dir / "index_daily.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    with sqlite3.connect(data_dir / "odb.db") as conn:
        return pd.read_sql('SELECT * FROM "index_daily"', conn)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build market regime filter CSV.")
    parser.add_argument("--data-dir", default="../data_file")
    parser.add_argument("--index-code", default="932000.CSI")
    parser.add_argument("--index-mode", default="all", choices=["all", "any"])
    parser.add_argument("--ma-window", type=int, default=20)
    parser.add_argument("--output", default="../data_file/market_filter.csv")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    index_data = load_index_data(args.data_dir)
    index_codes = [item.strip() for item in args.index_code.split(",") if item.strip()]
    if len(index_codes) == 1:
        market_filter = build_market_filter(index_data, index_codes[0], args.ma_window)
    else:
        market_filter = build_multi_index_market_filter(index_data, index_codes, args.ma_window, args.index_mode)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    market_filter.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"market_filter_csv: {output}")


if __name__ == "__main__":
    main()
