from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from stock_pool_module import build_all_a_stock_pool, write_stock_pool


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Export a local all-A-share stock pool from stock_basic_data.parquet.")
    parser.add_argument("--data-dir", default="data_file")
    parser.add_argument("--output", default="data_file/stock_pool_all_a.csv")
    parser.add_argument("--exclude-bj", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    data_dir = Path(args.data_dir)
    stock_basic = pd.read_parquet(data_dir / "stock_basic_data.parquet")
    codes = build_all_a_stock_pool(stock_basic, include_bj=not args.exclude_bj)
    write_stock_pool(codes, args.output)
    print(f"stock_pool_csv: {Path(args.output)}")
    print(f"stock_count: {len(codes)}")
    print(f"include_bj: {not args.exclude_bj}")


if __name__ == "__main__":
    main()
