from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_load_module import get_pro
from stock_pool_module import DEFAULT_INDEX_CODES, DEFAULT_INDEX_WEIGHT_DATE, fetch_index_stock_pool, write_stock_pool


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Export an index-constituent stock pool CSV.")
    parser.add_argument("--config", default="./config.json")
    parser.add_argument("--index-codes", default=",".join(DEFAULT_INDEX_CODES))
    parser.add_argument("--trade-date", default=DEFAULT_INDEX_WEIGHT_DATE)
    parser.add_argument("--output", default="../data_file/stock_pool_hs300_zz500.csv")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    ts_pro = get_pro(config["datasource"]["tushare_token"])
    index_codes = [item.strip() for item in args.index_codes.split(",") if item.strip()]
    codes = fetch_index_stock_pool(ts_pro, index_codes=index_codes, trade_date=args.trade_date)
    write_stock_pool(codes, args.output)
    print(f"stock_pool_csv: {Path(args.output)}")
    print(f"stock_count: {len(codes)}")
    print(f"index_codes: {','.join(index_codes)}")
    print(f"trade_date: {args.trade_date}")


if __name__ == "__main__":
    main()
