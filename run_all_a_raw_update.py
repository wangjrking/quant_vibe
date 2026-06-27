from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from data_load_module import date_range_pandas, get_pro
from project_paths import load_config, resolve_data_dir, resolve_data_path
from raw_table_db_module import replace_raw_table_full
from raw_data_update_module import (
    RAW_TABLE_SPECS,
    append_parquet_dedup,
    load_existing_stock_codes,
    load_stock_codes,
    next_start_date,
)


DEFAULT_TABLES = "daily,daily_basic,stk_factor,moneyflow,limit_list,cyq_perf,adj_factor,stock_st,index_daily"
DEFAULT_INDEX_CODES = ("000300.SH", "000905.SH", "000852.SH", "932000.CSI")


def _query_retry(call):
    import time

    while True:
        try:
            return call()
        except Exception:
            time.sleep(61)


def _fetch_stock_table(ts_pro, table: str, code: str, start: str, end: str) -> pd.DataFrame:
    if table == "daily":
        return _query_retry(lambda: ts_pro.query("daily", ts_code=code, start_date=start, end_date=end))
    if table == "daily_basic":
        return _query_retry(lambda: ts_pro.query("daily_basic", ts_code=code, start_date=start, end_date=end))
    if table == "moneyflow":
        return _query_retry(lambda: ts_pro.moneyflow(ts_code=code, start_date=start, end_date=end))
    if table == "stk_factor":
        return _query_retry(lambda: ts_pro.stk_factor_pro(ts_code=code, start_date=start, end_date=end))
    if table == "limit_list":
        return _query_retry(lambda: ts_pro.limit_list_d(ts_code=code, limit_type="U", start_date=start, end_date=end))
    if table == "cyq_perf":
        return _query_retry(lambda: ts_pro.cyq_perf(ts_code=code, start_date=start, end_date=end))
    raise ValueError(f"unsupported per-stock table: {table}")


def _fetch_date_table(ts_pro, table: str, start: str, end: str, index_codes: tuple[str, ...]) -> pd.DataFrame:
    frames = []
    if table == "adj_factor":
        for trade_date in date_range_pandas(start, end):
            frames.append(_query_retry(lambda trade_date=trade_date: ts_pro.adj_factor(trade_date=trade_date)))
    elif table == "top_list":
        for trade_date in date_range_pandas(start, end):
            frames.append(_query_retry(lambda trade_date=trade_date: ts_pro.top_list(trade_date=trade_date)))
    elif table == "ths_hot":
        for trade_date in date_range_pandas(start, end):
            frames.append(_query_retry(lambda trade_date=trade_date: ts_pro.ths_hot(trade_date=trade_date, market="热股")))
    elif table == "dc_hot":
        for trade_date in date_range_pandas(start, end):
            frames.append(_query_retry(lambda trade_date=trade_date: ts_pro.dc_hot(trade_date=trade_date, market="A股市场", hot_type="人气榜")))
    elif table == "stock_st":
        for trade_date in date_range_pandas(start, end):
            frames.append(_query_retry(lambda trade_date=trade_date: ts_pro.stock_st(trade_date=trade_date)))
    elif table == "index_daily":
        for index_code in index_codes:
            frames.append(_query_retry(lambda index_code=index_code: ts_pro.query("index_daily", ts_code=index_code, start_date=start, end_date=end)))
    else:
        raise ValueError(f"unsupported date table: {table}")
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _sync_sqlite_table(data_dir: Path, file_name: str, table_name: str) -> None:
    frame = pd.read_parquet(data_dir / file_name)
    replace_raw_table_full(data_dir, table_name, frame)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Incrementally update local raw data for the all-A-share universe.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--stock-pool", default="stock_pool_all_a.csv")
    parser.add_argument("--start", default="20100101")
    parser.add_argument("--end", required=True)
    parser.add_argument("--tables", default=DEFAULT_TABLES)
    parser.add_argument("--full-refresh", action="store_true")
    parser.add_argument("--limit-stocks", type=int)
    parser.add_argument("--sync-sqlite", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    data_dir = resolve_data_dir(args.data_dir)
    config = load_config(args.config)
    ts_pro = get_pro(config["datasource"]["tushare_token"])
    stock_pool_path = resolve_data_path(args.stock_pool, data_dir=data_dir)
    stock_codes = load_stock_codes(stock_pool_path)
    if args.limit_stocks:
        stock_codes = stock_codes[: args.limit_stocks]
    tables = [item.strip() for item in args.tables.split(",") if item.strip()]
    print(
        f"raw_update_start stocks={len(stock_codes)} tables={','.join(tables)} end={args.end} "
        f"stock_pool={stock_pool_path}",
        flush=True,
    )

    for table in tables:
        spec = RAW_TABLE_SPECS[table]
        path = data_dir / spec.file_name
        start = args.start if args.full_refresh else next_start_date(path, args.start, spec.date_column)
        if spec.per_stock:
            frames = []
            existing_codes = set() if args.full_refresh else load_existing_stock_codes(path, spec.key_columns[0])
            incremental_codes = stock_codes if args.full_refresh else [code for code in stock_codes if code in existing_codes]
            missing_codes = [] if args.full_refresh else [code for code in stock_codes if code not in existing_codes]

            if start <= args.end and incremental_codes:
                for idx, code in enumerate(incremental_codes, start=1):
                    frame = _fetch_stock_table(ts_pro, table, code, start, args.end)
                    if frame is not None and not frame.empty:
                        frames.append(frame)
                    if idx % 100 == 0:
                        print(
                            f"table_progress table={table} phase=incremental stocks={idx}/{len(incremental_codes)}",
                            flush=True,
                        )
            elif start > args.end and not missing_codes:
                print(f"table_skip table={table} start={start} end={args.end}", flush=True)
                continue

            if missing_codes:
                for idx, code in enumerate(missing_codes, start=1):
                    frame = _fetch_stock_table(ts_pro, table, code, args.start, args.end)
                    if frame is not None and not frame.empty:
                        frames.append(frame)
                    if idx % 100 == 0:
                        print(
                            f"table_progress table={table} phase=missing stocks={idx}/{len(missing_codes)}",
                            flush=True,
                        )
            rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        else:
            if start > args.end:
                print(f"table_skip table={table} start={start} end={args.end}", flush=True)
                continue
            rows = _fetch_date_table(ts_pro, table, start, args.end, DEFAULT_INDEX_CODES)
        added = append_parquet_dedup(path, rows, spec.key_columns)
        print(f"table_done table={table} downloaded={added} file={path}", flush=True)
        if args.sync_sqlite:
            sqlite_name = {
                "daily": "daily_data",
                "daily_basic": "daily_index_data",
                "limit_list": "limit_list_data",
            }.get(table, table)
            _sync_sqlite_table(data_dir, spec.file_name, sqlite_name)
            print(f"sqlite_synced table={sqlite_name}", flush=True)
    print("raw_update_done", flush=True)


if __name__ == "__main__":
    main()
