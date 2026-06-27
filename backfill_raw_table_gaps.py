from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from data_load_module import get_pro, optimize_dtypes
from project_paths import load_config, resolve_data_dir, resolve_data_path
from raw_data_update_module import RAW_TABLE_SPECS, append_parquet_dedup, load_stock_codes


def _fetch_stock_table(ts_pro, table: str, code: str, start: str, end: str) -> pd.DataFrame:
    while True:
        try:
            if table == "stk_factor":
                return ts_pro.stk_factor_pro(ts_code=code, start_date=start, end_date=end)
            if table == "moneyflow":
                return ts_pro.moneyflow(ts_code=code, start_date=start, end_date=end)
            if table == "cyq_perf":
                return ts_pro.cyq_perf(ts_code=code, start_date=start, end_date=end)
            if table == "daily":
                return ts_pro.query("daily", ts_code=code, start_date=start, end_date=end)
            if table == "daily_basic":
                return ts_pro.query("daily_basic", ts_code=code, start_date=start, end_date=end)
            raise ValueError(f"unsupported table: {table}")
        except Exception as exc:
            print(f"query_retry table={table} code={code} err={exc}", flush=True)
            time.sleep(61)


def _existing_codes_for_date(path: Path, date: str, code_col: str = "ts_code") -> set[str]:
    if not path.exists():
        return set()
    frame = pd.read_parquet(path, columns=[code_col, "trade_date"])
    day = frame[frame["trade_date"].astype(str) == str(date)]
    return set(day[code_col].astype(str).str.upper())


def _load_done(done_path: Path) -> set[str]:
    if not done_path.exists():
        return set()
    done = set()
    with done_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                done.add(line.strip())
    return done


def _mark_done(done_path: Path, key: str) -> None:
    done_path.parent.mkdir(parents=True, exist_ok=True)
    with done_path.open("a", encoding="utf-8") as file:
        file.write(key + "\n")


def _write_part(part_path: Path, frame: pd.DataFrame) -> None:
    part_path.parent.mkdir(parents=True, exist_ok=True)
    if frame is None or frame.empty:
        pd.DataFrame().to_parquet(part_path, index=False)
        return
    optimize_dtypes(frame).to_parquet(part_path, index=False)


def _merge_parts(table: str, data_dir: Path, parts_dir: Path) -> int:
    spec = RAW_TABLE_SPECS[table]
    part_files = sorted((parts_dir / table).glob("*.parquet"))
    frames = []
    for path in part_files:
        frame = pd.read_parquet(path)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return 0
    rows = pd.concat(frames, ignore_index=True)
    rows = rows.replace({"None": pd.NA, "none": pd.NA, "nan": pd.NA, "NaN": pd.NA, "": pd.NA})
    for col in rows.columns:
        if col in {"ts_code", "trade_date", "name", "area", "industry", "cnspell", "market", "list_date", "act_name", "act_ent_type"}:
            continue
        if rows[col].dtype == "object":
            converted = pd.to_numeric(rows[col], errors="coerce")
            if converted.notna().sum() >= rows[col].notna().sum() * 0.8:
                rows[col] = converted
    return append_parquet_dedup(data_dir / spec.file_name, rows, spec.key_columns)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Backfill historical partial-coverage raw table gaps by stock.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--data-dir")
    parser.add_argument("--stock-pool", default="stock_pool_all_a.csv")
    parser.add_argument("--tables", default="stk_factor,moneyflow,cyq_perf")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--coverage-date", required=True, help="A representative date in the bad range.")
    parser.add_argument("--parts-dir", required=True)
    parser.add_argument("--limit-stocks", type=int)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--download", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    data_dir = resolve_data_dir(args.data_dir)
    parts_dir = Path(args.parts_dir)
    tables = [item.strip() for item in args.tables.split(",") if item.strip()]
    config = load_config(args.config)
    ts_pro = get_pro(config["datasource"]["tushare_token"])
    stock_pool_path = resolve_data_path(args.stock_pool, data_dir=data_dir)
    all_codes = load_stock_codes(stock_pool_path)
    if args.limit_stocks:
        all_codes = all_codes[: args.limit_stocks]

    summary = {
        "tables": tables,
        "start": args.start,
        "end": args.end,
        "coverage_date": args.coverage_date,
        "parts_dir": str(parts_dir),
    }
    print("backfill_start " + json.dumps(summary, ensure_ascii=False), flush=True)

    if args.download:
        for table in tables:
            spec = RAW_TABLE_SPECS[table]
            raw_path = data_dir / spec.file_name
            existing = _existing_codes_for_date(raw_path, args.coverage_date, spec.key_columns[0])
            missing_codes = [code for code in all_codes if code not in existing]
            done_path = parts_dir / table / "_done.txt"
            done = _load_done(done_path)
            print(
                f"table_start table={table} all_codes={len(all_codes)} existing_on_date={len(existing)} missing={len(missing_codes)} done={len(done)}",
                flush=True,
            )
            for idx, code in enumerate(missing_codes, start=1):
                key = f"{table}|{code}|{args.start}|{args.end}"
                if key in done:
                    continue
                part_path = parts_dir / table / f"{code.replace('.', '_')}_{args.start}_{args.end}.parquet"
                frame = _fetch_stock_table(ts_pro, table, code, args.start, args.end)
                _write_part(part_path, frame)
                _mark_done(done_path, key)
                if idx % 20 == 0:
                    print(f"table_progress table={table} downloaded_missing={idx}/{len(missing_codes)} code={code}", flush=True)
                if args.sleep_seconds:
                    time.sleep(args.sleep_seconds)
            print(f"table_download_done table={table}", flush=True)

    if args.merge:
        for table in tables:
            added = _merge_parts(table, data_dir, parts_dir)
            print(f"table_merge_done table={table} added_rows={added}", flush=True)

    print("backfill_done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
