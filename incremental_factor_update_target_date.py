from __future__ import annotations

import argparse
import concurrent.futures
import gc
import json
import sqlite3
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from build_production_factor_parts import (
    KEY_COLUMNS,
    apply_industry_encode,
    default_industry_encode_mapping_path,
    extend_industry_encode_mapping,
    is_future_or_label_column,
    is_source_limited_column,
    load_industry_encode_mapping,
    production_raw_columns,
    save_industry_encode_mapping,
)
from data_process_module import group_factor_eng
from gtja_alpha_workflow import (
    GTJA_ALPHA_COLUMNS,
    compute_gtja_alpha_from_raw_factor,
    required_gtja_raw_columns,
)
from project_paths import resolve_data_dir
from rebuild_factor_data_batched import _normalize_types
from stock_daily_data_route import resolve_stock_daily_db_path


warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


def _part_index(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[-1])


def _read_part_codes(raw_part_path: Path) -> list[str]:
    table = pq.read_table(raw_part_path, columns=["stock_code"])
    return sorted({str(value) for value in table.column("stock_code").to_pylist() if value is not None})


def _read_stock_daily(db_path: Path, codes: list[str], read_start: str, target_date: str) -> pd.DataFrame:
    placeholders = ",".join(["?"] * len(codes))
    query = f"""
        SELECT *
        FROM STOCK_DAILY_DATA
        WHERE stock_code IN ({placeholders})
          AND trade_date >= ?
          AND trade_date <= ?
        ORDER BY stock_code, trade_date
    """
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=120) as conn:
        frame = pd.read_sql(query, conn, params=[*codes, read_start, target_date])
    frame.columns = frame.columns.str.lower()
    return frame


def _compute_raw_target(source: pd.DataFrame, target_date: str) -> pd.DataFrame:
    raw_lookback = _compute_raw_lookback(source)
    if raw_lookback.empty:
        return pd.DataFrame()
    return raw_lookback[raw_lookback["trade_date"].eq(target_date)].copy()


def _compute_raw_lookback(source: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for _, group in source.groupby("stock_code", sort=False):
        factor = group_factor_eng(group.sort_values("trade_date").copy())
        factor["trade_date"] = factor["trade_date"].astype(str)
        frames.append(factor)
    if not frames:
        return pd.DataFrame()
    return _normalize_types(pd.concat(frames, ignore_index=True))


def _align_to_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    aligned = frame.copy()
    for column in columns:
        if column not in aligned.columns:
            aligned[column] = np.nan
    return aligned[columns]


def _replace_target_rows(part_path: Path, target_rows: pd.DataFrame, target_date: str) -> tuple[int, int]:
    existing = pd.read_parquet(part_path)
    existing["trade_date"] = existing["trade_date"].astype(str)
    old_target_rows = int(existing["trade_date"].eq(target_date).sum())
    keep = existing[~existing["trade_date"].eq(target_date)].copy()
    target = _align_to_columns(target_rows, list(existing.columns))
    merged = pd.concat([keep, target], ignore_index=True)
    merged.sort_values(KEY_COLUMNS, inplace=True)
    merged.replace([np.inf, -np.inf], np.nan, inplace=True)

    temp_path = part_path.with_name(f"{part_path.name}.tmp_{target_date}")
    merged.to_parquet(temp_path, index=False)
    temp_path.replace(part_path)
    return old_target_rows, int(target.shape[0])


def _read_raw_window_for_gtja(raw_part_path: Path, read_start: str, target_date: str) -> pd.DataFrame:
    schema = pq.ParquetFile(raw_part_path).schema.names
    needed = [column for column in required_gtja_raw_columns() if column in schema]
    raw = pd.read_parquet(
        raw_part_path,
        columns=needed,
        filters=[("trade_date", ">=", read_start), ("trade_date", "<=", target_date)],
    )
    raw["trade_date"] = raw["trade_date"].astype(str)
    raw.sort_values(KEY_COLUMNS, inplace=True)
    return raw


def _compute_production_target(raw_part_path: Path, target_date: str, read_start: str) -> pd.DataFrame:
    schema = pq.ParquetFile(raw_part_path).schema.names
    raw_columns = production_raw_columns(schema)
    raw_target = pd.read_parquet(raw_part_path, columns=raw_columns, filters=[("trade_date", "=", target_date)])
    raw_target["trade_date"] = raw_target["trade_date"].astype(str)
    raw_target = raw_target.sort_values(KEY_COLUMNS)

    gtja_input = _read_raw_window_for_gtja(raw_part_path, read_start, target_date)
    gtja = compute_gtja_alpha_from_raw_factor(
        gtja_input,
        encode=False,
        drop_ts=True,
        cross_sectional_rank_mode="rank",
    )
    gtja["trade_date"] = gtja["trade_date"].astype(str)
    gtja_target = gtja[gtja["trade_date"].eq(target_date)].copy()
    gtja_keep = [*KEY_COLUMNS, *[column for column in GTJA_ALPHA_COLUMNS if column in gtja_target.columns]]
    merged = raw_target.merge(gtja_target[gtja_keep], on=KEY_COLUMNS, how="left", validate="one_to_one")
    mapping = extend_industry_encode_mapping(
        load_industry_encode_mapping(default_industry_encode_mapping_path()),
        merged["industry"] if "industry" in merged.columns else [],
    )
    save_industry_encode_mapping(mapping, default_industry_encode_mapping_path())
    merged = apply_industry_encode(merged, mapping)
    merged.replace([np.inf, -np.inf], np.nan, inplace=True)
    return merged


def process_part(
    part_index: int,
    db_path: str,
    raw_parts_dir: str,
    production_parts_dir: str,
    target_date: str,
    read_start: str,
) -> dict:
    started_at = time.time()
    raw_part_path = Path(raw_parts_dir) / f"raw_part_{part_index:04d}.parquet"
    production_part_path = Path(production_parts_dir) / f"production_factor_part_{part_index:04d}.parquet"
    if not raw_part_path.exists():
        raise FileNotFoundError(raw_part_path)
    if not production_part_path.exists():
        raise FileNotFoundError(production_part_path)

    codes = _read_part_codes(raw_part_path)
    source = _read_stock_daily(Path(db_path), codes, read_start, target_date)
    raw_target = _compute_raw_target(source, target_date)
    if raw_target.empty:
        return {
            "part_index": part_index,
            "status": "no_target_rows",
            "stock_count": len(codes),
            "source_rows": int(source.shape[0]),
            "elapsed_seconds": round(time.time() - started_at, 3),
        }

    raw_old_rows, raw_new_rows = _replace_target_rows(raw_part_path, raw_target, target_date)
    production_target = _compute_production_target(raw_part_path, target_date, read_start)
    production_old_rows, production_new_rows = _replace_target_rows(
        production_part_path,
        production_target,
        target_date,
    )

    del source, raw_target, production_target
    gc.collect()
    return {
        "part_index": part_index,
        "status": "updated",
        "stock_count": len(codes),
        "raw_old_rows": raw_old_rows,
        "raw_new_rows": raw_new_rows,
        "production_old_rows": production_old_rows,
        "production_new_rows": production_new_rows,
        "elapsed_seconds": round(time.time() - started_at, 3),
    }


def _target_date_codes(db_path: Path, target_date: str) -> set[str]:
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=120) as conn:
        rows = conn.execute(
            "SELECT stock_code FROM STOCK_DAILY_DATA WHERE trade_date = ?",
            [target_date],
        ).fetchall()
    return {str(row[0]) for row in rows}


def _raw_part_codes(raw_parts_dir: Path) -> set[str]:
    codes: set[str] = set()
    for path in sorted(raw_parts_dir.glob("raw_part_*.parquet")):
        codes.update(_read_part_codes(path))
    return codes


def process_new_stock_codes(
    codes: list[str],
    db_path: str,
    raw_parts_dir: str,
    production_parts_dir: str,
    target_date: str,
    read_start: str,
) -> dict:
    if not codes:
        return {"status": "no_new_stock_codes", "new_stock_count": 0}

    raw_parts = sorted(Path(raw_parts_dir).glob("raw_part_*.parquet"))
    if not raw_parts:
        raise RuntimeError(f"no raw parts found: {raw_parts_dir}")
    raw_part_path = raw_parts[-1]
    part_index = _part_index(raw_part_path)
    production_part_path = Path(production_parts_dir) / f"production_factor_part_{part_index:04d}.parquet"

    source = _read_stock_daily(Path(db_path), codes, read_start, target_date)
    raw_lookback = _compute_raw_lookback(source)
    raw_target_new = raw_lookback[raw_lookback["trade_date"].eq(target_date)].copy()
    if raw_target_new.empty:
        return {
            "status": "new_stock_codes_no_target_rows",
            "new_stock_count": len(codes),
            "new_stock_codes": codes,
            "part_index": part_index,
        }

    existing_raw_target = pd.read_parquet(raw_part_path, filters=[("trade_date", "=", target_date)])
    combined_raw_target = pd.concat([existing_raw_target, raw_target_new], ignore_index=True)
    combined_raw_target = combined_raw_target.drop_duplicates(KEY_COLUMNS, keep="last")
    raw_old_rows, raw_new_rows = _replace_target_rows(raw_part_path, combined_raw_target, target_date)

    raw_schema = pq.ParquetFile(raw_part_path).schema.names
    raw_columns = production_raw_columns(raw_schema)
    raw_target_prod_new = _align_to_columns(raw_target_new, raw_schema)[raw_columns]
    gtja_columns = [column for column in required_gtja_raw_columns() if column in raw_lookback.columns]
    gtja_input = raw_lookback[gtja_columns].copy().sort_values(KEY_COLUMNS)
    gtja = compute_gtja_alpha_from_raw_factor(
        gtja_input,
        encode=False,
        drop_ts=True,
        cross_sectional_rank_mode="rank",
    )
    gtja["trade_date"] = gtja["trade_date"].astype(str)
    gtja_target = gtja[gtja["trade_date"].eq(target_date)].copy()
    gtja_keep = [*KEY_COLUMNS, *[column for column in GTJA_ALPHA_COLUMNS if column in gtja_target.columns]]
    production_target_new = raw_target_prod_new.merge(
        gtja_target[gtja_keep],
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    mapping = extend_industry_encode_mapping(
        load_industry_encode_mapping(default_industry_encode_mapping_path()),
        production_target_new["industry"] if "industry" in production_target_new.columns else [],
    )
    save_industry_encode_mapping(mapping, default_industry_encode_mapping_path())
    production_target_new = apply_industry_encode(production_target_new, mapping)
    production_target_new.replace([np.inf, -np.inf], np.nan, inplace=True)

    existing_production_target = pd.read_parquet(production_part_path, filters=[("trade_date", "=", target_date)])
    combined_production_target = pd.concat([existing_production_target, production_target_new], ignore_index=True)
    combined_production_target = combined_production_target.drop_duplicates(KEY_COLUMNS, keep="last")
    production_old_rows, production_new_rows = _replace_target_rows(
        production_part_path,
        combined_production_target,
        target_date,
    )
    return {
        "status": "new_stock_codes_updated",
        "part_index": part_index,
        "new_stock_count": len(codes),
        "new_stock_codes": codes,
        "raw_old_rows": raw_old_rows,
        "raw_new_rows": raw_new_rows,
        "production_old_rows": production_old_rows,
        "production_new_rows": production_new_rows,
    }


def _selected_parts(raw_parts_dir: Path, start_part: int | None, end_part: int | None) -> list[int]:
    indexes = [_part_index(path) for path in sorted(raw_parts_dir.glob("raw_part_*.parquet"))]
    if start_part is not None:
        indexes = [idx for idx in indexes if idx >= start_part]
    if end_part is not None:
        indexes = [idx for idx in indexes if idx <= end_part]
    return indexes


def _audit_parts(parts_dir: Path, target_date: str) -> dict:
    part_paths = sorted(parts_dir.glob("production_factor_part_*.parquet"))
    rows = 0
    files_with_target = 0
    schema = pq.ParquetFile(part_paths[0]).schema.names if part_paths else []
    future_columns = [column for column in schema if is_future_or_label_column(column)]
    source_limited_columns = [column for column in schema if is_source_limited_column(column)]
    for path in part_paths:
        table = pq.read_table(path, columns=["trade_date"], filters=[("trade_date", "=", target_date)])
        count = table.num_rows
        if count:
            files_with_target += 1
            rows += count
    return {
        "part_count": len(part_paths),
        "files_with_target_date": files_with_target,
        "rows_target_date": rows,
        "column_count": len(schema),
        "future_column_count": len(future_columns),
        "source_limited_column_count": len(source_limited_columns),
        "future_columns": future_columns,
        "source_limited_columns": source_limited_columns,
    }


def parse_args(argv=None):
    data_dir = resolve_data_dir()
    parser = argparse.ArgumentParser(description="Incrementally append one target date to raw and production factor parts.")
    parser.add_argument("--db-path", default=str(resolve_stock_daily_db_path()))
    parser.add_argument("--raw-parts-dir", default=str(data_dir / "raw_factor_by_stock_parts"))
    parser.add_argument("--production-parts-dir", default=str(data_dir / "production_factor_parts"))
    parser.add_argument("--target-date", required=True)
    parser.add_argument("--read-start", default="20250101")
    parser.add_argument("--start-part", type=int)
    parser.add_argument("--end-part", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--report-path")
    parser.add_argument("--no-new-stock-fill", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    raw_parts_dir = Path(args.raw_parts_dir)
    production_parts_dir = Path(args.production_parts_dir)
    part_indexes = _selected_parts(raw_parts_dir, args.start_part, args.end_part)
    if not part_indexes:
        raise RuntimeError(f"no raw parts selected: {raw_parts_dir}")

    print(
        f"incremental_factor_update_start target_date={args.target_date} parts={len(part_indexes)} "
        f"workers={args.workers} read_start={args.read_start}",
        flush=True,
    )
    worker_args = [
        (
            idx,
            args.db_path,
            str(raw_parts_dir),
            str(production_parts_dir),
            args.target_date,
            args.read_start,
        )
        for idx in part_indexes
    ]
    results: list[dict] = []
    if args.workers <= 1:
        for item in worker_args:
            result = process_part(*item)
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_part, *item) for item in worker_args]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                print(json.dumps(result, ensure_ascii=False), flush=True)

    new_stock_result = {"status": "skipped"}
    if not args.no_new_stock_fill and args.start_part is None and args.end_part is None:
        missing_codes = sorted(_target_date_codes(Path(args.db_path), args.target_date) - _raw_part_codes(raw_parts_dir))
        new_stock_result = process_new_stock_codes(
            missing_codes,
            args.db_path,
            str(raw_parts_dir),
            str(production_parts_dir),
            args.target_date,
            args.read_start,
        )
        print(json.dumps(new_stock_result, ensure_ascii=False), flush=True)

    results.sort(key=lambda row: row["part_index"])
    audit = _audit_parts(production_parts_dir, args.target_date)
    report = {
        "target_date": args.target_date,
        "read_start": args.read_start,
        "db_path": str(Path(args.db_path)),
        "raw_parts_dir": str(raw_parts_dir),
        "production_parts_dir": str(production_parts_dir),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "selected_part_count": len(part_indexes),
        "updated_part_count": sum(1 for row in results if row.get("status") == "updated"),
        "raw_rows_written": sum(int(row.get("raw_new_rows", 0)) for row in results),
        "production_rows_written": sum(int(row.get("production_new_rows", 0)) for row in results),
        "new_stock_result": new_stock_result,
        "results": results,
        "production_audit": audit,
    }
    if args.report_path:
        report_path = Path(args.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"report_written path={report_path}", flush=True)
    print(json.dumps({k: report[k] for k in ["updated_part_count", "raw_rows_written", "production_rows_written"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
