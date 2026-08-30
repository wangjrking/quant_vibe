from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from build_production_factor_parts import (
    GTJA_TO_PRODUCTION_COLUMN_MAP,
    KEY_COLUMNS,
    apply_industry_encode,
    default_industry_encode_mapping_path,
    production_raw_columns,
    update_industry_encode_mapping,
)
from gtja_alpha_workflow import GTJA_ALPHA_COLUMNS, compute_gtja_alpha_from_raw_factor, required_gtja_raw_columns
from incremental_factor_update_target_date import _compute_raw_lookback, _compute_raw_target
from l3_active_writer_lease import acquire_active_l3_writer_lease, release_active_l3_writer_lease
from l3_duckdb_sync import (
    resolve_l3_feature_duckdb_path,
    resolve_l3_feature_duckdb_table,
    resolve_l3_label_duckdb_path,
    resolve_l3_label_duckdb_table,
    sync_feature_parts_full_to_duckdb,
    sync_feature_parts_target_date_to_duckdb,
    sync_label_parts_full_to_duckdb,
)
from project_paths import resolve_data_dir
from stock_daily_data_route import resolve_stock_daily_duckdb_path


ENV_ALLOW_LEGACY_L3_PARQUET_PARTS = "QUANT_ALLOW_LEGACY_L3_PARQUET_PARTS"
RAW_TARGET_REQUIRED_COLUMNS = [
    "stock_code",
    "trade_date",
    "name",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "close_qfq",
    "pre_close_qfq",
    "vol",
    "amount",
    "his_low",
    "his_high",
    "cost_5pct",
    "cost_15pct",
    "cost_50pct",
    "cost_85pct",
    "cost_95pct",
    "weight_avg",
    "fd_amount",
    "first_time",
    "last_time",
    "up_stat",
    "index_2000_close",
]


def _gtja_qfq_columns() -> list[str]:
    return [f"{column}_qfq" for column in GTJA_ALPHA_COLUMNS]


def _load_no_bj_codes(
    *,
    db_path: Path,
    read_start: str,
    target_date: str,
) -> list[str]:
    query = """
        SELECT DISTINCT stock_code
        FROM STOCK_DAILY_DATA
        WHERE trade_date >= ?
          AND trade_date <= ?
          AND stock_code NOT LIKE '%.BJ'
        ORDER BY stock_code
    """
    with duckdb.connect(str(db_path), read_only=True) as conn:
        rows = conn.execute(query, [read_start, target_date]).fetchall()
    return [str(row[0]) for row in rows if row and row[0] is not None]


def _split_code_buckets(codes: list[str], bucket_count: int) -> list[list[str]]:
    if not codes:
        return []
    bucket_count = max(1, min(bucket_count, len(codes)))
    bucket_size = (len(codes) + bucket_count - 1) // bucket_count
    return [codes[idx : idx + bucket_size] for idx in range(0, len(codes), bucket_size)]


def _compute_raw_target_bucket(
    codes: list[str],
    *,
    db_path: Path,
    read_start: str,
    target_date: str,
) -> dict[str, object]:
    started_at = time.time()
    placeholders = ",".join(["?"] * len(codes))
    query = f"""
        SELECT {", ".join(RAW_TARGET_REQUIRED_COLUMNS)}
        FROM STOCK_DAILY_DATA
        WHERE stock_code IN ({placeholders})
          AND trade_date >= ?
          AND trade_date <= ?
          AND stock_code NOT LIKE '%.BJ'
        ORDER BY stock_code, trade_date
    """
    read_started_at = time.time()
    with duckdb.connect(str(db_path), read_only=True) as conn:
        source = conn.execute(query, [*codes, read_start, target_date]).fetchdf()
    read_elapsed = round(time.time() - read_started_at, 3)
    source.columns = source.columns.str.lower()
    compute_started_at = time.time()
    target = _compute_raw_target(source, target_date)
    compute_elapsed = round(time.time() - compute_started_at, 3)
    if target.empty:
        target = pd.DataFrame()
    else:
        target = target[target["stock_code"].astype(str).str.endswith(".BJ") == False].copy()
        target["trade_date"] = target["trade_date"].astype(str)
        target.drop_duplicates(KEY_COLUMNS, keep="last", inplace=True)
        target.sort_values(KEY_COLUMNS, inplace=True)
    return {
        "bucket_codes": len(codes),
        "source_rows": int(source.shape[0]),
        "rows": int(target.shape[0]),
        "stocks": int(target["stock_code"].nunique()) if not target.empty else 0,
        "read_elapsed_seconds": read_elapsed,
        "compute_elapsed_seconds": compute_elapsed,
        "total_elapsed_seconds": round(time.time() - started_at, 3),
        "frame": target,
    }


def _compute_raw_recent_bucket(
    codes: list[str],
    *,
    db_path: Path,
    read_start: str,
    target_date: str,
    min_output_trade_date: str | None,
) -> dict[str, object]:
    started_at = time.time()
    placeholders = ",".join(["?"] * len(codes))
    query = f"""
        SELECT {", ".join(RAW_TARGET_REQUIRED_COLUMNS)}
        FROM STOCK_DAILY_DATA
        WHERE stock_code IN ({placeholders})
          AND trade_date >= ?
          AND trade_date <= ?
          AND stock_code NOT LIKE '%.BJ'
        ORDER BY stock_code, trade_date
    """
    read_started_at = time.time()
    with duckdb.connect(str(db_path), read_only=True) as conn:
        source = conn.execute(query, [*codes, read_start, target_date]).fetchdf()
    read_elapsed = round(time.time() - read_started_at, 3)
    source.columns = source.columns.str.lower()
    compute_started_at = time.time()
    recent = _compute_raw_lookback(source)
    compute_elapsed = round(time.time() - compute_started_at, 3)
    if recent.empty:
        recent = pd.DataFrame()
    else:
        recent["trade_date"] = recent["trade_date"].astype(str)
        if min_output_trade_date is not None:
            recent = recent[recent["trade_date"].gt(min_output_trade_date)].copy()
        recent = recent[recent["trade_date"].le(target_date)].copy()
        recent = recent[recent["stock_code"].astype(str).str.endswith(".BJ") == False].copy()
        recent.drop_duplicates(KEY_COLUMNS, keep="last", inplace=True)
        recent.sort_values(KEY_COLUMNS, inplace=True)
    return {
        "bucket_codes": len(codes),
        "source_rows": int(source.shape[0]),
        "rows": int(recent.shape[0]),
        "stocks": int(recent["stock_code"].nunique()) if not recent.empty else 0,
        "read_elapsed_seconds": read_elapsed,
        "compute_elapsed_seconds": compute_elapsed,
        "total_elapsed_seconds": round(time.time() - started_at, 3),
        "frame": recent,
    }


def _compute_raw_target_frames(
    *,
    db_path: Path,
    read_start: str,
    target_date: str,
    workers: int,
) -> pd.DataFrame:
    started_at = time.time()
    code_load_started_at = time.time()
    codes = _load_no_bj_codes(db_path=db_path, read_start=read_start, target_date=target_date)
    bucket_workers = max(1, min(4, len(codes), workers if workers > 1 else 4))
    code_buckets = _split_code_buckets(codes, bucket_workers)
    print(
        json.dumps(
            {
                "stage": "raw_target_bucket_plan",
                "read_start": read_start,
                "target_date": target_date,
                "column_count": len(RAW_TARGET_REQUIRED_COLUMNS),
                "codes": len(codes),
                "bucket_count": len(code_buckets),
                "bucket_workers": bucket_workers,
                "code_load_elapsed_seconds": round(time.time() - code_load_started_at, 3),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    results: list[dict[str, object]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=bucket_workers) as executor:
        futures = [
            executor.submit(
                _compute_raw_target_bucket,
                bucket,
                db_path=db_path,
                read_start=read_start,
                target_date=target_date,
            )
            for bucket in code_buckets
        ]
        for completed_idx, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(
                json.dumps(
                    {
                        "stage": "raw_target_bucket_finish",
                        "completed_buckets": completed_idx,
                        "total_buckets": len(code_buckets),
                        "bucket_codes": int(result["bucket_codes"]),
                        "source_rows": int(result["source_rows"]),
                        "rows": int(result["rows"]),
                        "stocks": int(result["stocks"]),
                        "read_elapsed_seconds": float(result["read_elapsed_seconds"]),
                        "compute_elapsed_seconds": float(result["compute_elapsed_seconds"]),
                        "total_elapsed_seconds": float(result["total_elapsed_seconds"]),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    frames = [result["frame"] for result in results if isinstance(result.get("frame"), pd.DataFrame) and not result["frame"].empty]
    if not frames:
        return pd.DataFrame()
    target = pd.concat(frames, ignore_index=True)
    target.drop_duplicates(KEY_COLUMNS, keep="last", inplace=True)
    target.sort_values(KEY_COLUMNS, inplace=True)
    print(
        json.dumps(
            {
                "stage": "raw_target_slice_compute_finish",
                "rows": int(target.shape[0]),
                "stocks": int(target["stock_code"].nunique()),
                "compute_elapsed_seconds": round(time.time() - started_at, 3),
                "total_elapsed_seconds": round(time.time() - started_at, 3),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return target


def _raw_parts_max_trade_date(raw_parts_dir: Path) -> str | None:
    raw_glob = str(raw_parts_dir / "raw_part_*.parquet")
    if not any(raw_parts_dir.glob("raw_part_*.parquet")):
        return None
    with duckdb.connect() as conn:
        row = conn.execute(
            "SELECT MAX(trade_date)::VARCHAR FROM read_parquet(?)",
            [raw_glob],
        ).fetchone()
    if not row or row[0] is None:
        return None
    return str(row[0])


def _compute_recent_raw_frames(
    *,
    db_path: Path,
    raw_parts_dir: Path,
    read_start: str,
    target_date: str,
    workers: int,
) -> tuple[pd.DataFrame, str | None]:
    started_at = time.time()
    historical_max_trade_date = _raw_parts_max_trade_date(raw_parts_dir)
    code_load_started_at = time.time()
    codes = _load_no_bj_codes(db_path=db_path, read_start=read_start, target_date=target_date)
    bucket_workers = max(1, min(4, len(codes), workers if workers > 1 else 4))
    code_buckets = _split_code_buckets(codes, bucket_workers)
    print(
        json.dumps(
            {
                "stage": "raw_recent_bucket_plan",
                "read_start": read_start,
                "target_date": target_date,
                "historical_max_trade_date": historical_max_trade_date,
                "column_count": len(RAW_TARGET_REQUIRED_COLUMNS),
                "codes": len(codes),
                "bucket_count": len(code_buckets),
                "bucket_workers": bucket_workers,
                "code_load_elapsed_seconds": round(time.time() - code_load_started_at, 3),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    results: list[dict[str, object]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=bucket_workers) as executor:
        futures = [
            executor.submit(
                _compute_raw_recent_bucket,
                bucket,
                db_path=db_path,
                read_start=read_start,
                target_date=target_date,
                min_output_trade_date=historical_max_trade_date,
            )
            for bucket in code_buckets
        ]
        for completed_idx, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(
                json.dumps(
                    {
                        "stage": "raw_recent_bucket_finish",
                        "completed_buckets": completed_idx,
                        "total_buckets": len(code_buckets),
                        "bucket_codes": int(result["bucket_codes"]),
                        "source_rows": int(result["source_rows"]),
                        "rows": int(result["rows"]),
                        "stocks": int(result["stocks"]),
                        "read_elapsed_seconds": float(result["read_elapsed_seconds"]),
                        "compute_elapsed_seconds": float(result["compute_elapsed_seconds"]),
                        "total_elapsed_seconds": float(result["total_elapsed_seconds"]),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    frames = [result["frame"] for result in results if isinstance(result.get("frame"), pd.DataFrame) and not result["frame"].empty]
    if not frames:
        return pd.DataFrame(), historical_max_trade_date
    recent = pd.concat(frames, ignore_index=True)
    recent.drop_duplicates(KEY_COLUMNS, keep="last", inplace=True)
    recent.sort_values(KEY_COLUMNS, inplace=True)
    print(
        json.dumps(
            {
                "stage": "raw_recent_slice_compute_finish",
                "historical_max_trade_date": historical_max_trade_date,
                "rows": int(recent.shape[0]),
                "stocks": int(recent["stock_code"].nunique()),
                "compute_elapsed_seconds": round(time.time() - started_at, 3),
                "total_elapsed_seconds": round(time.time() - started_at, 3),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return recent, historical_max_trade_date


def _build_gtja_target_frame(
    raw_parts_dir: Path,
    raw_recent: pd.DataFrame,
    *,
    read_start: str,
    target_date: str,
) -> pd.DataFrame:
    def _gtja_progress(event: dict[str, object]) -> None:
        print(
            json.dumps(
                {
                    "stage": "gtja_compute_batch_done",
                    **event,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    needed = required_gtja_raw_columns()
    historical_frames: list[pd.DataFrame] = []
    history_started_at = time.time()
    print(
        json.dumps(
            {
                "stage": "gtja_history_read_start",
                "read_start": read_start,
                "target_date": target_date,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    for path in sorted(raw_parts_dir.glob("raw_part_*.parquet")):
        frame = pd.read_parquet(
            path,
            columns=[column for column in needed if column in pq.ParquetFile(path).schema.names],
            filters=[("trade_date", ">=", read_start), ("trade_date", "<", target_date)],
        )
        if not frame.empty:
            frame = frame[frame["stock_code"].astype(str).str.endswith(".BJ") == False].copy()
        if not frame.empty:
            frame["trade_date"] = frame["trade_date"].astype(str)
            historical_frames.append(frame)
    history_concat_started_at = time.time()
    raw_recent_gtja = raw_recent[[column for column in needed if column in raw_recent.columns]].copy()
    raw_recent_gtja = raw_recent_gtja[raw_recent_gtja["stock_code"].astype(str).str.endswith(".BJ") == False].copy()
    raw_recent_gtja["trade_date"] = raw_recent_gtja["trade_date"].astype(str)
    gtja_input = pd.concat([*historical_frames, raw_recent_gtja], ignore_index=True)
    gtja_input.sort_values(KEY_COLUMNS, inplace=True)
    print(
        json.dumps(
            {
                "stage": "gtja_history_read_done",
                "history_frame_count": len(historical_frames),
                "input_rows": int(gtja_input.shape[0]),
                "input_stocks": int(gtja_input["stock_code"].nunique()) if not gtja_input.empty else 0,
                "read_elapsed_seconds": round(history_concat_started_at - history_started_at, 3),
                "concat_sort_elapsed_seconds": round(time.time() - history_concat_started_at, 3),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    compute_started_at = time.time()
    print(json.dumps({"stage": "gtja_compute_start"}, ensure_ascii=False), flush=True)
    gtja = compute_gtja_alpha_from_raw_factor(
        gtja_input,
        encode=False,
        drop_ts=True,
        cross_sectional_rank_mode="rank",
        output_dates=[target_date],
        alpha_batch_size=4,
        progress_callback=_gtja_progress,
    )
    print(
        json.dumps(
            {
                "stage": "gtja_compute_done",
                "rows": int(gtja.shape[0]),
                "stocks": int(gtja["stock_code"].nunique()) if not gtja.empty else 0,
                "compute_elapsed_seconds": round(time.time() - compute_started_at, 3),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    filter_started_at = time.time()
    print(json.dumps({"stage": "gtja_target_filter_start"}, ensure_ascii=False), flush=True)
    gtja["trade_date"] = gtja["trade_date"].astype(str)
    gtja_target = gtja[gtja["trade_date"].eq(target_date)].copy()
    keep = [*KEY_COLUMNS, *[column for column in GTJA_ALPHA_COLUMNS if column in gtja_target.columns]]
    print(
        json.dumps(
            {
                "stage": "gtja_target_filter_done",
                "rows": int(gtja_target.shape[0]),
                "stocks": int(gtja_target["stock_code"].nunique()) if not gtja_target.empty else 0,
                "filter_elapsed_seconds": round(time.time() - filter_started_at, 3),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return gtja_target[keep]


def _build_target_feature_frame(
    raw_parts_dir: Path,
    raw_target: pd.DataFrame,
    gtja_target: pd.DataFrame,
) -> pd.DataFrame:
    raw_columns = production_raw_columns(list(raw_target.columns))
    feature = raw_target[raw_columns].copy()
    gtja_production = gtja_target.rename(
        columns={column: GTJA_TO_PRODUCTION_COLUMN_MAP[column] for column in gtja_target.columns if column in GTJA_TO_PRODUCTION_COLUMN_MAP}
    )
    feature = feature.merge(gtja_production, on=KEY_COLUMNS, how="left", validate="one_to_one")
    mapping = update_industry_encode_mapping(
        feature["industry"] if "industry" in feature.columns else [],
        default_industry_encode_mapping_path(),
    )
    feature = apply_industry_encode(feature, mapping)
    feature.replace([np.inf, -np.inf], np.nan, inplace=True)
    feature.sort_values(KEY_COLUMNS, inplace=True)
    return feature


def _write_target_feature_part(feature: pd.DataFrame, output_dir: Path, target_date: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"production_factor_part_target_{target_date}.parquet"
    feature.to_parquet(output_path, index=False)
    return output_path


def _query_feature_audit(target_date: str) -> dict:
    db_path = resolve_l3_feature_duckdb_path()
    table_name = resolve_l3_feature_duckdb_table()
    qfq_gtja_cols = _gtja_qfq_columns()
    with duckdb.connect(str(db_path), read_only=True) as conn:
        total = conn.execute(
            f"""
            SELECT
                COUNT(*) AS rows_n,
                COUNT(DISTINCT stock_code) AS stocks_n,
                MIN(trade_date) AS min_d,
                MAX(trade_date) AS max_d,
                SUM(CASE WHEN stock_code LIKE '%.BJ' THEN 1 ELSE 0 END) AS bj_rows,
                COUNT(*) - COUNT(DISTINCT stock_code || '|' || trade_date) AS duplicate_rows
            FROM {table_name}
            """
        ).fetchone()
        cols = [row[1] for row in conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()]
        target = conn.execute(
            f"""
            SELECT
                COUNT(*) AS rows_n,
                COUNT(DISTINCT stock_code) AS stocks_n,
                SUM(CASE WHEN stock_code LIKE '%.BJ' THEN 1 ELSE 0 END) AS bj_rows
            FROM {table_name}
            WHERE trade_date = ?
            """,
            [target_date],
        ).fetchone()
    return {
        "duckdb_path": str(db_path),
        "table_name": table_name,
        "rows": int(total[0]),
        "stocks": int(total[1]),
        "min_trade_date": str(total[2]),
        "max_trade_date": str(total[3]),
        "bj_rows": int(total[4] or 0),
        "duplicate_rows": int(total[5] or 0),
        "target_date_rows": int(target[0]),
        "target_date_stocks": int(target[1]),
        "target_date_bj_rows": int(target[2] or 0),
        "column_count": len(cols),
        "has_naked_price_columns": [column for column in ("open", "high", "low", "close", "pre_close") if column in cols],
        "has_qfq_price_columns": [column for column in ("open_qfq", "high_qfq", "low_qfq", "close_qfq", "pre_close_qfq") if column in cols],
        "has_legacy_gtja_columns": [column for column in GTJA_ALPHA_COLUMNS if column in cols],
        "has_qfq_gtja_columns": [column for column in qfq_gtja_cols if column in cols],
    }


def _query_label_audit() -> dict:
    db_path = resolve_l3_label_duckdb_path()
    table_name = resolve_l3_label_duckdb_table()
    with duckdb.connect(str(db_path), read_only=True) as conn:
        total = conn.execute(
            f"""
            SELECT
                COUNT(*) AS rows_n,
                COUNT(DISTINCT stock_code) AS stocks_n,
                MIN(trade_date) AS min_d,
                MAX(trade_date) AS max_d,
                SUM(CASE WHEN stock_code LIKE '%.BJ' THEN 1 ELSE 0 END) AS bj_rows,
                COUNT(*) - COUNT(DISTINCT stock_code || '|' || trade_date) AS duplicate_rows
            FROM {table_name}
            """
        ).fetchone()
    return {
        "duckdb_path": str(db_path),
        "table_name": table_name,
        "rows": int(total[0]),
        "stocks": int(total[1]),
        "min_trade_date": str(total[2]),
        "max_trade_date": str(total[3]),
        "bj_rows": int(total[4] or 0),
        "duplicate_rows": int(total[5] or 0),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deliver L3 active DuckDB assets from legacy staging with no-BJ/qfq governance.")
    parser.add_argument("--target-date", required=True)
    parser.add_argument("--read-start", default="20250101")
    parser.add_argument("--workspace-dir", required=True)
    parser.add_argument("--raw-parts-dir", required=True)
    parser.add_argument("--label-parts-dir")
    parser.add_argument(
        "--skip-label-sync",
        action="store_true",
        help="Skip label parquet staging sync and only audit the active DuckDB label table.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--report-md", required=True)
    return parser.parse_args(argv)


def _run_delivery(args: argparse.Namespace) -> None:
    os.environ[ENV_ALLOW_LEGACY_L3_PARQUET_PARTS] = "1"

    workspace_dir = Path(args.workspace_dir)
    raw_parts_dir = Path(args.raw_parts_dir)
    label_parts_dir = Path(args.label_parts_dir) if args.label_parts_dir else None
    if not args.skip_label_sync and label_parts_dir is None:
        raise ValueError("--label-parts-dir is required unless --skip-label-sync is set")
    mini_feature_dir = workspace_dir / "feature_target_parts"
    data_dir = resolve_data_dir()
    db_path = resolve_stock_daily_duckdb_path(data_dir=data_dir, require_exists=True)

    print(
        json.dumps(
            {
                "stage": "raw_target_start",
                "target_date": args.target_date,
                "workers": args.workers,
                "workspace_dir": str(workspace_dir),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    raw_recent, historical_max_trade_date = _compute_recent_raw_frames(
        db_path=db_path,
        raw_parts_dir=raw_parts_dir,
        read_start=args.read_start,
        target_date=args.target_date,
        workers=args.workers,
    )
    if raw_recent.empty:
        raise RuntimeError(f"no recent raw rows computed for {args.target_date}")
    raw_target = raw_recent[raw_recent["trade_date"].astype(str).eq(args.target_date)].copy()
    if raw_target.empty:
        raise RuntimeError(f"no raw target rows computed for {args.target_date}")
    print(
        json.dumps(
            {
                "stage": "raw_target_done",
                "rows": int(raw_target.shape[0]),
                "stocks": int(raw_target["stock_code"].nunique()),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    print(json.dumps({"stage": "gtja_target_start"}, ensure_ascii=False), flush=True)
    gtja_target = _build_gtja_target_frame(
        raw_parts_dir,
        raw_recent,
        read_start=args.read_start,
        target_date=args.target_date,
    )
    print(
        json.dumps(
            {
                "stage": "gtja_target_done",
                "rows": int(gtja_target.shape[0]),
                "stocks": int(gtja_target["stock_code"].nunique()) if not gtja_target.empty else 0,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    print(json.dumps({"stage": "feature_target_build_start"}, ensure_ascii=False), flush=True)
    feature_target = _build_target_feature_frame(raw_parts_dir, raw_target, gtja_target)
    feature_part_path = _write_target_feature_part(feature_target, mini_feature_dir, args.target_date)
    print(
        json.dumps(
            {
                "stage": "feature_target_part_written",
                "path": str(feature_part_path),
                "rows": int(feature_target.shape[0]),
                "stocks": int(feature_target["stock_code"].nunique()),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    print(json.dumps({"stage": "feature_sync_start"}, ensure_ascii=False), flush=True)
    feature_rewrite = {
        "mode": "skipped_full_rewrite_for_target_date_delivery",
        "reason": "active DuckDB feature table already exists; this run writes the rebuilt target-date feature part only",
        "target_date": args.target_date,
    }
    feature_target_sync = sync_feature_parts_target_date_to_duckdb(
        data_dir=None,
        target_date=args.target_date,
        parts_dir=mini_feature_dir,
    )
    print(
        json.dumps(
            {
                "stage": "feature_sync_done",
                "feature_rewrite": feature_rewrite,
                "feature_target_sync": feature_target_sync,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.skip_label_sync:
        label_full_sync = {
            "mode": "skipped",
            "reason": "active labels are audited read-only; no mature future labels are fabricated or backfilled from legacy staging",
        }
        print(
            json.dumps(
                {
                    "stage": "label_sync_skipped",
                    "label_full_sync": label_full_sync,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    else:
        print(json.dumps({"stage": "label_sync_start"}, ensure_ascii=False), flush=True)
        label_full_sync = sync_label_parts_full_to_duckdb(
            data_dir=None,
            parts_dir=label_parts_dir,
        )
        print(
            json.dumps(
                {
                    "stage": "label_sync_done",
                    "label_full_sync": label_full_sync,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    feature_audit = _query_feature_audit(args.target_date)
    label_audit = _query_label_audit()
    payload = {
        "target_date": args.target_date,
        "read_start": args.read_start,
        "workspace_dir": str(workspace_dir),
        "raw_parts_dir": str(raw_parts_dir),
        "label_parts_dir": str(label_parts_dir) if label_parts_dir is not None else None,
        "feature_target_part_path": str(feature_part_path),
        "raw_history_max_trade_date": historical_max_trade_date,
        "legacy_staging_opt_in": {
            "env_name": ENV_ALLOW_LEGACY_L3_PARQUET_PARTS,
            "value": "1",
            "note": "temporary staging opt-in only; not a production rollback",
        },
        "feature_rewrite": feature_rewrite,
        "feature_target_sync": feature_target_sync,
        "label_full_sync": label_full_sync,
        "raw_target_rows": int(raw_target.shape[0]),
        "raw_target_stocks": int(raw_target["stock_code"].nunique()),
        "feature_target_rows": int(feature_target.shape[0]),
        "feature_target_stocks": int(feature_target["stock_code"].nunique()),
        "feature_audit": feature_audit,
        "label_audit": label_audit,
    }
    report_json = Path(args.report_json)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report_md = Path(args.report_md)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {args.target_date} L3 全量交付报告",
        "",
        "## 执行摘要",
        f"- 临时 staging 路径：`{workspace_dir}`",
        f"- legacy staging opt-in：`{ENV_ALLOW_LEGACY_L3_PARQUET_PARTS}=1`，仅用于本轮临时 parquet staging，不是生产链回退。",
        f"- active feature：`{feature_audit['duckdb_path']}::{feature_audit['table_name']}`",
        f"- active label：`{label_audit['duckdb_path']}::{label_audit['table_name']}`",
        "",
        "## Feature 结果",
        f"- 全表：`{feature_audit['rows']}` 行 / `{feature_audit['stocks']}` 只股票 / 日期 `{feature_audit['min_trade_date']}` -> `{feature_audit['max_trade_date']}`",
        f"- 目标日 `{args.target_date}`：`{feature_audit['target_date_rows']}` 行 / `{feature_audit['target_date_stocks']}` 只股票",
        f"- 重复键：`{feature_audit['duplicate_rows']}`",
        f"- no-BJ：全表 `.BJ` 行数 `{feature_audit['bj_rows']}`，目标日 `.BJ` 行数 `{feature_audit['target_date_bj_rows']}`",
        f"- qfq 裸价格列检查：`{feature_audit['has_naked_price_columns']}`",
        f"- qfq 价格列检查：`{feature_audit['has_qfq_price_columns']}`",
        f"- legacy GTJA 裸列检查：`{feature_audit['has_legacy_gtja_columns'][:5]}` ... 共 `{len(feature_audit['has_legacy_gtja_columns'])}` 列",
        f"- qfq GTJA 列检查：`{feature_audit['has_qfq_gtja_columns'][:5]}` ... 共 `{len(feature_audit['has_qfq_gtja_columns'])}` 列",
        "",
        "## Label 结果",
        f"- 全表：`{label_audit['rows']}` 行 / `{label_audit['stocks']}` 只股票 / 日期 `{label_audit['min_trade_date']}` -> `{label_audit['max_trade_date']}`",
        f"- 重复键：`{label_audit['duplicate_rows']}`",
        f"- no-BJ：全表 `.BJ` 行数 `{label_audit['bj_rows']}`",
        f"- label 不前推：active label 最大日期保持 `{label_audit['max_trade_date']}`，未为 `{args.target_date}` 伪造未来标签。",
        "",
        "## Source-Limited / Null 说明",
        f"- 本轮 feature 目标日原始目标行数：`{payload['raw_target_rows']}`，生产目标行数：`{payload['feature_target_rows']}`。",
        "- source-limited 与辅助可空字段继续按既有治理口径保留真实空值，不因 DuckDB 交付而伪造补值。",
        "",
        "## 建议",
        "- 审计通过后可清理本轮 workspace staging；在审计通过前保留作为证据与可重复执行输入。",
    ]
    report_md.write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "stage": "report_written",
                "report_json": str(report_json),
                "report_md": str(report_md),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    workflow_run_id = f"refresh-l3-active-duckdb-full-delivery-{args.target_date}"
    report_dir = Path(args.report_json).resolve().parent
    lease = acquire_active_l3_writer_lease(
        workflow_run_id=workflow_run_id,
        workspace=Path(args.workspace_dir),
        report_dir=report_dir,
        process_role="refresh_l3_active_duckdb_full_delivery",
    )
    release_path = report_dir / f"l3_active_writer_lease_release_{args.target_date}.json"
    try:
        _run_delivery(args)
    finally:
        release = release_active_l3_writer_lease(lease)
        release_path.parent.mkdir(parents=True, exist_ok=True)
        release_path.write_text(json.dumps(release, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
