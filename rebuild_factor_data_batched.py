from __future__ import annotations

import argparse
import gc
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from data_process_module import get_factor_data
from gtja_alpha_workflow import read_gtja_audit_frame, rebuild_full_market_gtja, write_gtja_rank_audit
from stock_daily_data_route import connect_stock_daily_readonly, resolve_stock_daily_db_path


TEXT_COLUMNS = {"stock_code", "trade_date", "name", "industry", "act_ent_type"}


def _stock_codes(db_path: Path) -> list[str]:
    with closing(connect_stock_daily_readonly(db_path=db_path)) as conn:
        rows = conn.execute(
            'SELECT DISTINCT stock_code FROM STOCK_DAILY_DATA ORDER BY stock_code'
        ).fetchall()
    return [row[0] for row in rows]


def _load_batch(db_path: Path, codes: list[str]) -> pd.DataFrame:
    placeholders = ",".join(["?"] * len(codes))
    with closing(connect_stock_daily_readonly(db_path=db_path)) as conn:
        frame = pd.read_sql(
            f"""
            SELECT *
            FROM STOCK_DAILY_DATA
            WHERE stock_code IN ({placeholders})
            ORDER BY stock_code, trade_date
            """,
            conn,
            params=codes,
        )
    frame.columns = frame.columns.str.lower()
    return frame


def _normalize_types(frame: pd.DataFrame) -> pd.DataFrame:
    frame["trade_date"] = frame["trade_date"].astype(str)
    if "st_type" in frame.columns:
        st_as_text = frame["st_type"].astype("string")
        frame["st_type"] = np.where(st_as_text.eq("ST").fillna(False), 1.0, 0.0)
    for col in frame.columns:
        if col in TEXT_COLUMNS or col == "st_type":
            continue
        if pd.api.types.is_object_dtype(frame[col]) or pd.api.types.is_string_dtype(frame[col]):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame.replace([np.inf, -np.inf], np.nan, inplace=True)
    return frame


def _chunks(values: list[str], size: int):
    for start in range(0, len(values), size):
        yield start // size, values[start : start + size]


def rebuild(
    data_dir: Path,
    batch_size: int,
    resume: bool,
    end_date: str | None = None,
    start_batch: int | None = None,
    end_batch: int | None = None,
    no_merge: bool = False,
    global_gtja: bool = True,
    gtja_audit_path: Path | None = None,
) -> Path:
    db_path = resolve_stock_daily_db_path(data_dir=data_dir)
    output_dir = data_dir / "factor_rebuild_parts"
    output_dir.mkdir(parents=True, exist_ok=True)
    codes = _stock_codes(db_path)
    print(f"rebuild_start stocks={len(codes)} batch_size={batch_size}", flush=True)

    for batch_idx, batch_codes in _chunks(codes, batch_size):
        if start_batch is not None and batch_idx < start_batch:
            continue
        if end_batch is not None and batch_idx > end_batch:
            continue
        part_path = output_dir / f"part_{batch_idx:04d}.parquet"
        if resume and part_path.exists():
            print(f"batch_skip index={batch_idx} path={part_path}", flush=True)
            continue
        print(f"batch_start index={batch_idx} stocks={len(batch_codes)} first={batch_codes[0]} last={batch_codes[-1]}", flush=True)
        integ = _load_batch(db_path, batch_codes)
        factor = get_factor_data(integ)
        factor = _normalize_types(factor)
        factor.to_parquet(part_path, index=False)
        print(f"batch_done index={batch_idx} rows={factor.shape[0]} path={part_path}", flush=True)
        del integ, factor
        gc.collect()

    final_path = data_dir / "stock_factor_data.parquet"
    if no_merge:
        print(f"no_merge output_dir={output_dir}", flush=True)
        return final_path
    part_paths = sorted(output_dir.glob("part_*.parquet"))
    print(f"merge_start parts={len(part_paths)}", flush=True)
    temp_path = final_path.with_suffix(".tmp.parquet")
    if temp_path.exists():
        temp_path.unlink()
    writer = None
    total_rows = 0
    max_date = ""
    try:
        for path in part_paths:
            frame = _normalize_types(pd.read_parquet(path))
            if end_date and "trade_date" in frame.columns:
                frame = frame[frame["trade_date"].astype(str) <= end_date].copy()
            if "trade_date" in frame.columns:
                max_date = max(max_date, str(frame["trade_date"].max()))
            table = pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(None)
            if writer is None:
                writer = pq.ParquetWriter(temp_path, table.schema)
            elif table.schema != writer.schema:
                table = table.cast(writer.schema)
            writer.write_table(table)
            total_rows += frame.shape[0]
            print(f"merge_part path={path.name} rows={frame.shape[0]} total={total_rows}", flush=True)
            del frame, table
            gc.collect()
    finally:
        if writer is not None:
            writer.close()
    temp_path.replace(final_path)
    print(f"merge_done rows={total_rows} max={max_date} output={final_path}", flush=True)

    if global_gtja:
        print("global_gtja_start mode=full_market_recompute", flush=True)
        final_path = rebuild_full_market_gtja(data_dir, final_path, end_date=end_date, source_parquet=final_path)
        print(f"global_gtja_done output={final_path}", flush=True)
    else:
        print("global_gtja_skipped warning=gtja_alpha_rank_may_be_batch_scoped", flush=True)

    audit_path = gtja_audit_path or data_dir / "reports" / "gtja_alpha_rank_audit.json"
    audit_frame = read_gtja_audit_frame(final_path)
    audit = write_gtja_rank_audit(audit_frame, audit_path)
    print(
        f"global_gtja_audit passed={audit['passed']} issue_count={audit.get('issue_count', 0)} path={audit_path}",
        flush=True,
    )
    return final_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Rebuild stock_factor_data.parquet in resumable stock batches.")
    parser.add_argument("--data-dir", default="data_file")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--end-date", help="Optional YYYYMMDD cutoff applied when merging factor parts.")
    parser.add_argument("--start-batch", type=int)
    parser.add_argument("--end-batch", type=int)
    parser.add_argument("--no-merge", action="store_true")
    parser.add_argument(
        "--skip-global-gtja",
        action="store_true",
        help="Skip full-market GTJA recompute. Unsafe for production because cross-sectional ranks stay batch-scoped.",
    )
    parser.add_argument("--gtja-audit-path")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rebuild(
        Path(args.data_dir),
        batch_size=args.batch_size,
        resume=args.resume,
        end_date=args.end_date,
        start_batch=args.start_batch,
        end_batch=args.end_batch,
        no_merge=args.no_merge,
        global_gtja=not args.skip_global_gtja,
        gtja_audit_path=Path(args.gtja_audit_path) if args.gtja_audit_path else None,
    )


if __name__ == "__main__":
    main()
