from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from build_production_factor_parts import KEY_COLUMNS, production_raw_columns
from gtja_alpha_workflow import GTJA_ALPHA_COLUMNS, compute_gtja_alpha_from_raw_factor, required_gtja_raw_columns
from l3_duckdb_sync import l3_duckdb_sync_enabled, sync_feature_parts_full_to_duckdb


def build_production_raw_gtja_part(
    raw_part_path: Path,
    output_dir: Path,
    raw_factor_columns: list[str],
    gtja_input_columns: list[str],
    *,
    resume: bool = True,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / raw_part_path.name.replace("raw_part_", "production_factor_part_")
    if resume and output_path.exists():
        print(f"production_raw_gtja_skip path={output_path}", flush=True)
        return output_path

    raw = pd.read_parquet(raw_part_path, columns=raw_factor_columns)
    raw["trade_date"] = raw["trade_date"].astype(str)
    gtja_input = pd.read_parquet(raw_part_path, columns=gtja_input_columns)
    gtja_input["trade_date"] = gtja_input["trade_date"].astype(str)
    gtja = compute_gtja_alpha_from_raw_factor(
        gtja_input,
        encode=False,
        drop_ts=True,
        cross_sectional_rank_mode="identity",
    )
    keep_gtja = [*KEY_COLUMNS, *[column for column in GTJA_ALPHA_COLUMNS if column in gtja.columns]]
    gtja = gtja[keep_gtja]
    merged = raw.merge(gtja, on=KEY_COLUMNS, how="left", validate="one_to_one")
    merged.replace([np.inf, -np.inf], np.nan, inplace=True)
    merged.sort_values(KEY_COLUMNS, inplace=True)
    merged.to_parquet(output_path, index=False)
    print(
        f"production_raw_gtja_done rows={merged.shape[0]} cols={merged.shape[1]} "
        f"raw_part={raw_part_path.name} path={output_path}",
        flush=True,
    )
    return output_path


def write_schema_report(raw_factor_columns: list[str], output_path: Path) -> None:
    payload = {
        "key_columns": KEY_COLUMNS,
        "raw_factor_columns": raw_factor_columns,
        "raw_factor_count": len(raw_factor_columns) - len(KEY_COLUMNS),
        "gtja_alpha_columns": GTJA_ALPHA_COLUMNS,
        "gtja_alpha_count": len(GTJA_ALPHA_COLUMNS),
        "gtja_rank_mode": "identity",
        "note": "GTJA RANK() is intentionally not applied. RANK(x) inputs are used as raw numeric values, then merged into the production factor table.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build production factor parts by merging raw usable factors with GTJA formulas without RANK.")
    parser.add_argument("--raw-parts-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-part", type=int)
    parser.add_argument("--end-part", type=int)
    parser.add_argument("--schema-report")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    args = parse_args(argv)
    raw_parts_dir = Path(args.raw_parts_dir)
    output_dir = Path(args.output_dir)
    first_raw_part = next(iter(sorted(raw_parts_dir.glob("raw_part_*.parquet"))), None)
    if first_raw_part is None:
        raise RuntimeError(f"no raw parts found: {raw_parts_dir}")

    raw_schema = pq.ParquetFile(first_raw_part).schema.names
    raw_factor_columns = production_raw_columns(raw_schema)
    gtja_input_columns = [column for column in required_gtja_raw_columns() if column in raw_schema]
    missing_gtja = sorted(set(required_gtja_raw_columns()) - set(gtja_input_columns))
    if missing_gtja:
        raise RuntimeError(f"raw parts missing GTJA input columns: {missing_gtja}")
    if args.schema_report:
        write_schema_report(raw_factor_columns, Path(args.schema_report))

    for raw_part_path in sorted(raw_parts_dir.glob("raw_part_*.parquet")):
        part_index = int(raw_part_path.stem.rsplit("_", 1)[-1])
        if args.start_part is not None and part_index < args.start_part:
            continue
        if args.end_part is not None and part_index > args.end_part:
            continue
        build_production_raw_gtja_part(
            raw_part_path,
            output_dir,
            raw_factor_columns,
            gtja_input_columns,
            resume=not args.no_resume,
        )
    if l3_duckdb_sync_enabled():
        sync_result = sync_feature_parts_full_to_duckdb(
            data_dir=None,
            parts_dir=output_dir,
        )
        print(json.dumps({"duckdb_sync": sync_result}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
