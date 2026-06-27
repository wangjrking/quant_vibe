from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from gtja_alpha_workflow import GTJA_ALPHA_COLUMNS
from project_paths import resolve_data_dir


KEY_COLUMNS = ["trade_date", "stock_code"]
INDUSTRY_ENCODE_MAPPING_NAME = "production_factor_industry_encode_mapping.json"

BASE_NON_FACTOR_COLUMNS = {
    "name",
    "symbol",
    "area",
    "cnspell",
    "market",
    "list_date",
    "season_ann_date",
    "season_end_date",
    "year_ann_date",
    "year_end_date",
    "reason",
    "st_type_name",
    "change",
    "pct_chg",
}

FUTURE_PREFIXES = (
    "post",
    "open2_",
    "open3_",
    "open4_",
    "open5_",
    "open6_",
    "open12_",
    "open22_",
)

FUTURE_SUFFIXES = ("_yield_rate", "_tag")

EXPLICIT_FUTURE_COLUMNS = {
    "tag",
    "yield_rate",
    "close_yield_rate",
    "open_yield_rate",
    "log_yield_rate",
    "next_open_yield_rate",
    "close_open_yield_rate",
    "close_low_yield_rate",
    "low_close_yield_rate",
    "index_2000_post10_yield_rate",
    "adjust_10d_yield_rate",
}

SOURCE_LIMITED_PREFIXES = (
    "cost_",
    "std_cost_",
    "up_stat",
)

SOURCE_LIMITED_COLUMNS = {
    # cyq_perf starts later than the full 2010 history.
    "his_low",
    "his_high",
    "weight_avg",
    "winner_rate",
    "std_his_low",
    "std_his_high",
    "std_weight_avg",
    # limit_list_d / limit_list_data starts later than the full 2010 history.
    "fd_amount",
    "fd_amount_rate",
    "fd_vol_rate",
    "first_time",
    "first_time_int",
    "last_time",
    "last_time_int",
    "limit_amount",
    "limit_times",
    "open_times",
    # stock_st starts later than the full 2010 history.
    "st_code",
    "st_type",
    "st_type_name",
}

NON_DEFAULT_AUXILIARY_PREFIXES = (
    "ths_",
    "dc_",
    "top_list",
)

NON_DEFAULT_AUXILIARY_COLUMNS = {
    "ths_hot",
    "ths_rank",
    "dc_hot",
    "dc_rank",
    "top_list",
}


def is_future_or_label_column(column: str) -> bool:
    if column in EXPLICIT_FUTURE_COLUMNS:
        return True
    if column.startswith(FUTURE_PREFIXES):
        return True
    if column.endswith(FUTURE_SUFFIXES):
        return True
    return False


def is_source_limited_column(column: str) -> bool:
    if column in SOURCE_LIMITED_COLUMNS:
        return True
    if column.startswith(SOURCE_LIMITED_PREFIXES):
        return True
    return False


def is_non_default_auxiliary_column(column: str) -> bool:
    if column in NON_DEFAULT_AUXILIARY_COLUMNS:
        return True
    if column.startswith(NON_DEFAULT_AUXILIARY_PREFIXES):
        return True
    return False


def is_raw_factor_column(column: str) -> bool:
    if column in KEY_COLUMNS:
        return True
    if column.startswith("_ts_"):
        return False
    if column in BASE_NON_FACTOR_COLUMNS:
        return False
    if is_future_or_label_column(column):
        return False
    if is_source_limited_column(column):
        return False
    if is_non_default_auxiliary_column(column):
        return False
    return True


def production_raw_columns(raw_columns: list[str]) -> list[str]:
    return [column for column in raw_columns if is_raw_factor_column(column)]


def default_industry_encode_mapping_path(data_dir: str | Path | None = None) -> Path:
    return resolve_data_dir(data_dir) / "runtime" / INDUSTRY_ENCODE_MAPPING_NAME


def load_industry_encode_mapping(mapping_path: str | Path | None = None) -> dict[str, int]:
    path = Path(mapping_path) if mapping_path else default_industry_encode_mapping_path()
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(key): int(value) for key, value in payload.items()}


def save_industry_encode_mapping(mapping: dict[str, int], mapping_path: str | Path | None = None) -> Path:
    path = Path(mapping_path) if mapping_path else default_industry_encode_mapping_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = {key: mapping[key] for key in sorted(mapping, key=lambda item: mapping[item])}
    path.write_text(json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def normalize_industry_values(industry_values) -> list[str]:
    normalized: list[str] = []
    for value in industry_values:
        if pd.isna(value):
            normalized.append("")
        else:
            normalized.append(str(value))
    return normalized


def extend_industry_encode_mapping(mapping: dict[str, int], industry_values) -> dict[str, int]:
    updated = dict(mapping)
    missing = sorted({value for value in normalize_industry_values(industry_values) if value not in updated})
    next_code = max(updated.values(), default=-1) + 1
    for offset, value in enumerate(missing):
        updated[value] = next_code + offset
    return updated


def sync_industry_encode_mapping_from_raw_parts(
    raw_parts_dir: Path,
    mapping_path: str | Path | None = None,
) -> dict[str, int]:
    mapping = load_industry_encode_mapping(mapping_path)
    industries: set[str] = set()
    for part_path in sorted(raw_parts_dir.glob("raw_part_*.parquet")):
        schema = pq.ParquetFile(part_path).schema.names
        if "industry" not in schema:
            continue
        table = pq.read_table(part_path, columns=["industry"])
        industries.update(normalize_industry_values(table.column("industry").to_pylist()))
    mapping = extend_industry_encode_mapping(mapping, industries)
    save_industry_encode_mapping(mapping, mapping_path)
    return mapping


def apply_industry_encode(frame: pd.DataFrame, mapping: dict[str, int]) -> pd.DataFrame:
    if "industry" not in frame.columns:
        return frame
    enriched = frame.copy()
    enriched["industry"] = normalize_industry_values(enriched["industry"])
    enriched["industry_encode"] = enriched["industry"].map(mapping)
    return enriched


def read_raw_window(raw_parts_dir: Path, read_start: str, read_end: str, columns: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for part_path in sorted(raw_parts_dir.glob("raw_part_*.parquet")):
        frame = pd.read_parquet(
            part_path,
            columns=columns,
            filters=[("trade_date", ">=", read_start), ("trade_date", "<=", read_end)],
        )
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=columns)
    frame = pd.concat(frames, ignore_index=True)
    frame["trade_date"] = frame["trade_date"].astype(str)
    return frame


def read_gtja_part(gtja_part_path: Path) -> pd.DataFrame:
    gtja = pd.read_parquet(gtja_part_path)
    gtja["trade_date"] = gtja["trade_date"].astype(str)
    keep = [*KEY_COLUMNS, *[column for column in GTJA_ALPHA_COLUMNS if column in gtja.columns]]
    return gtja[keep]


def build_production_part(
    raw_parts_dir: Path,
    gtja_part_path: Path,
    output_dir: Path,
    raw_columns: list[str],
    industry_mapping: dict[str, int],
    *,
    resume: bool = True,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / gtja_part_path.name.replace("factor_standard_part_", "production_factor_part_")
    if resume and output_path.exists():
        print(f"production_part_skip path={output_path}", flush=True)
        return output_path

    gtja = read_gtja_part(gtja_part_path)
    dates = sorted(gtja["trade_date"].unique().tolist())
    if not dates:
        raise RuntimeError(f"empty gtja part: {gtja_part_path}")
    raw = read_raw_window(raw_parts_dir, dates[0], dates[-1], raw_columns)
    if raw.empty:
        raise RuntimeError(f"empty raw window for gtja part: {gtja_part_path}")

    merged = raw.merge(gtja, on=KEY_COLUMNS, how="left", validate="one_to_one")
    merged = apply_industry_encode(merged, industry_mapping)
    merged.replace([np.inf, -np.inf], np.nan, inplace=True)
    merged.sort_values(KEY_COLUMNS, inplace=True)
    merged.to_parquet(output_path, index=False)
    print(
        f"production_part_done rows={merged.shape[0]} cols={merged.shape[1]} "
        f"gtja_part={gtja_part_path.name} path={output_path}",
        flush=True,
    )
    return output_path


def write_schema_report(raw_columns: list[str], output_path: Path) -> None:
    payload = {
        "key_columns": KEY_COLUMNS,
        "raw_factor_columns": raw_columns,
        "raw_factor_count": len(raw_columns) - len(KEY_COLUMNS),
        "gtja_alpha_columns": GTJA_ALPHA_COLUMNS,
        "gtja_alpha_count": len(GTJA_ALPHA_COLUMNS),
        "industry_encode_mapping_path": str(default_industry_encode_mapping_path()),
        "excluded_rules": {
            "base_non_factor_columns": sorted(BASE_NON_FACTOR_COLUMNS),
            "future_prefixes": list(FUTURE_PREFIXES),
            "future_suffixes": list(FUTURE_SUFFIXES),
            "explicit_future_columns": sorted(EXPLICIT_FUTURE_COLUMNS),
            "source_limited_columns": sorted(SOURCE_LIMITED_COLUMNS),
            "source_limited_prefixes": list(SOURCE_LIMITED_PREFIXES),
            "non_default_auxiliary_columns": sorted(NON_DEFAULT_AUXILIARY_COLUMNS),
            "non_default_auxiliary_prefixes": list(NON_DEFAULT_AUXILIARY_PREFIXES),
            "ts_intermediate_prefix": "_ts_",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build production factor parts from raw factors plus GTJA full-market factors.")
    parser.add_argument("--raw-parts-dir", required=True)
    parser.add_argument("--gtja-parts-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-part", type=int)
    parser.add_argument("--end-part", type=int)
    parser.add_argument("--schema-report")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    raw_parts_dir = Path(args.raw_parts_dir)
    gtja_parts_dir = Path(args.gtja_parts_dir)
    output_dir = Path(args.output_dir)
    first_raw_part = next(iter(sorted(raw_parts_dir.glob("raw_part_*.parquet"))), None)
    if first_raw_part is None:
        raise RuntimeError(f"no raw parts found: {raw_parts_dir}")

    raw_columns = production_raw_columns(pq.ParquetFile(first_raw_part).schema.names)
    industry_mapping = sync_industry_encode_mapping_from_raw_parts(raw_parts_dir)
    if args.schema_report:
        write_schema_report(raw_columns, Path(args.schema_report))

    for part_path in sorted(gtja_parts_dir.glob("factor_standard_part_*.parquet")):
        part_index = int(part_path.stem.rsplit("_", 1)[-1])
        if args.start_part is not None and part_index < args.start_part:
            continue
        if args.end_part is not None and part_index > args.end_part:
            continue
        build_production_part(
            raw_parts_dir,
            part_path,
            output_dir,
            raw_columns,
            industry_mapping,
            resume=not args.no_resume,
        )


if __name__ == "__main__":
    main()
