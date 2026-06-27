from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


KEY_COLUMNS = ["trade_date", "stock_code"]

DEFAULT_EXISTING_LABELS = [
    "tag",
    "5d_tag",
    "2d_tag",
    "yield_rate",
    "close_yield_rate",
    "open_yield_rate",
    "open2_yield_rate",
    "open3_yield_rate",
    "open4_yield_rate",
    "open5_yield_rate",
    "open6_yield_rate",
    "open12_yield_rate",
    "open22_yield_rate",
    "5d_yield_rate",
    "6d_yield_rate",
    "10d_yield_rate",
    "15d_yield_rate",
    "22d_yield_rate",
    "2d_yield_rate",
    "log_yield_rate",
    "log_2d_yield_rate",
    "next_open_yield_rate",
    "close_open_yield_rate",
    "close_low_yield_rate",
    "low_close_yield_rate",
    "post5_most_high_yield_rate",
    "index_2000_post10_yield_rate",
    "adjust_10d_yield_rate",
]

EXECUTABLE_LABEL_SPECS = {
    "executable_1d_open_return": "post2_open",
    "executable_2d_open_return": "post2_open",
    "executable_3d_open_return": "post4_open",
    "executable_5d_open_return": "post6_open",
    "executable_10d_open_return": "post12_open",
}

DEFAULT_COMPUTED_LABELS = list(EXECUTABLE_LABEL_SPECS)

TAG_LABEL_DEPENDENCIES = {
    "tag": "yield_rate",
    "2d_tag": "2d_yield_rate",
    "5d_tag": "5d_yield_rate",
}

LABEL_NULL_DEPENDENCIES = {
    "low_close_yield_rate": "post_low",
}

BUY_COMMISSION_RATE = 0.0003
SELL_COMMISSION_RATE = 0.0003
SELL_TAX_RATE = 0.0005
SLIPPAGE_RATE = 0.001


def available_labels(raw_columns: list[str], requested: list[str] | None = None) -> list[str]:
    labels = requested or [*DEFAULT_EXISTING_LABELS, *DEFAULT_COMPUTED_LABELS]
    existing = set(raw_columns)
    result: list[str] = []
    for label in labels:
        if label in EXECUTABLE_LABEL_SPECS:
            if "post_open" in existing and EXECUTABLE_LABEL_SPECS[label] in existing:
                result.append(label)
        elif label in existing:
            result.append(label)
    return result


def required_raw_columns(raw_columns: list[str], labels: list[str]) -> list[str]:
    existing = set(raw_columns)
    columns = [column for column in KEY_COLUMNS if column in existing]
    for label in labels:
        if label in EXECUTABLE_LABEL_SPECS:
            for column in ("post_open", EXECUTABLE_LABEL_SPECS[label]):
                if column in existing and column not in columns:
                    columns.append(column)
        elif label in TAG_LABEL_DEPENDENCIES:
            dependency = TAG_LABEL_DEPENDENCIES[label]
            for column in (label, dependency):
                if column in existing and column not in columns:
                    columns.append(column)
        elif label in LABEL_NULL_DEPENDENCIES:
            dependency = LABEL_NULL_DEPENDENCIES[label]
            for column in (label, dependency):
                if column in existing and column not in columns:
                    columns.append(column)
        elif label in existing and label not in columns:
            columns.append(label)
    missing_keys = [column for column in KEY_COLUMNS if column not in columns]
    if missing_keys:
        raise ValueError(f"raw parts missing key columns: {missing_keys}")
    return columns


def add_executable_labels(frame: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    for label in labels:
        sell_column = EXECUTABLE_LABEL_SPECS.get(label)
        if sell_column is None:
            continue
        buy = pd.to_numeric(frame["post_open"], errors="coerce")
        sell = pd.to_numeric(frame[sell_column], errors="coerce")
        entry_cash = buy * (1.0 + BUY_COMMISSION_RATE + SLIPPAGE_RATE)
        exit_cash = sell * (1.0 - SELL_COMMISSION_RATE - SELL_TAX_RATE - SLIPPAGE_RATE)
        frame[label] = exit_cash / entry_cash - 1.0
    return frame


def sanitize_existing_labels(frame: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    for label, dependency in TAG_LABEL_DEPENDENCIES.items():
        if label not in labels or dependency not in frame.columns:
            continue
        returns = pd.to_numeric(frame[dependency], errors="coerce")
        sanitized = pd.Series(np.nan, index=frame.index, dtype="float64")
        valid = returns.notna()
        sanitized.loc[valid] = (returns.loc[valid] > 0).astype("float64")
        frame[label] = sanitized
    for label, dependency in LABEL_NULL_DEPENDENCIES.items():
        if label not in labels or dependency not in frame.columns:
            continue
        values = pd.to_numeric(frame[label], errors="coerce")
        valid = pd.to_numeric(frame[dependency], errors="coerce").notna()
        frame[label] = values.where(valid, np.nan)
    return frame


def build_label_part(
    raw_part_path: Path,
    output_dir: Path,
    labels: list[str],
    raw_columns: list[str],
    *,
    resume: bool = True,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / raw_part_path.name.replace("raw_part_", "prediction_label_part_")
    if resume and output_path.exists():
        print(f"label_part_skip path={output_path}", flush=True)
        return output_path

    read_columns = required_raw_columns(raw_columns, labels)
    frame = pd.read_parquet(raw_part_path, columns=read_columns)
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame = add_executable_labels(frame, labels)
    frame = sanitize_existing_labels(frame, labels)
    keep = [*KEY_COLUMNS, *labels]
    frame = frame[[column for column in keep if column in frame.columns]]
    frame.replace([np.inf, -np.inf], np.nan, inplace=True)
    frame.sort_values(KEY_COLUMNS, inplace=True)
    frame.to_parquet(output_path, index=False)
    print(
        f"label_part_done rows={frame.shape[0]} cols={frame.shape[1]} "
        f"raw_part={raw_part_path.name} path={output_path}",
        flush=True,
    )
    return output_path


def write_schema_report(labels: list[str], output_path: Path) -> None:
    payload = {
        "key_columns": KEY_COLUMNS,
        "existing_label_columns": [label for label in labels if label not in EXECUTABLE_LABEL_SPECS],
        "computed_label_columns": [label for label in labels if label in EXECUTABLE_LABEL_SPECS],
        "computed_label_formula": {
            "entry": "post_open * (1 + buy_commission + slippage)",
            "exit": "exit_open * (1 - sell_commission - sell_tax - slippage)",
            "return": "exit / entry - 1",
            "exit_open_by_label": EXECUTABLE_LABEL_SPECS,
            "buy_commission": BUY_COMMISSION_RATE,
            "sell_commission": SELL_COMMISSION_RATE,
            "sell_tax": SELL_TAX_RATE,
            "slippage": SLIPPAGE_RATE,
        },
        "note": "Top-quantile labels are intentionally excluded here because they require a full-market per-date ranking pass.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build prediction label parts for model training.")
    parser.add_argument("--raw-parts-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-part", type=int)
    parser.add_argument("--end-part", type=int)
    parser.add_argument("--labels", nargs="*", help="Optional explicit label list.")
    parser.add_argument("--schema-report")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    raw_parts_dir = Path(args.raw_parts_dir)
    output_dir = Path(args.output_dir)
    first_raw_part = next(iter(sorted(raw_parts_dir.glob("raw_part_*.parquet"))), None)
    if first_raw_part is None:
        raise RuntimeError(f"no raw parts found: {raw_parts_dir}")

    raw_columns = pq.ParquetFile(first_raw_part).schema.names
    labels = available_labels(raw_columns, args.labels)
    if args.labels and len(labels) != len(args.labels):
        missing = sorted(set(args.labels) - set(labels))
        raise ValueError(f"requested labels are unavailable or missing required raw columns: {missing}")
    if args.schema_report:
        write_schema_report(labels, Path(args.schema_report))

    for raw_part_path in sorted(raw_parts_dir.glob("raw_part_*.parquet")):
        part_index = int(raw_part_path.stem.rsplit("_", 1)[-1])
        if args.start_part is not None and part_index < args.start_part:
            continue
        if args.end_part is not None and part_index > args.end_part:
            continue
        build_label_part(
            raw_part_path,
            output_dir,
            labels,
            raw_columns,
            resume=not args.no_resume,
        )


if __name__ == "__main__":
    main()
