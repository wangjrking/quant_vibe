from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from build_prediction_label_parts import (
    KEY_COLUMNS,
    add_executable_labels,
    available_labels,
    required_raw_columns,
    sanitize_existing_labels,
)


def _part_index(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[-1])


def _align_to_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    aligned = frame.copy()
    for column in columns:
        if column not in aligned.columns:
            aligned[column] = np.nan
    return aligned[columns]


def build_target_label_frame(
    raw_part_path: Path,
    target_date: str,
    labels: list[str],
    raw_columns: list[str],
) -> pd.DataFrame:
    read_columns = required_raw_columns(raw_columns, labels)
    frame = pd.read_parquet(raw_part_path, columns=read_columns, filters=[("trade_date", "=", target_date)])
    if frame.empty:
        return pd.DataFrame(columns=[*KEY_COLUMNS, *labels])
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame = add_executable_labels(frame, labels)
    frame = sanitize_existing_labels(frame, labels)
    keep = [*KEY_COLUMNS, *labels]
    frame = frame[[column for column in keep if column in frame.columns]]
    frame.replace([np.inf, -np.inf], np.nan, inplace=True)
    frame.sort_values(KEY_COLUMNS, inplace=True)
    return frame


def replace_target_rows(part_path: Path, target_rows: pd.DataFrame, target_date: str) -> tuple[int, int]:
    if part_path.exists():
        existing = pd.read_parquet(part_path)
        existing["trade_date"] = existing["trade_date"].astype(str)
    else:
        existing = pd.DataFrame(columns=target_rows.columns.tolist())
    old_target_rows = int(existing["trade_date"].eq(target_date).sum()) if "trade_date" in existing.columns else 0
    keep = existing[~existing["trade_date"].eq(target_date)].copy() if "trade_date" in existing.columns else existing
    target = _align_to_columns(target_rows, list(existing.columns)) if not existing.empty else target_rows.copy()
    merged = pd.concat([keep, target], ignore_index=True)
    merged.sort_values(KEY_COLUMNS, inplace=True)
    merged.replace([np.inf, -np.inf], np.nan, inplace=True)
    temp_path = part_path.with_name(f"{part_path.name}.tmp_{target_date}")
    merged.to_parquet(temp_path, index=False)
    temp_path.replace(part_path)
    return old_target_rows, int(target.shape[0])


def audit_label_parts(output_dir: Path, target_date: str, labels: list[str]) -> dict:
    part_paths = sorted(output_dir.glob("prediction_label_part_*.parquet"))
    rows_target_date = 0
    files_with_target_date = 0
    nonnull_counts = {label: 0 for label in labels}
    for path in part_paths:
        table = pq.read_table(path, filters=[("trade_date", "=", target_date)])
        if table.num_rows == 0:
            continue
        files_with_target_date += 1
        rows_target_date += table.num_rows
        frame = table.to_pandas()
        for label in labels:
            if label in frame.columns:
                nonnull_counts[label] += int(pd.to_numeric(frame[label], errors="coerce").notna().sum())
    return {
        "part_count": len(part_paths),
        "files_with_target_date": files_with_target_date,
        "rows_target_date": rows_target_date,
        "nonnull_counts": nonnull_counts,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Incrementally update prediction label parts for one target date.")
    parser.add_argument("--raw-parts-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-date", required=True)
    parser.add_argument("--labels", nargs="*", help="Optional explicit label list.")
    parser.add_argument("--report-path")
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

    results: list[dict] = []
    for raw_part_path in sorted(raw_parts_dir.glob("raw_part_*.parquet")):
        part_index = _part_index(raw_part_path)
        target_rows = build_target_label_frame(raw_part_path, args.target_date, labels, raw_columns)
        output_path = output_dir / raw_part_path.name.replace("raw_part_", "prediction_label_part_")
        old_rows, new_rows = replace_target_rows(output_path, target_rows, args.target_date)
        result = {
            "part_index": part_index,
            "old_rows": old_rows,
            "new_rows": new_rows,
        }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    audit = audit_label_parts(output_dir, args.target_date, labels)
    payload = {
        "target_date": args.target_date,
        "raw_parts_dir": str(raw_parts_dir),
        "output_dir": str(output_dir),
        "labels": labels,
        "updated_part_count": len(results),
        "results": results,
        "audit": audit,
    }
    if args.report_path:
        report_path = Path(args.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
