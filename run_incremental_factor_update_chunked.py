from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from incremental_factor_update_target_date import (
    _part_index,
    _raw_part_codes,
    _target_date_codes,
    process_new_stock_codes,
)
from project_paths import resolve_data_dir
from stock_daily_data_route import resolve_stock_daily_db_path


def parse_args(argv=None):
    data_dir = resolve_data_dir()
    parser = argparse.ArgumentParser(
        description="Run target-date factor incremental update in part chunks to avoid long single-batch hangs."
    )
    parser.add_argument("--target-date", required=True)
    parser.add_argument("--db-path", default=str(resolve_stock_daily_db_path()))
    parser.add_argument("--raw-parts-dir", default=str(data_dir / "raw_factor_by_stock_parts"))
    parser.add_argument("--production-parts-dir", default=str(data_dir / "production_factor_parts"))
    parser.add_argument("--read-start", default="20250101")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=130)
    parser.add_argument("--start-part", type=int)
    parser.add_argument("--end-part", type=int)
    parser.add_argument("--report-dir")
    parser.add_argument("--report-path")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args(argv)


def selected_part_indexes(raw_parts_dir: Path, start_part: int | None, end_part: int | None) -> list[int]:
    indexes = [_part_index(path) for path in sorted(raw_parts_dir.glob("raw_part_*.parquet"))]
    if start_part is not None:
        indexes = [idx for idx in indexes if idx >= start_part]
    if end_part is not None:
        indexes = [idx for idx in indexes if idx <= end_part]
    return indexes


def chunk_ranges(indexes: list[int], chunk_size: int) -> list[tuple[int, int]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not indexes:
        return []
    ranges: list[tuple[int, int]] = []
    for i in range(0, len(indexes), chunk_size):
        chunk = indexes[i : i + chunk_size]
        ranges.append((chunk[0], chunk[-1]))
    return ranges


def should_run_new_stock_fill(start_part: int | None, end_part: int | None) -> bool:
    return start_part is None and end_part is None


def build_chunk_command(
    args: argparse.Namespace,
    chunk_start: int,
    chunk_end: int,
    report_path: Path,
) -> list[str]:
    return [
        args.python_executable,
        "quant/main/incremental_factor_update_target_date.py",
        "--target-date",
        args.target_date,
        "--db-path",
        args.db_path,
        "--raw-parts-dir",
        args.raw_parts_dir,
        "--production-parts-dir",
        args.production_parts_dir,
        "--read-start",
        args.read_start,
        "--start-part",
        str(chunk_start),
        "--end-part",
        str(chunk_end),
        "--workers",
        str(args.workers),
        "--report-path",
        str(report_path),
    ]


def default_report_dir(target_date: str) -> Path:
    return resolve_data_dir() / "reports" / f"incremental_factor_update_{target_date}_chunked"


def main(argv=None):
    args = parse_args(argv)
    raw_parts_dir = Path(args.raw_parts_dir)
    report_dir = Path(args.report_dir) if args.report_dir else default_report_dir(args.target_date)
    report_dir.mkdir(parents=True, exist_ok=True)

    indexes = selected_part_indexes(raw_parts_dir, args.start_part, args.end_part)
    if not indexes:
        raise RuntimeError(f"no raw parts selected: {raw_parts_dir}")
    ranges = chunk_ranges(indexes, args.chunk_size)

    started_at = datetime.now().isoformat(timespec="seconds")
    chunk_reports: list[dict] = []
    failures: list[dict] = []

    for chunk_start, chunk_end in ranges:
        chunk_report_path = report_dir / f"chunk_{chunk_start:04d}_{chunk_end:04d}.json"
        command = build_chunk_command(args, chunk_start, chunk_end, chunk_report_path)
        completed = subprocess.run(command, check=False)
        chunk_item = {
            "chunk_start": chunk_start,
            "chunk_end": chunk_end,
            "report_path": str(chunk_report_path),
            "return_code": int(completed.returncode),
            "command": command,
        }
        chunk_reports.append(chunk_item)
        if completed.returncode != 0:
            failures.append(chunk_item)
            if not args.continue_on_error:
                break

    new_stock_result = {"status": "skipped"}
    if not failures and should_run_new_stock_fill(args.start_part, args.end_part):
        missing_codes = sorted(_target_date_codes(Path(args.db_path), args.target_date) - _raw_part_codes(raw_parts_dir))
        new_stock_result = process_new_stock_codes(
            missing_codes,
            args.db_path,
            args.raw_parts_dir,
            args.production_parts_dir,
            args.target_date,
            args.read_start,
        )

    report = {
        "target_date": args.target_date,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "started_at": started_at,
        "raw_parts_dir": str(raw_parts_dir),
        "production_parts_dir": args.production_parts_dir,
        "db_path": args.db_path,
        "workers": args.workers,
        "chunk_size": args.chunk_size,
        "selected_part_count": len(indexes),
        "selected_part_min": min(indexes),
        "selected_part_max": max(indexes),
        "chunk_count": len(ranges),
        "success_chunk_count": sum(1 for item in chunk_reports if item["return_code"] == 0),
        "failure_chunk_count": len(failures),
        "chunk_reports": chunk_reports,
        "failed_chunks": failures,
        "new_stock_result": new_stock_result,
    }
    report_path = Path(args.report_path) if args.report_path else report_dir / "summary.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report_path": str(report_path), "failure_chunk_count": len(failures)}, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
