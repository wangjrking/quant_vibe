from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MAIN_DIR = PROJECT_ROOT / "quant" / "main"
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))


MAINLINE_L3_WRITERS = [
    PROJECT_ROOT / "quant" / "main" / "incremental_factor_update_target_date.py",
    PROJECT_ROOT / "quant" / "main" / "incremental_prediction_label_update_target_date.py",
    PROJECT_ROOT / "quant" / "main" / "build_production_factor_parts.py",
    PROJECT_ROOT / "quant" / "main" / "build_prediction_label_parts.py",
    PROJECT_ROOT / "quant" / "main" / "build_production_factor_raw_gtja_parts.py",
]

LEGACY_OR_RESEARCH = [
    PROJECT_ROOT / "quant" / "main" / "run_incremental_cdb_update.py",
    PROJECT_ROOT / "quant" / "main" / "rebuild_factor_data_batched.py",
    PROJECT_ROOT / "quant" / "main" / "add_legacy_factor_columns.py",
]

DIRECT_WRITE_MARKERS = [
    ".to_parquet(",
    "parquetwriter(",
]

L3_OUTPUT_MARKERS = [
    "production_factor_part_",
    "prediction_label_part_",
]

SYNC_MARKERS = [
    "l3_duckdb_sync_enabled",
    '"duckdb_sync"',
]


def rel(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scan L3 parquet write-chain coverage for DuckDB sync.")
    parser.add_argument("--report", required=True)
    args = parser.parse_args(argv)

    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = PROJECT_ROOT / report_path

    entries: list[dict[str, object]] = []
    parquet_only_bypass_count = 0
    unexpected_direct_writer_count = 0

    for path in MAINLINE_L3_WRITERS:
        text = read_text(path)
        markers_present = all(marker in text for marker in SYNC_MARKERS)
        entry = {
            "path": rel(path),
            "classification": "mainline_l3_writer",
            "writes_parquet": any(marker in text.lower() for marker in DIRECT_WRITE_MARKERS),
            "duckdb_sync_hook_present": markers_present,
            "status": "covered" if markers_present else "parquet_only_bypass",
        }
        if not markers_present:
            parquet_only_bypass_count += 1
        entries.append(entry)

    for path in LEGACY_OR_RESEARCH:
        text = read_text(path).lower()
        entries.append(
            {
                "path": rel(path),
                "classification": "legacy_or_research",
                "writes_parquet": any(marker in text for marker in DIRECT_WRITE_MARKERS),
                "duckdb_sync_hook_present": "l3_duckdb_sync_enabled" in text,
                "status": "excluded_from_current_l3_registry_scope",
            }
        )

    scanned = {path.resolve() for path in MAINLINE_L3_WRITERS + LEGACY_OR_RESEARCH}
    for path in (PROJECT_ROOT / "quant" / "main").glob("*.py"):
        resolved = path.resolve()
        if resolved in scanned:
            continue
        text = read_text(path).lower()
        if not any(marker in text for marker in L3_OUTPUT_MARKERS):
            continue
        if not any(marker in text for marker in DIRECT_WRITE_MARKERS):
            continue
        unexpected_direct_writer_count += 1
        entries.append(
            {
                "path": rel(path),
                "classification": "unexpected_direct_writer",
                "writes_parquet": True,
                "duckdb_sync_hook_present": "l3_duckdb_sync_enabled" in text,
                "status": "review_required",
            }
        )

    result = {
        "status": "ok" if parquet_only_bypass_count == 0 and unexpected_direct_writer_count == 0 else "failed",
        "mainline_writer_count": len(MAINLINE_L3_WRITERS),
        "mainline_writer_covered_count": len(MAINLINE_L3_WRITERS) - parquet_only_bypass_count,
        "parquet_only_l3_bypass_count": parquet_only_bypass_count,
        "unexpected_direct_writer_count": unexpected_direct_writer_count,
        "entries": entries,
        "boundary": {
            "reads_source_files_only": True,
            "edits_production_registry": False,
            "switches_mainline_routes": False,
            "runs_business_pipeline": False,
        },
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(str(report_path))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
