from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MAIN_DIR = PROJECT_ROOT / "quant" / "main"
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))


MAINLINE_L2_WRITERS = [
    PROJECT_ROOT / "quant" / "data_file" / "backfill_parts" / "integrate_stock_daily_target_date_from_split_raw_20260617.py",
    PROJECT_ROOT / "quant" / "data_file" / "backfill_parts" / "integrate_stock_daily_target_date_from_split_raw_20260618.py",
    PROJECT_ROOT / "quant" / "data_file" / "backfill_parts" / "integrate_stock_daily_target_date_from_split_raw_20260622.py",
    PROJECT_ROOT / "quant" / "data_file" / "backfill_parts" / "integrate_stock_daily_target_date_from_split_raw_20260623.py",
    PROJECT_ROOT / "quant" / "data_file" / "backfill_parts" / "integrate_stock_daily_target_date_from_split_raw_20260624.py",
    PROJECT_ROOT / "quant" / "data_file" / "backfill_parts" / "integrate_stock_daily_target_date_from_split_raw_20260625.py",
]

LEGACY_OR_BOOTSTRAP = [
    PROJECT_ROOT / "run_0609_dual_signals.py",
    PROJECT_ROOT / "quant" / "main" / "database_module.py",
    PROJECT_ROOT / "quant" / "data_file" / "backfill_parts" / "extract_stock_daily_data_db_20260616.py",
]

DIRECT_WRITE_MARKERS = [
    "insert into stock_daily_data",
    "delete from stock_daily_data",
    "update stock_daily_data",
    "create table stock_daily_data",
    'to_sql("stock_daily_data"',
    "to_sql('stock_daily_data'",
]

SYNC_MARKERS = [
    "sync_stock_daily_trade_range_to_duckdb",
    "l2_duckdb_sync_enabled",
    '"duckdb_sync"',
]


def rel(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scan L2 STOCK_DAILY_DATA write-chain coverage for DuckDB sync.")
    parser.add_argument("--report", required=True)
    args = parser.parse_args(argv)

    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = PROJECT_ROOT / report_path

    entries: list[dict[str, object]] = []
    sqlite_only_bypass_count = 0
    unexpected_direct_writer_count = 0

    for path in MAINLINE_L2_WRITERS:
        text = read_text(path)
        markers_present = all(marker in text for marker in SYNC_MARKERS)
        entry = {
            "path": rel(path),
            "classification": "mainline_dedicated_l2",
            "writes_stock_daily_data": any(marker in text.lower() for marker in DIRECT_WRITE_MARKERS),
            "duckdb_sync_hook_present": markers_present,
            "status": "covered" if markers_present else "sqlite_only_bypass",
        }
        if not markers_present:
            sqlite_only_bypass_count += 1
        entries.append(entry)

    for path in LEGACY_OR_BOOTSTRAP:
        text = read_text(path).lower()
        classification = "legacy_or_mixed_db"
        if path.name == "extract_stock_daily_data_db_20260616.py":
            classification = "bootstrap_full_extract"
        entries.append(
            {
                "path": rel(path),
                "classification": classification,
                "writes_stock_daily_data": any(marker in text for marker in DIRECT_WRITE_MARKERS),
                "duckdb_sync_hook_present": "sync_stock_daily_trade_range_to_duckdb" in text,
                "status": "excluded_from_current_l2_registry_scope",
            }
        )

    scanned_paths = {path.resolve() for path in MAINLINE_L2_WRITERS + LEGACY_OR_BOOTSTRAP}
    for path in list(PROJECT_ROOT.glob("*.py")) + list((PROJECT_ROOT / "quant" / "main").glob("*.py")):
        resolved = path.resolve()
        if resolved in scanned_paths:
            continue
        text = read_text(path).lower()
        if not any(marker in text for marker in DIRECT_WRITE_MARKERS):
            continue
        unexpected_direct_writer_count += 1
        entries.append(
            {
                "path": rel(path),
                "classification": "unexpected_direct_writer",
                "writes_stock_daily_data": True,
                "duckdb_sync_hook_present": "sync_stock_daily_trade_range_to_duckdb" in text,
                "status": "review_required",
            }
        )

    result = {
        "status": "ok" if sqlite_only_bypass_count == 0 and unexpected_direct_writer_count == 0 else "failed",
        "mainline_writer_count": len(MAINLINE_L2_WRITERS),
        "mainline_writer_covered_count": len(MAINLINE_L2_WRITERS) - sqlite_only_bypass_count,
        "sqlite_only_l2_bypass_count": sqlite_only_bypass_count,
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
