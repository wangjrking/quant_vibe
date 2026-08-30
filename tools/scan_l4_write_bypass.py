from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MAIN_DIR = PROJECT_ROOT / "quant" / "main"
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))


MAINLINE_L4_WRITERS = [
    PROJECT_ROOT / "quant" / "main" / "ai_module.py",
    PROJECT_ROOT / "quant" / "main" / "run_pdb_update.py",
    PROJECT_ROOT / "quant" / "main" / "run_light_pdb_update.py",
    PROJECT_ROOT / "quant" / "main" / "run_mlp_pdb_update.py",
    PROJECT_ROOT / "quant" / "main" / "predict_saved_models_standard_chain.py",
    PROJECT_ROOT / "quant" / "main" / "refit_fold09_predict_20260616_production_chain.py",
]

LEGACY_OR_RESEARCH = [
    PROJECT_ROOT / "quant" / "main" / "refit_fold09_predict_20260616.py",
]

EXCLUDED_NAME_PATTERNS = [
    "publish_research_",
    "research_",
    "archive_",
    "scan_",
    "tune_",
    "materialize_four_year_",
    "build_four_year_",
    "incremental_formal_l4_",
]

EXCLUDED_EXACT_NAMES = {
    "build_standard_chain_score_asset.py",
    "rolling_train_module.py",
    "run_parallel_expanding2010_folds.py",
}

DIRECT_WRITE_MARKERS = [
    "to_sql(",
]

MODEL_DB_MARKERS = [
    "resolve_model_prediction_db_path",
    "model_predictions.db",
]

SYNC_MARKERS = [
    "l4_duckdb_sync_enabled",
    "duckdb_sync",
]


def rel(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scan L4 MODEL_PREDICTIONS write-chain coverage for DuckDB sync.")
    parser.add_argument("--report", required=True)
    args = parser.parse_args(argv)

    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = PROJECT_ROOT / report_path

    entries: list[dict[str, object]] = []
    sqlite_only_bypass_count = 0
    unexpected_direct_writer_count = 0

    for path in MAINLINE_L4_WRITERS:
        text = read_text(path).lower()
        markers_present = all(marker in text for marker in SYNC_MARKERS)
        entry = {
            "path": rel(path),
            "classification": "mainline_l4_writer",
            "writes_prediction_db": any(marker in text for marker in DIRECT_WRITE_MARKERS)
            and any(marker in text for marker in MODEL_DB_MARKERS),
            "duckdb_sync_hook_present": markers_present,
            "status": "covered" if markers_present else "sqlite_only_bypass",
        }
        if not markers_present:
            sqlite_only_bypass_count += 1
        entries.append(entry)

    for path in LEGACY_OR_RESEARCH:
        text = read_text(path).lower()
        entries.append(
            {
                "path": rel(path),
                "classification": "legacy_or_research",
                "writes_prediction_db": any(marker in text for marker in DIRECT_WRITE_MARKERS)
                and any(marker in text for marker in MODEL_DB_MARKERS),
                "duckdb_sync_hook_present": "l4_duckdb_sync_enabled" in text,
                "status": "excluded_from_current_l4_registry_scope",
            }
        )

    scanned = {path.resolve() for path in MAINLINE_L4_WRITERS + LEGACY_OR_RESEARCH}
    for path in (PROJECT_ROOT / "quant" / "main").glob("*.py"):
        resolved = path.resolve()
        if resolved in scanned:
            continue
        text = read_text(path).lower()
        if not any(marker in text for marker in DIRECT_WRITE_MARKERS):
            continue
        if not any(marker in text for marker in MODEL_DB_MARKERS):
            continue
        lowered_name = path.name.lower()
        if lowered_name in EXCLUDED_EXACT_NAMES or any(lowered_name.startswith(pattern) for pattern in EXCLUDED_NAME_PATTERNS):
            entries.append(
                {
                    "path": rel(path),
                    "classification": "legacy_or_research",
                    "writes_prediction_db": True,
                    "duckdb_sync_hook_present": "l4_duckdb_sync_enabled" in text,
                    "status": "excluded_from_current_l4_registry_scope",
                }
            )
            continue
        unexpected_direct_writer_count += 1
        entries.append(
            {
                "path": rel(path),
                "classification": "unexpected_direct_writer",
                "writes_prediction_db": True,
                "duckdb_sync_hook_present": "l4_duckdb_sync_enabled" in text,
                "status": "review_required",
            }
        )

    result = {
        "status": "ok" if sqlite_only_bypass_count == 0 and unexpected_direct_writer_count == 0 else "failed",
        "mainline_writer_count": len(MAINLINE_L4_WRITERS),
        "mainline_writer_covered_count": len(MAINLINE_L4_WRITERS) - sqlite_only_bypass_count,
        "sqlite_only_l4_bypass_count": sqlite_only_bypass_count,
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
