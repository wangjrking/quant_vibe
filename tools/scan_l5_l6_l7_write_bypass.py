from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MAIN_DIR = PROJECT_ROOT / "quant" / "main"
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))


MAINLINE_WRITERS = [
    PROJECT_ROOT / "quant" / "main" / "run_production_tasks.py",
]

READONLY_CONSUMERS = [
    PROJECT_ROOT / "quant" / "main" / "build_manual_trade_package.py",
    PROJECT_ROOT / "quant" / "main" / "tools" / "qmt_position_monitor.py",
]

EXCLUDED_PREFIXES = (
    "publish_",
    "archive_",
    "research_",
    "tune_",
    "screen_",
    "scan_",
    "materialize_",
    "build_four_year_",
    "analyze_",
    "check_",
    "optimize_",
    "rebacktest_",
    "validate_",
    "stress_",
    "evaluate_",
)

EXCLUDED_FILENAMES = {
    "agent_observability.py",
}

TARGET_PATH_MARKERS = (
    "strategy_library/production",
    "strategy_library\\production",
    "production_signals",
    "strategy_library/registry.json",
    "strategy_library\\registry.json",
)

TARGET_FILE_MARKERS = (
    "strategy_manifest.json",
    "validation.json",
    "time_slices.csv",
    "summary.json",
)

TARGET_SCOPE_VARIABLES = (
    "strategy_dir",
    "signal_dir",
    "registry_path",
    "strategy_root",
    "production_root",
    "production_dir",
)

WRITE_MARKERS = [
    "write_text(",
    "json.dump(",
    "to_csv(",
    "csv.dictwriter(",
]

SYNC_MARKERS = [
    "sync_strategy_registry_to_duckdb",
    "sync_strategy_backtests_to_duckdb",
    "sync_production_signal_artifacts_to_duckdb",
]


def rel(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_mainline_step_writers() -> list[Path]:
    config_path = PROJECT_ROOT / "quant" / "main" / "config" / "production_tasks.example.json"
    if not config_path.exists():
        return []
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    writers: list[Path] = []
    for strategy in payload.get("strategies", []) or []:
        for step in strategy.get("steps", []) or []:
            command = [str(item) for item in step.get("command", []) or []]
            for item in command:
                if item.endswith(".py") and "{" not in item:
                    candidate = (PROJECT_ROOT / "quant" / "main" / item).resolve()
                    if candidate.exists() and candidate not in writers:
                        writers.append(candidate)
    return writers


def writes_current_scope_assets(text: str) -> bool:
    lowered = text.lower()
    lines = lowered.splitlines()
    for idx, line in enumerate(lines):
        if not any(marker in line for marker in WRITE_MARKERS):
            continue
        window = "\n".join(lines[max(0, idx - 4) : idx + 5])
        if any(marker in window for marker in TARGET_PATH_MARKERS):
            return True
        has_target_file = any(marker in window for marker in TARGET_FILE_MARKERS)
        has_scope_variable = any(marker in window for marker in TARGET_SCOPE_VARIABLES)
        if has_target_file and has_scope_variable:
            return True
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scan current L5/L6/L7 mainline write coverage for DuckDB sync hooks.")
    parser.add_argument("--report", required=True)
    args = parser.parse_args(argv)

    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = PROJECT_ROOT / report_path

    entries: list[dict[str, object]] = []
    sqlite_only_bypass_count = 0
    unexpected_direct_writer_count = 0
    mainline_step_writers = load_mainline_step_writers()

    for path in MAINLINE_WRITERS:
        text = read_text(path).lower()
        covered = all(marker in text for marker in SYNC_MARKERS)
        entries.append(
            {
                "path": rel(path),
                "classification": "mainline_l5_l6_l7_writer",
                "writes_target_assets": True,
                "duckdb_sync_hook_present": covered,
                "status": "covered" if covered else "legacy_only_bypass",
            }
        )
        if not covered:
            sqlite_only_bypass_count += 1

    for path in mainline_step_writers:
        text = read_text(path).lower()
        entries.append(
            {
                "path": rel(path),
                "classification": "mainline_step_writer",
                "writes_target_assets": writes_current_scope_assets(text),
                "duckdb_sync_hook_present": False,
                "status": "covered_by_run_production_tasks_post_step_sync",
            }
        )

    for path in READONLY_CONSUMERS:
        text = read_text(path).lower()
        entries.append(
            {
                "path": rel(path),
                "classification": "readonly_consumer",
                "writes_target_assets": writes_current_scope_assets(text),
                "duckdb_sync_hook_present": False,
                "status": "read_only_scope",
            }
        )

    scanned = {path.resolve() for path in MAINLINE_WRITERS + READONLY_CONSUMERS + mainline_step_writers}
    for path in (PROJECT_ROOT / "quant" / "main").glob("*.py"):
        resolved = path.resolve()
        if resolved in scanned:
            continue
        text = read_text(path).lower()
        if not writes_current_scope_assets(text):
            continue
        if path.name in EXCLUDED_FILENAMES or path.name.startswith(EXCLUDED_PREFIXES):
            entries.append(
                {
                    "path": rel(path),
                    "classification": "legacy_or_manual_publish",
                    "writes_target_assets": True,
                    "duckdb_sync_hook_present": any(marker in text for marker in SYNC_MARKERS),
                    "status": "excluded_from_current_l5_l6_l7_registry_scope",
                }
            )
            continue
        unexpected_direct_writer_count += 1
        entries.append(
            {
                "path": rel(path),
                "classification": "unexpected_direct_writer",
                "writes_target_assets": True,
                "duckdb_sync_hook_present": any(marker in text for marker in SYNC_MARKERS),
                "status": "review_required",
            }
        )

    result = {
        "status": "ok" if sqlite_only_bypass_count == 0 and unexpected_direct_writer_count == 0 else "failed",
        "mainline_writer_count": len(MAINLINE_WRITERS),
        "mainline_writer_covered_count": len(MAINLINE_WRITERS) - sqlite_only_bypass_count,
        "legacy_only_bypass_count": sqlite_only_bypass_count,
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
