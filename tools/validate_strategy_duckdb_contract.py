from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MAIN_DIR = PROJECT_ROOT / "quant" / "main"
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from duckdb_asset_route import connect_duckdb_readonly
from l5_duckdb_sync import DEFAULT_MANIFEST_TABLE, DEFAULT_REGISTRY_TABLE
from l6_duckdb_sync import (
    DEFAULT_BACKTEST_CSV_TABLE,
    DEFAULT_BACKTEST_FILE_TABLE,
    DEFAULT_BACKTEST_JSON_TABLE,
    DEFAULT_VALIDATION_TABLE,
    resolve_l6_table_duckdb_path,
)
from l7_duckdb_sync import (
    DEFAULT_SIGNAL_FILES_TABLE,
    DEFAULT_SIGNAL_ROWS_TABLE,
    DEFAULT_SIGNAL_STATUS_TABLE,
)
from project_paths import resolve_data_dir, resolve_project_path
from strategy_asset_route import (
    load_current_production_strategy_context,
    load_strategy_validation_payload,
    resolve_strategy_duckdb_path,
    resolve_strategy_manifest_duckdb_path,
    resolve_strategy_registry_duckdb_path,
    resolve_strategy_signal_files_duckdb_path,
    resolve_strategy_signal_rows_duckdb_path,
    resolve_strategy_signal_status_duckdb_path,
    resolve_strategy_signal_duckdb_path,
    resolve_strategy_root,
    resolve_strategy_validation_duckdb_path,
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return max(sum(1 for _ in csv.reader(handle)) - 1, 0)


def _current_production_entries(registry: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in registry.get("production", {}).get("strategies", [])
        if item.get("status") == "production" and item.get("strategy_id")
    ]


def _infer_strategy_id(file_name: str, strategy_ids: set[str]) -> str | None:
    matches = [strategy_id for strategy_id in strategy_ids if file_name.startswith(strategy_id)]
    if not matches:
        return None
    return max(matches, key=len)


def _source_summary(project_dir: Path, data_dir: Path, registry: dict[str, Any]) -> dict[str, Any]:
    entries = _current_production_entries(registry)
    active_ids = {str(item["strategy_id"]) for item in entries}
    strategy_root = resolve_strategy_root(project_dir / "strategy_library" / "production")
    signal_root = data_dir / "production_signals"

    manifest_count = 0
    validation_count = 0
    backtest_file_count = 0
    backtest_json_count = 0
    backtest_csv_row_count = 0

    for item in entries:
        strategy_dir = strategy_root / str(item["strategy_id"])
        if (strategy_dir / "strategy_manifest.json").is_file():
            manifest_count += 1
        if (strategy_dir / "validation.json").is_file():
            validation_count += 1
        backtest_dir = strategy_dir / "backtests"
        if not backtest_dir.is_dir():
            continue
        for path in backtest_dir.iterdir():
            if not path.is_file():
                continue
            backtest_file_count += 1
            if path.suffix.lower() == ".json":
                backtest_json_count += 1
            elif path.suffix.lower() == ".csv":
                backtest_csv_row_count += _csv_row_count(path)

    signal_file_count = 0
    signal_status_count = 0
    signal_row_count = 0
    for path in signal_root.glob("*"):
        if not path.is_file():
            continue
        if active_ids and _infer_strategy_id(path.name, active_ids) is None:
            continue
        signal_file_count += 1
        if path.suffix.lower() == ".json":
            signal_status_count += 1
        elif path.suffix.lower() == ".csv":
            signal_row_count += _csv_row_count(path)

    return {
        "production_strategy_count": len(entries),
        "manifest_count": manifest_count,
        "validation_count": validation_count,
        "backtest_file_count": backtest_file_count,
        "backtest_json_count": backtest_json_count,
        "backtest_csv_row_count": backtest_csv_row_count,
        "signal_file_count": signal_file_count,
        "signal_status_count": signal_status_count,
        "signal_row_count": signal_row_count,
    }


def _count_rows(path: Path, table: str) -> int:
    with connect_duckdb_readonly(path) as conn:
        return int(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])


def _duckdb_summary(paths: dict[str, Path]) -> dict[str, Any]:
    l5 = {
        "registry_rows": _count_rows(paths["l5_registry"], DEFAULT_REGISTRY_TABLE),
        "manifest_rows": _count_rows(paths["l5_manifest"], DEFAULT_MANIFEST_TABLE),
    }
    l6 = {
        "validation_rows": _count_rows(paths["l6_validation"], DEFAULT_VALIDATION_TABLE),
        "backtest_file_rows": _count_rows(paths["l6_files"], DEFAULT_BACKTEST_FILE_TABLE),
        "backtest_json_rows": _count_rows(paths["l6_json"], DEFAULT_BACKTEST_JSON_TABLE),
        "backtest_csv_rows": _count_rows(paths["l6_csv"], DEFAULT_BACKTEST_CSV_TABLE),
    }
    l7 = {
        "signal_file_rows": _count_rows(paths["l7_files"], DEFAULT_SIGNAL_FILES_TABLE),
        "signal_status_rows": _count_rows(paths["l7_status"], DEFAULT_SIGNAL_STATUS_TABLE),
        "signal_rows": _count_rows(paths["l7_rows"], DEFAULT_SIGNAL_ROWS_TABLE),
    }
    return {"l5": l5, "l6": l6, "l7": l7}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate L5/L6/L7 DuckDB strategy asset contract.")
    parser.add_argument("--report", required=True)
    parser.add_argument("--project-dir", default=str(PROJECT_ROOT / "quant" / "main"))
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "quant" / "data_file"))
    parser.add_argument("--registry-file", default="strategy_library/registry.json")
    args = parser.parse_args(argv)

    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = PROJECT_ROOT / report_path

    project_dir = resolve_project_path(args.project_dir)
    data_dir = resolve_data_dir(args.data_dir)
    registry_path = resolve_project_path(args.registry_file)
    resolved_paths = {
        "l5_registry": resolve_strategy_registry_duckdb_path(data_dir=data_dir, require_exists=True),
        "l5_manifest": resolve_strategy_manifest_duckdb_path(data_dir=data_dir, require_exists=True),
        "l6_validation": resolve_strategy_validation_duckdb_path(data_dir=data_dir, require_exists=True),
        "l6_files": resolve_l6_table_duckdb_path(DEFAULT_BACKTEST_FILE_TABLE, data_dir=data_dir, require_exists=True),
        "l6_json": resolve_l6_table_duckdb_path(DEFAULT_BACKTEST_JSON_TABLE, data_dir=data_dir, require_exists=True),
        "l6_csv": resolve_l6_table_duckdb_path(DEFAULT_BACKTEST_CSV_TABLE, data_dir=data_dir, require_exists=True),
        "l7_rows": resolve_strategy_signal_rows_duckdb_path(data_dir=data_dir, require_exists=True),
        "l7_files": resolve_strategy_signal_files_duckdb_path(data_dir=data_dir, require_exists=True),
        "l7_status": resolve_strategy_signal_status_duckdb_path(data_dir=data_dir, require_exists=True),
    }
    registry = _load_json(registry_path)
    source = _source_summary(project_dir, data_dir, registry)
    duckdb = _duckdb_summary(resolved_paths)
    current_context = load_current_production_strategy_context(
        registry_file=registry_path,
        strategy_root=project_dir / "strategy_library" / "production",
        signal_dir=data_dir / "production_signals",
        data_dir=data_dir,
        backend="duckdb",
    )
    validation_payload = load_strategy_validation_payload(
        current_context["strategy_entry"]["strategy_id"],
        data_dir=data_dir,
        backend="duckdb",
    )

    checks = {
        "registry_rows_match": duckdb["l5"]["registry_rows"] == source["production_strategy_count"],
        "manifest_rows_match": duckdb["l5"]["manifest_rows"] == source["manifest_count"],
        "validation_rows_match": duckdb["l6"]["validation_rows"] == source["validation_count"],
        "backtest_file_rows_match": duckdb["l6"]["backtest_file_rows"] == source["backtest_file_count"],
        "backtest_json_rows_match": duckdb["l6"]["backtest_json_rows"] == source["backtest_json_count"],
        "backtest_csv_rows_match": duckdb["l6"]["backtest_csv_rows"] == source["backtest_csv_row_count"],
        "signal_file_rows_match": duckdb["l7"]["signal_file_rows"] == source["signal_file_count"],
        "signal_status_rows_match": duckdb["l7"]["signal_status_rows"] == source["signal_status_count"],
        "signal_rows_match": duckdb["l7"]["signal_rows"] == source["signal_row_count"],
        "current_strategy_context_ok": bool(current_context["signal_rows"]),
        "current_strategy_validation_ok": isinstance(validation_payload, dict) and bool(validation_payload),
    }
    result = {
        "status": "ok" if all(checks.values()) else "failed",
        "duckdb_paths": {
            key: str(value) for key, value in resolved_paths.items()
        },
        "registry_path": str(registry_path),
        "source_summary": source,
        "duckdb_summary": duckdb,
        "checks": checks,
        "current_strategy": {
            "strategy_id": current_context["strategy_entry"].get("strategy_id"),
            "signal_source": current_context["signal_path"],
            "signal_row_count": len(current_context["signal_rows"]),
            "validation_status": validation_payload.get("status"),
        },
        "boundary": {
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
