from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from duckdb_asset_route import resolve_duckdb_path
from production_asset_registry import active_main_workflow_asset, active_main_workflow_asset_for_table, split_asset_path
from project_paths import PROJECT_ROOT, resolve_data_dir


ENV_L6_DUCKDB_SYNC = "QUANT_L6_DUCKDB_SYNC"
DEFAULT_VALIDATION_TABLE = "prod_l6_strategy_validation_current"
DEFAULT_BACKTEST_FILE_TABLE = "prod_l6_backtest_files_current"
DEFAULT_BACKTEST_JSON_TABLE = "prod_l6_backtest_json_current"
DEFAULT_BACKTEST_CSV_TABLE = "prod_l6_backtest_csv_rows_current"


def _default_l6_table_duckdb_path(table_name: str, data_dir: str | Path | None = None) -> Path:
    base = resolve_data_dir(data_dir) / "production_assets" / "duckdb" / "production" / "l6"
    filename_by_table = {
        DEFAULT_VALIDATION_TABLE: "prod_l6_strategy_validation_current.duckdb",
        DEFAULT_BACKTEST_FILE_TABLE: "prod_l6_backtest_files_current.duckdb",
        DEFAULT_BACKTEST_JSON_TABLE: "prod_l6_backtest_json_current.duckdb",
        DEFAULT_BACKTEST_CSV_TABLE: "prod_l6_backtest_csv_rows_current.duckdb",
    }
    filename = filename_by_table.get(str(table_name).strip())
    if filename:
        return base / filename
    raise ValueError(f"unsupported L6 current table for split DuckDB route: {table_name}")


def l6_duckdb_sync_enabled() -> bool:
    value = str(os.environ.get(ENV_L6_DUCKDB_SYNC, "1")).strip().lower()
    return value not in {"0", "false", "no", "off"}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_latest_signal_status(
    strategy_dir: Path,
    manifest: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, Any]:
    auto_status_path = strategy_dir / "signals" / "latest_signal_status_auto.json"
    if auto_status_path.is_file():
        payload = _load_json(auto_status_path)
        if isinstance(payload, dict):
            return payload

    current_signal = manifest.get("current_signal")
    if isinstance(current_signal, dict) and current_signal:
        return {
            "signal_date": current_signal.get("signal_date"),
            "buy_date": current_signal.get("buy_date"),
            "latest_signal_date": current_signal.get("signal_date"),
            "latest_buy_date": current_signal.get("buy_date"),
            "buy_day_hard_gate_complete": current_signal.get("buy_day_hard_gate_complete"),
            "selected_stock_code": current_signal.get("selected_stock_code"),
        }

    latest_signal_status = validation.get("latest_signal_status")
    if isinstance(latest_signal_status, dict):
        return latest_signal_status
    return {}


def _resolve_strategy_dir(project_dir: Path, registry_item: dict[str, Any]) -> Path | None:
    raw = registry_item.get("path")
    if raw in (None, ""):
        return None
    path = Path(str(raw)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (project_dir / path).resolve()


def _csv_rows(path: Path, strategy_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for index, row in enumerate(reader, start=1):
            payload = dict(row)
            rows.append(
                {
                    "strategy_id": strategy_id,
                    "file_name": path.name,
                    "source_path": str(path),
                    "row_index": index,
                    "trade_date": payload.get("trade_date"),
                    "annual": payload.get("annual"),
                    "sharpe": payload.get("sharpe"),
                    "max_drawdown": payload.get("max_drawdown"),
                    "signal_file": payload.get("signal_file"),
                    "log_file": payload.get("log_file"),
                    "row_payload_json": json.dumps(payload, ensure_ascii=False),
                }
            )
    return rows


def _validation_row(strategy_id: str, strategy_dir: Path, manifest: dict[str, Any], validation: dict[str, Any]) -> dict[str, Any]:
    metrics = validation.get("metrics", {}) if isinstance(validation.get("metrics"), dict) else {}
    latest_signal_status = _resolve_latest_signal_status(strategy_dir, manifest, validation)
    hard_gate_audit = validation.get("hard_gate_audit", {}) if isinstance(validation.get("hard_gate_audit"), dict) else {}
    return {
        "strategy_id": strategy_id,
        "strategy_dir": str(strategy_dir),
        "strategy_name": manifest.get("strategy_name") or manifest.get("name"),
        "status": validation.get("status") or manifest.get("status"),
        "validation_platform": validation.get("validation_platform") or manifest.get("validation_platform"),
        "published_at": manifest.get("published_at"),
        "annual_return": metrics.get("annual_return"),
        "pnl_ratio": metrics.get("pnl_ratio"),
        "sharpe": metrics.get("sharpe"),
        "max_drawdown": metrics.get("max_drawdown"),
        "avg_invested_pct": metrics.get("avg_invested_pct"),
        "max_active_positions": metrics.get("max_active_positions"),
        "latest_signal_date": latest_signal_status.get("signal_date"),
        "latest_buy_date": latest_signal_status.get("buy_date"),
        "buy_day_hard_gate_complete": latest_signal_status.get("buy_day_hard_gate_complete"),
        "selected_stock_code": latest_signal_status.get("selected_stock_code"),
        "hard_gate_failed_files": hard_gate_audit.get("failed_files"),
        "hard_gate_signal_rows": hard_gate_audit.get("signal_rows"),
        "validation_json_path": str(strategy_dir / "validation.json"),
        "validation_payload_json": json.dumps(validation, ensure_ascii=False),
    }


def resolve_l6_duckdb_path(
    data_dir: str | Path | None = None,
    *,
    duckdb_path: str | Path | None = None,
    require_exists: bool = False,
) -> Path:
    return resolve_l6_table_duckdb_path(
        DEFAULT_VALIDATION_TABLE,
        data_dir=data_dir,
        duckdb_path=duckdb_path,
        require_exists=require_exists,
    )


def resolve_l6_table_duckdb_path(
    table_name: str,
    data_dir: str | Path | None = None,
    *,
    duckdb_path: str | Path | None = None,
    require_exists: bool = False,
) -> Path:
    if duckdb_path:
        path = Path(duckdb_path)
    else:
        asset = active_main_workflow_asset_for_table("L6_backtest", table_name, data_dir=data_dir)
        asset_path, asset_table = split_asset_path(asset.get("asset_path")) if asset else (None, None)
        if asset_path and asset_path.suffix.lower() == ".duckdb" and str(asset_table or "").strip() == str(table_name).strip():
            path = asset_path
        else:
            asset = active_main_workflow_asset("L6_backtest", data_dir=data_dir)
            asset_path, _table = split_asset_path(asset.get("asset_path")) if asset else (None, None)
            path = (
                asset_path
                if asset_path and asset_path.suffix.lower() == ".duckdb"
                else _default_l6_table_duckdb_path(table_name, data_dir=data_dir)
            )
    if require_exists and not path.is_file():
        raise FileNotFoundError(f"L6 DuckDB asset not found: {path}")
    return path


def sync_strategy_backtests_to_duckdb(
    *,
    project_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
    registry_path: str | Path | None = None,
    duckdb_path: str | Path | None = None,
    validation_table: str = DEFAULT_VALIDATION_TABLE,
    backtest_file_table: str = DEFAULT_BACKTEST_FILE_TABLE,
    backtest_json_table: str = DEFAULT_BACKTEST_JSON_TABLE,
    backtest_csv_table: str = DEFAULT_BACKTEST_CSV_TABLE,
) -> dict[str, int | str]:
    project_base = Path(project_dir).resolve() if project_dir else PROJECT_ROOT
    raw_registry_path = Path(registry_path) if registry_path is not None else (project_base / "strategy_library" / "registry.json")
    registry_file = raw_registry_path.resolve() if raw_registry_path.is_absolute() else (project_base / raw_registry_path).resolve()
    registry = _load_json(registry_file)
    strategies = [
        item
        for item in (registry.get("production", {}).get("strategies", []) or [])
        if item.get("status") == "production" and item.get("strategy_id")
    ]

    validation_rows: list[dict[str, Any]] = []
    file_rows: list[dict[str, Any]] = []
    json_rows: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []

    for item in strategies:
        strategy_id = str(item.get("strategy_id"))
        strategy_dir = _resolve_strategy_dir(project_base, item)
        if not strategy_dir:
            continue
        manifest_path = strategy_dir / "strategy_manifest.json"
        validation_path = strategy_dir / "validation.json"
        if not manifest_path.is_file() or not validation_path.is_file():
            continue
        manifest = _load_json(manifest_path)
        validation = _load_json(validation_path)
        validation_rows.append(_validation_row(strategy_id, strategy_dir, manifest, validation))

        backtest_dir = strategy_dir / "backtests"
        if not backtest_dir.is_dir():
            continue
        for path in sorted(backtest_dir.iterdir()):
            if not path.is_file():
                continue
            modified_at = datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
            file_rows.append(
                {
                    "strategy_id": strategy_id,
                    "file_name": path.name,
                    "source_path": str(path),
                    "file_suffix": path.suffix.lower(),
                    "size_bytes": path.stat().st_size,
                    "modified_at": modified_at,
                }
            )
            if path.suffix.lower() == ".json":
                payload = _load_json(path)
                json_rows.append(
                    {
                        "strategy_id": strategy_id,
                        "file_name": path.name,
                        "source_path": str(path),
                        "payload_json": json.dumps(payload, ensure_ascii=False),
                    }
                )
            elif path.suffix.lower() == ".csv":
                csv_rows.extend(_csv_rows(path, strategy_id))

    validation_path_target = resolve_l6_table_duckdb_path(
        validation_table,
        data_dir=data_dir,
        duckdb_path=duckdb_path,
    )
    file_path_target = resolve_l6_table_duckdb_path(
        backtest_file_table,
        data_dir=data_dir,
        duckdb_path=duckdb_path if duckdb_path else None,
    )
    json_path_target = resolve_l6_table_duckdb_path(
        backtest_json_table,
        data_dir=data_dir,
        duckdb_path=duckdb_path if duckdb_path else None,
    )
    csv_path_target = resolve_l6_table_duckdb_path(
        backtest_csv_table,
        data_dir=data_dir,
        duckdb_path=duckdb_path if duckdb_path else None,
    )
    for path in (validation_path_target, file_path_target, json_path_target, csv_path_target):
        path.parent.mkdir(parents=True, exist_ok=True)

    validation_df = pd.DataFrame(
        validation_rows,
        columns=[
            "strategy_id",
            "strategy_dir",
            "strategy_name",
            "status",
            "validation_platform",
            "published_at",
            "annual_return",
            "pnl_ratio",
            "sharpe",
            "max_drawdown",
            "avg_invested_pct",
            "max_active_positions",
            "latest_signal_date",
            "latest_buy_date",
            "buy_day_hard_gate_complete",
            "selected_stock_code",
            "hard_gate_failed_files",
            "hard_gate_signal_rows",
            "validation_json_path",
            "validation_payload_json",
        ],
    )
    file_df = pd.DataFrame(
        file_rows,
        columns=[
            "strategy_id",
            "file_name",
            "source_path",
            "file_suffix",
            "size_bytes",
            "modified_at",
        ],
    )
    json_df = pd.DataFrame(
        json_rows,
        columns=[
            "strategy_id",
            "file_name",
            "source_path",
            "payload_json",
        ],
    )
    csv_df = pd.DataFrame(
        csv_rows,
        columns=[
            "strategy_id",
            "file_name",
            "source_path",
            "row_index",
            "trade_date",
            "annual",
            "sharpe",
            "max_drawdown",
            "signal_file",
            "log_file",
            "row_payload_json",
        ],
    )

    import duckdb

    with duckdb.connect(str(validation_path_target)) as conn:
        conn.register("_validation_df", validation_df)
        conn.execute(f'CREATE OR REPLACE TABLE "{validation_table}" AS SELECT * FROM _validation_df')
        conn.unregister("_validation_df")
    with duckdb.connect(str(file_path_target)) as conn:
        conn.register("_file_df", file_df)
        conn.execute(f'CREATE OR REPLACE TABLE "{backtest_file_table}" AS SELECT * FROM _file_df')
        conn.unregister("_file_df")
    with duckdb.connect(str(json_path_target)) as conn:
        conn.register("_json_df", json_df)
        conn.execute(f'CREATE OR REPLACE TABLE "{backtest_json_table}" AS SELECT * FROM _json_df')
        conn.unregister("_json_df")
    with duckdb.connect(str(csv_path_target)) as conn:
        conn.register("_csv_df", csv_df)
        conn.execute(f'CREATE OR REPLACE TABLE "{backtest_csv_table}" AS SELECT * FROM _csv_df')
        conn.unregister("_csv_df")

    return {
        "duckdb_path": str(validation_path_target),
        "validation_duckdb_path": str(validation_path_target),
        "backtest_file_duckdb_path": str(file_path_target),
        "backtest_json_duckdb_path": str(json_path_target),
        "backtest_csv_duckdb_path": str(csv_path_target),
        "registry_path": str(registry_file),
        "validation_table": validation_table,
        "backtest_file_table": backtest_file_table,
        "backtest_json_table": backtest_json_table,
        "backtest_csv_table": backtest_csv_table,
        "validation_row_count": int(len(validation_rows)),
        "backtest_file_count": int(len(file_rows)),
        "backtest_json_count": int(len(json_rows)),
        "backtest_csv_row_count": int(len(csv_rows)),
    }
