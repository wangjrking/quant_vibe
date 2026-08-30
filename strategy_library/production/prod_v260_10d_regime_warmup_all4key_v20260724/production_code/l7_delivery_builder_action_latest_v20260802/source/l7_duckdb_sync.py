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


ENV_L7_DUCKDB_SYNC = "QUANT_L7_DUCKDB_SYNC"
DEFAULT_SIGNAL_ROWS_TABLE = "prod_l7_signal_rows_current"
DEFAULT_SIGNAL_STATUS_TABLE = "prod_l7_signal_status_current"
DEFAULT_SIGNAL_FILES_TABLE = "prod_l7_signal_files_current"


def _default_l7_table_duckdb_path(table_name: str, data_dir: str | Path | None = None) -> Path:
    base = resolve_data_dir(data_dir) / "production_assets" / "duckdb" / "production" / "l7"
    filename_by_table = {
        DEFAULT_SIGNAL_ROWS_TABLE: "prod_l7_signal_rows_current.duckdb",
        DEFAULT_SIGNAL_STATUS_TABLE: "prod_l7_signal_status_current.duckdb",
        DEFAULT_SIGNAL_FILES_TABLE: "prod_l7_signal_files_current.duckdb",
    }
    filename = filename_by_table.get(str(table_name).strip())
    if filename:
        return base / filename
    raise ValueError(f"unsupported L7 current table for split DuckDB route: {table_name}")


def l7_duckdb_sync_enabled() -> bool:
    value = str(os.environ.get(ENV_L7_DUCKDB_SYNC, "1")).strip().lower()
    return value not in {"0", "false", "no", "off"}


def _infer_strategy_id(file_name: str, strategy_ids: set[str]) -> str | None:
    matches = [strategy_id for strategy_id in strategy_ids if file_name.startswith(strategy_id)]
    if not matches:
        return None
    return max(matches, key=len)


def _load_registry_strategy_ids(
    *,
    registry_path: str | Path | None = None,
    project_dir: str | Path | None = None,
) -> set[str]:
    project_base = Path(project_dir).resolve() if project_dir else PROJECT_ROOT
    raw_registry_path = Path(registry_path) if registry_path is not None else (project_base / "strategy_library" / "registry.json")
    registry_file = raw_registry_path.resolve() if raw_registry_path.is_absolute() else (project_base / raw_registry_path).resolve()
    if not registry_file.is_file():
        return set()
    registry = json.loads(registry_file.read_text(encoding="utf-8"))
    return {
        str(item.get("strategy_id"))
        for item in (registry.get("production", {}).get("strategies", []) or [])
        if item.get("status") == "production" and item.get("strategy_id")
    }


def require_current_production_strategy_id(
    *,
    registry_path: str | Path | None = None,
    project_dir: str | Path | None = None,
) -> str:
    project_base = Path(project_dir).resolve() if project_dir else PROJECT_ROOT
    raw_registry_path = Path(registry_path) if registry_path is not None else (project_base / "strategy_library" / "registry.json")
    registry_file = raw_registry_path.resolve() if raw_registry_path.is_absolute() else (project_base / raw_registry_path).resolve()
    if not registry_file.is_file():
        raise FileNotFoundError(f"strategy registry not found: {registry_file}")
    registry = json.loads(registry_file.read_text(encoding="utf-8-sig"))
    production = registry.get("production", {})
    current = str(production.get("current") or "").strip()
    if not current:
        raise ValueError(
            "production.current missing in strategy registry; "
            "L7 signal synchronization is fail-closed"
        )
    production_ids = {
        str(item.get("strategy_id"))
        for item in (production.get("strategies", []) or [])
        if item.get("status") == "production" and item.get("strategy_id")
    }
    if production_ids != {current}:
        raise ValueError(
            "production strategy registry must contain exactly current before L7 synchronization: "
            f"current={current}, production_ids={sorted(production_ids)}"
        )
    return current


def _csv_rows(path: Path, strategy_id: str | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for index, row in enumerate(reader, start=1):
            row_payload = dict(row)
            rows.append(
                {
                    "strategy_id": strategy_id,
                    "file_name": path.name,
                    "source_path": str(path),
                    "row_index": index,
                    "signal_date": row_payload.get("signal_date"),
                    "buy_date": row_payload.get("buy_date"),
                    "stock_code": row_payload.get("stock_code"),
                    "symbol": row_payload.get("symbol"),
                    "name": row_payload.get("name"),
                    "rank": row_payload.get("rank"),
                    "pred_prob": row_payload.get("pred_prob"),
                    "target_pct": row_payload.get("target_pct"),
                    "row_payload_json": json.dumps(row_payload, ensure_ascii=False),
                }
            )
    return rows


def _json_status_row(
    path: Path,
    strategy_id: str | None,
    *,
    payload_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload_overrides:
        payload.update(payload_overrides)
    return {
        "strategy_id": strategy_id,
        "file_name": path.name,
        "source_path": str(path),
        "status": payload.get("status"),
        "latest_signal_date": payload.get("latest_signal_date"),
        "latest_buy_date": payload.get("latest_buy_date"),
        "published_at": payload.get("published_at"),
        "payload_json": json.dumps(payload, ensure_ascii=False),
    }


def resolve_l7_duckdb_path(
    data_dir: str | Path | None = None,
    *,
    duckdb_path: str | Path | None = None,
    require_exists: bool = False,
) -> Path:
    return resolve_l7_table_duckdb_path(
        DEFAULT_SIGNAL_ROWS_TABLE,
        data_dir=data_dir,
        duckdb_path=duckdb_path,
        require_exists=require_exists,
    )


def resolve_l7_table_duckdb_path(
    table_name: str,
    data_dir: str | Path | None = None,
    *,
    duckdb_path: str | Path | None = None,
    require_exists: bool = False,
) -> Path:
    if duckdb_path:
        path = Path(duckdb_path)
    else:
        asset = active_main_workflow_asset_for_table("L7_trading_delivery", table_name, data_dir=data_dir)
        asset_path, asset_table = split_asset_path(asset.get("asset_path")) if asset else (None, None)
        if asset_path and asset_path.suffix.lower() == ".duckdb" and str(asset_table or "").strip() == str(table_name).strip():
            path = asset_path
        else:
            asset = active_main_workflow_asset("L7_trading_delivery", data_dir=data_dir)
            asset_path, _table = split_asset_path(asset.get("asset_path")) if asset else (None, None)
            path = (
                asset_path
                if asset_path and asset_path.suffix.lower() == ".duckdb"
                else _default_l7_table_duckdb_path(table_name, data_dir=data_dir)
            )
    if require_exists and not path.is_file():
        raise FileNotFoundError(f"L7 DuckDB asset not found: {path}")
    return path


def sync_production_signal_artifacts_to_duckdb(
    *,
    data_dir: str | Path | None = None,
    signal_dir: str | Path | None = None,
    duckdb_path: str | Path | None = None,
    project_dir: str | Path | None = None,
    registry_path: str | Path | None = None,
    strategy_ids: list[str] | set[str] | None = None,
    signal_rows_table: str = DEFAULT_SIGNAL_ROWS_TABLE,
    signal_status_table: str = DEFAULT_SIGNAL_STATUS_TABLE,
    signal_files_table: str = DEFAULT_SIGNAL_FILES_TABLE,
    status_payload_overrides: dict[str, Any] | None = None,
) -> dict[str, int | str]:
    base_data_dir = resolve_data_dir(data_dir)
    signal_root = Path(signal_dir).resolve() if signal_dir else (base_data_dir / "production_signals")
    signal_rows_path_target = resolve_l7_table_duckdb_path(
        signal_rows_table,
        data_dir=data_dir,
        duckdb_path=duckdb_path,
    )
    signal_status_path_target = resolve_l7_table_duckdb_path(
        signal_status_table,
        data_dir=data_dir,
        duckdb_path=duckdb_path if duckdb_path else None,
    )
    signal_files_path_target = resolve_l7_table_duckdb_path(
        signal_files_table,
        data_dir=data_dir,
        duckdb_path=duckdb_path if duckdb_path else None,
    )
    for path in (signal_rows_path_target, signal_status_path_target, signal_files_path_target):
        path.parent.mkdir(parents=True, exist_ok=True)

    current_strategy_id = require_current_production_strategy_id(
        registry_path=registry_path,
        project_dir=project_dir,
    )
    requested_ids = {str(item) for item in (strategy_ids or []) if str(item).strip()}
    if requested_ids and requested_ids != {current_strategy_id}:
        raise ValueError(
            "L7 synchronization strategy_ids must exactly match production.current: "
            f"current={current_strategy_id}, requested={sorted(requested_ids)}"
        )
    allowed_ids = {current_strategy_id}
    file_rows: list[dict[str, Any]] = []
    signal_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []

    for path in sorted(signal_root.glob("*")):
        if not path.is_file():
            continue
        strategy_id = _infer_strategy_id(path.name, allowed_ids) if allowed_ids else None
        if allowed_ids and strategy_id is None:
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
        if path.suffix.lower() == ".csv":
            signal_rows.extend(_csv_rows(path, strategy_id))
        elif path.suffix.lower() == ".json":
            status_rows.append(
                _json_status_row(
                    path,
                    strategy_id,
                    payload_overrides=status_payload_overrides,
                )
            )

    import duckdb

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
    signal_df = pd.DataFrame(
        signal_rows,
        columns=[
            "strategy_id",
            "file_name",
            "source_path",
            "row_index",
            "signal_date",
            "buy_date",
            "stock_code",
            "symbol",
            "name",
            "rank",
            "pred_prob",
            "target_pct",
            "row_payload_json",
        ],
    )
    status_df = pd.DataFrame(
        status_rows,
        columns=[
            "strategy_id",
            "file_name",
            "source_path",
            "status",
            "latest_signal_date",
            "latest_buy_date",
            "published_at",
            "payload_json",
        ],
    )
    with duckdb.connect(str(signal_files_path_target)) as conn:
        conn.register("_signal_files_df", file_df)
        conn.execute(f'CREATE OR REPLACE TABLE "{signal_files_table}" AS SELECT * FROM _signal_files_df')
        conn.unregister("_signal_files_df")
    with duckdb.connect(str(signal_rows_path_target)) as conn:
        table_exists = conn.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = 'main' AND table_name = ?
            """,
            [signal_rows_table],
        ).fetchone()[0]
        if signal_df.empty and table_exists:
            # A zero-action batch must clear the current rows without letting
            # pandas infer an all-INTEGER schema from an empty DataFrame.
            conn.execute(f'DELETE FROM "{signal_rows_table}"')
        else:
            conn.register("_signal_rows_df", signal_df)
            conn.execute(f'CREATE OR REPLACE TABLE "{signal_rows_table}" AS SELECT * FROM _signal_rows_df')
            conn.unregister("_signal_rows_df")
    with duckdb.connect(str(signal_status_path_target)) as conn:
        conn.register("_signal_status_df", status_df)
        conn.execute(f'CREATE OR REPLACE TABLE "{signal_status_table}" AS SELECT * FROM _signal_status_df')
        conn.unregister("_signal_status_df")

    return {
        "duckdb_path": str(signal_rows_path_target),
        "signal_rows_duckdb_path": str(signal_rows_path_target),
        "signal_status_duckdb_path": str(signal_status_path_target),
        "signal_files_duckdb_path": str(signal_files_path_target),
        "signal_dir": str(signal_root),
        "signal_rows_table": signal_rows_table,
        "signal_status_table": signal_status_table,
        "signal_files_table": signal_files_table,
        "signal_file_count": int(len(file_rows)),
        "signal_row_count": int(len(signal_rows)),
        "status_row_count": int(len(status_rows)),
    }
