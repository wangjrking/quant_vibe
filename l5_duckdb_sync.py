from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from duckdb_asset_route import resolve_duckdb_path
from production_asset_registry import active_main_workflow_asset, active_main_workflow_asset_for_table, split_asset_path
from project_paths import PROJECT_ROOT, resolve_data_dir


ENV_L5_DUCKDB_SYNC = "QUANT_L5_DUCKDB_SYNC"
DEFAULT_REGISTRY_TABLE = "prod_l5_strategy_registry_current"
DEFAULT_MANIFEST_TABLE = "prod_l5_strategy_manifest_current"


def _default_l5_table_duckdb_path(table_name: str, data_dir: str | Path | None = None) -> Path:
    base = resolve_data_dir(data_dir) / "production_assets" / "duckdb" / "production" / "l5"
    filename_by_table = {
        DEFAULT_REGISTRY_TABLE: "prod_l5_strategy_registry_current.duckdb",
        DEFAULT_MANIFEST_TABLE: "prod_l5_strategy_manifest_current.duckdb",
    }
    filename = filename_by_table.get(str(table_name).strip())
    if filename:
        return base / filename
    raise ValueError(f"unsupported L5 current table for split DuckDB route: {table_name}")


def l5_duckdb_sync_enabled() -> bool:
    value = str(os.environ.get(ENV_L5_DUCKDB_SYNC, "1")).strip().lower()
    return value not in {"0", "false", "no", "off"}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_current_signal(current_signal: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(current_signal, dict):
        return {}
    signal_date = current_signal.get("latest_signal_date") or current_signal.get("signal_date")
    buy_date = current_signal.get("latest_buy_date") or current_signal.get("buy_date")
    normalized = dict(current_signal)
    normalized["latest_signal_date"] = signal_date
    normalized["latest_buy_date"] = buy_date
    return normalized


def _resolve_strategy_dir(project_dir: Path, registry_item: dict[str, Any]) -> Path | None:
    raw = registry_item.get("path")
    if raw in (None, ""):
        return None
    path = Path(str(raw)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (project_dir / path).resolve()


def resolve_l5_duckdb_path(
    data_dir: str | Path | None = None,
    *,
    duckdb_path: str | Path | None = None,
    require_exists: bool = False,
) -> Path:
    return resolve_l5_table_duckdb_path(
        DEFAULT_REGISTRY_TABLE,
        data_dir=data_dir,
        duckdb_path=duckdb_path,
        require_exists=require_exists,
    )


def resolve_l5_table_duckdb_path(
    table_name: str,
    data_dir: str | Path | None = None,
    *,
    duckdb_path: str | Path | None = None,
    require_exists: bool = False,
) -> Path:
    if duckdb_path:
        path = Path(duckdb_path)
    else:
        asset = active_main_workflow_asset_for_table("L5_strategy_signal", table_name, data_dir=data_dir)
        asset_path, asset_table = split_asset_path(asset.get("asset_path")) if asset else (None, None)
        if asset_path and asset_path.suffix.lower() == ".duckdb" and str(asset_table or "").strip() == str(table_name).strip():
            path = asset_path
        else:
            asset = active_main_workflow_asset("L5_strategy_signal", data_dir=data_dir)
            asset_path, _table = split_asset_path(asset.get("asset_path")) if asset else (None, None)
            path = (
                asset_path
                if asset_path and asset_path.suffix.lower() == ".duckdb"
                else _default_l5_table_duckdb_path(table_name, data_dir=data_dir)
            )
    if require_exists and not path.is_file():
        raise FileNotFoundError(f"L5 DuckDB asset not found: {path}")
    return path


def sync_strategy_registry_to_duckdb(
    *,
    project_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
    registry_path: str | Path | None = None,
    duckdb_path: str | Path | None = None,
    registry_table: str = DEFAULT_REGISTRY_TABLE,
    manifest_table: str = DEFAULT_MANIFEST_TABLE,
) -> dict[str, int | str]:
    project_base = Path(project_dir).resolve() if project_dir else PROJECT_ROOT
    raw_registry_path = Path(registry_path) if registry_path is not None else (project_base / "strategy_library" / "registry.json")
    registry_file = raw_registry_path.resolve() if raw_registry_path.is_absolute() else (project_base / raw_registry_path).resolve()
    registry = _load_json(registry_file)
    production = registry.get("production", {}) or {}
    current_strategy_id = str(production.get("current") or "")
    strategies = list(production.get("strategies", []) or [])

    registry_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    for item in strategies:
        strategy_id = str(item.get("strategy_id") or "")
        strategy_dir = _resolve_strategy_dir(project_base, item)
        registry_rows.append(
            {
                "strategy_id": strategy_id,
                "name": item.get("name"),
                "registry_status": item.get("status"),
                "is_current_production": strategy_id == current_strategy_id,
                "strategy_path": str(strategy_dir) if strategy_dir else None,
                "published_at": item.get("published_at"),
                "annual_return": item.get("annual_return"),
                "sharpe": item.get("sharpe"),
                "max_drawdown": item.get("max_drawdown"),
                "validation_platform": item.get("validation_platform"),
                "registry_payload_json": json.dumps(item, ensure_ascii=False),
            }
        )
        if not strategy_dir:
            continue
        manifest_path = strategy_dir / "strategy_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = _load_json(manifest_path)
        current_signal = _normalize_current_signal(manifest.get("current_signal"))
        input_contract = manifest.get("input_contract", {}) if isinstance(manifest.get("input_contract"), dict) else {}
        validation = manifest.get("validation", {}) if isinstance(manifest.get("validation"), dict) else {}
        formal_manifest_1d = input_contract.get("formal_manifest_1d")
        formal_manifest_current = input_contract.get("formal_manifest")
        manifest_rows.append(
            {
                "strategy_id": strategy_id,
                "name": manifest.get("name"),
                "manifest_status": manifest.get("status"),
                "strategy_dir": str(strategy_dir),
                "manifest_path": str(manifest_path),
                "is_current_production": strategy_id == current_strategy_id,
                "published_at": manifest.get("published_at"),
                "production_version": manifest.get("production_version"),
                "latest_signal_file": current_signal.get("latest_file"),
                "latest_signal_date": current_signal.get("latest_signal_date"),
                "latest_buy_date": current_signal.get("latest_buy_date"),
                "formal_manifest": formal_manifest_current or formal_manifest_1d,
                "formal_manifest_3d": input_contract.get("formal_manifest_3d"),
                "formal_manifest_5d": input_contract.get("formal_manifest_5d"),
                "formal_manifest_10d": input_contract.get("formal_manifest_10d"),
                "annual_return": validation.get("annual_return"),
                "sharpe": validation.get("sharpe"),
                "max_drawdown": validation.get("max_drawdown"),
                "manifest_payload_json": json.dumps(manifest, ensure_ascii=False),
            }
        )

    registry_path_target = resolve_l5_table_duckdb_path(
        registry_table,
        data_dir=data_dir,
        duckdb_path=duckdb_path,
    )
    manifest_path_target = resolve_l5_table_duckdb_path(
        manifest_table,
        data_dir=data_dir,
        duckdb_path=duckdb_path if duckdb_path else None,
    )
    registry_path_target.parent.mkdir(parents=True, exist_ok=True)
    manifest_path_target.parent.mkdir(parents=True, exist_ok=True)

    import duckdb

    registry_df = pd.DataFrame(
        registry_rows,
        columns=[
            "strategy_id",
            "name",
            "registry_status",
            "is_current_production",
            "strategy_path",
            "published_at",
            "annual_return",
            "sharpe",
            "max_drawdown",
            "validation_platform",
            "registry_payload_json",
        ],
    )
    manifest_df = pd.DataFrame(
        manifest_rows,
        columns=[
            "strategy_id",
            "name",
            "manifest_status",
            "strategy_dir",
            "manifest_path",
            "is_current_production",
            "published_at",
            "production_version",
            "latest_signal_file",
            "latest_signal_date",
            "latest_buy_date",
            "formal_manifest",
            "formal_manifest_3d",
            "formal_manifest_5d",
            "formal_manifest_10d",
            "annual_return",
            "sharpe",
            "max_drawdown",
            "manifest_payload_json",
        ],
    )
    with duckdb.connect(str(registry_path_target)) as conn:
        conn.register("_registry_df", registry_df)
        conn.execute(f'CREATE OR REPLACE TABLE "{registry_table}" AS SELECT * FROM _registry_df')
        conn.unregister("_registry_df")
    with duckdb.connect(str(manifest_path_target)) as conn:
        conn.register("_manifest_df", manifest_df)
        conn.execute(f'CREATE OR REPLACE TABLE "{manifest_table}" AS SELECT * FROM _manifest_df')
        conn.unregister("_manifest_df")

    return {
        "duckdb_path": str(registry_path_target),
        "registry_duckdb_path": str(registry_path_target),
        "manifest_duckdb_path": str(manifest_path_target),
        "registry_path": str(registry_file),
        "registry_table": registry_table,
        "manifest_table": manifest_table,
        "registry_rows": int(len(registry_rows)),
        "manifest_rows": int(len(manifest_rows)),
        "current_production_count": int(sum(1 for row in registry_rows if row["is_current_production"])),
    }
