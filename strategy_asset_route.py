from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

from duckdb_asset_route import (
    BACKEND_DUCKDB,
    BACKEND_LEGACY,
    ENV_ALLOW_LEGACY_ROLLBACK,
    connect_duckdb_readonly,
    resolve_duckdb_path,
)
from production_asset_registry import (
    active_main_workflow_asset,
    active_main_workflow_asset_for_table,
    split_asset_path,
)
from project_paths import resolve_data_path, resolve_project_path


ENV_STRATEGY_ASSET_BACKEND = "QUANT_STRATEGY_ASSET_BACKEND"
ENV_STRATEGY_DUCKDB = "QUANT_STRATEGY_DUCKDB"
ENV_STRATEGY_REGISTRY_FILE = "QUANT_STRATEGY_REGISTRY_FILE"
ENV_STRATEGY_ROOT = "QUANT_STRATEGY_ROOT"
ENV_SIGNAL_DIR = "QUANT_PRODUCTION_SIGNAL_DIR"

DEFAULT_REGISTRY_FILE = "strategy_library/registry.json"
DEFAULT_STRATEGY_ROOT = "strategy_library/production"
DEFAULT_SIGNAL_DIR = "production_signals"

DEFAULT_REGISTRY_TABLE = "prod_l5_strategy_registry_current"
DEFAULT_MANIFEST_TABLE = "prod_l5_strategy_manifest_current"
DEFAULT_VALIDATION_TABLE = "prod_l6_strategy_validation_current"
DEFAULT_SIGNAL_ROWS_TABLE = "prod_l7_signal_rows_current"
DEFAULT_SIGNAL_FILES_TABLE = "prod_l7_signal_files_current"
DEFAULT_SIGNAL_STATUS_TABLE = "prod_l7_signal_status_current"


def resolve_strategy_asset_backend(value: str | None = None, *, data_dir: str | Path | None = None) -> str:
    explicit = value if value is not None else os.environ.get(ENV_STRATEGY_ASSET_BACKEND)
    if explicit not in (None, ""):
        backend = str(explicit).strip().lower()
        if backend in {"sqlite", "parquet", "legacy_sqlite_parquet", "json_csv", "file"}:
            raise ValueError(
                "deprecated strategy asset backend alias is not allowed: "
                f"{backend}. Use 'duckdb' for current production or explicit 'legacy' only for approved rollback flows."
            )
        if backend not in {BACKEND_DUCKDB, BACKEND_LEGACY}:
            raise ValueError(f"unsupported strategy asset backend: {backend}")
        if backend == BACKEND_LEGACY and not legacy_strategy_asset_opted_in():
            raise RuntimeError(
                "explicit legacy strategy asset backend selection requires opt-in via "
                f"{ENV_ALLOW_LEGACY_ROLLBACK}=1"
            )
        return backend

    for layer in ("L7_trading_delivery", "L6_backtest", "L5_strategy_signal"):
        asset = active_main_workflow_asset(layer, data_dir=data_dir)
        asset_type = str(asset.get("asset_type", "")).lower() if asset else ""
        asset_path = str(asset.get("asset_path", "")).lower() if asset else ""
        if "duckdb" in asset_type or asset_path.endswith(".duckdb") or ".duckdb::" in asset_path:
            return BACKEND_DUCKDB
    return BACKEND_DUCKDB


def legacy_strategy_asset_opted_in() -> bool:
    return str(os.environ.get(ENV_ALLOW_LEGACY_ROLLBACK, "")).strip() == "1"


def resolve_strategy_registry_path(registry_path: str | Path | None = None) -> Path:
    explicit = registry_path or os.environ.get(ENV_STRATEGY_REGISTRY_FILE) or DEFAULT_REGISTRY_FILE
    return resolve_project_path(explicit)


def resolve_strategy_root(strategy_root: str | Path | None = None) -> Path:
    explicit = strategy_root or os.environ.get(ENV_STRATEGY_ROOT) or DEFAULT_STRATEGY_ROOT
    return resolve_project_path(explicit)


def resolve_signal_dir(signal_dir: str | Path | None = None, *, data_dir: str | Path | None = None) -> Path:
    explicit = signal_dir or os.environ.get(ENV_SIGNAL_DIR) or DEFAULT_SIGNAL_DIR
    return resolve_data_path(explicit, data_dir=data_dir)


def _resolve_active_strategy_layer_duckdb_path(
    layer: str,
    *,
    data_dir: str | Path | None = None,
    explicit_path: str | Path | None = None,
    require_exists: bool = False,
    fallback_layer: str,
    label: str,
) -> Path:
    explicit = explicit_path or os.environ.get(ENV_STRATEGY_DUCKDB)
    if explicit:
        path = Path(explicit).expanduser()
        path = path if path.is_absolute() else path.resolve()
    else:
        asset = active_main_workflow_asset(layer, data_dir=data_dir)
        asset_path, _table = split_asset_path(asset.get("asset_path")) if asset else (None, None)
        path = asset_path if asset_path and asset_path.suffix.lower() == ".duckdb" else resolve_duckdb_path(
            fallback_layer,
            data_dir=data_dir,
        )
    if require_exists and not path.is_file():
        raise FileNotFoundError(f"{label} DuckDB asset not found: {path}")
    return path


def _resolve_active_strategy_table_duckdb_path(
    layer: str,
    table: str,
    *,
    data_dir: str | Path | None = None,
    explicit_path: str | Path | None = None,
    require_exists: bool = False,
    fallback_layer: str,
    label: str,
) -> Path:
    explicit = explicit_path or os.environ.get(ENV_STRATEGY_DUCKDB)
    if explicit:
        path = Path(explicit).expanduser()
        path = path if path.is_absolute() else path.resolve()
    else:
        asset = active_main_workflow_asset_for_table(layer, table, data_dir=data_dir)
        asset_path, asset_table = split_asset_path(asset.get("asset_path")) if asset else (None, None)
        if asset_path and asset_path.suffix.lower() == ".duckdb" and str(asset_table or "").strip() == str(table).strip():
            path = asset_path
        else:
            path = _resolve_active_strategy_layer_duckdb_path(
                layer,
                data_dir=data_dir,
                explicit_path=None,
                require_exists=False,
                fallback_layer=fallback_layer,
                label=label,
            )
    if require_exists and not path.is_file():
        raise FileNotFoundError(f"{label} DuckDB asset not found: {path}")
    return path


def resolve_strategy_duckdb_path(
    data_dir: str | Path | None = None,
    duckdb_path: str | Path | None = None,
    *,
    require_exists: bool = False,
) -> Path:
    return resolve_strategy_registry_duckdb_path(
        data_dir=data_dir,
        duckdb_path=duckdb_path,
        require_exists=require_exists,
    )


def resolve_strategy_registry_duckdb_path(
    data_dir: str | Path | None = None,
    duckdb_path: str | Path | None = None,
    *,
    require_exists: bool = False,
    registry_table: str = DEFAULT_REGISTRY_TABLE,
) -> Path:
    return _resolve_active_strategy_table_duckdb_path(
        "L5_strategy_signal",
        registry_table,
        data_dir=data_dir,
        explicit_path=duckdb_path,
        require_exists=require_exists,
        fallback_layer="L5",
        label="strategy registry",
    )


def resolve_strategy_manifest_duckdb_path(
    data_dir: str | Path | None = None,
    duckdb_path: str | Path | None = None,
    *,
    require_exists: bool = False,
    manifest_table: str = DEFAULT_MANIFEST_TABLE,
) -> Path:
    return _resolve_active_strategy_table_duckdb_path(
        "L5_strategy_signal",
        manifest_table,
        data_dir=data_dir,
        explicit_path=duckdb_path,
        require_exists=require_exists,
        fallback_layer="L5",
        label="strategy manifest",
    )


def resolve_strategy_validation_duckdb_path(
    data_dir: str | Path | None = None,
    duckdb_path: str | Path | None = None,
    *,
    require_exists: bool = False,
) -> Path:
    return _resolve_active_strategy_table_duckdb_path(
        "L6_backtest",
        DEFAULT_VALIDATION_TABLE,
        data_dir=data_dir,
        explicit_path=duckdb_path,
        require_exists=require_exists,
        fallback_layer="L6",
        label="strategy validation",
    )


def resolve_strategy_signal_duckdb_path(
    data_dir: str | Path | None = None,
    duckdb_path: str | Path | None = None,
    *,
    require_exists: bool = False,
) -> Path:
    return resolve_strategy_signal_rows_duckdb_path(
        data_dir=data_dir,
        duckdb_path=duckdb_path,
        require_exists=require_exists,
    )


def resolve_strategy_signal_rows_duckdb_path(
    data_dir: str | Path | None = None,
    duckdb_path: str | Path | None = None,
    *,
    require_exists: bool = False,
    signal_rows_table: str = DEFAULT_SIGNAL_ROWS_TABLE,
) -> Path:
    return _resolve_active_strategy_table_duckdb_path(
        "L7_trading_delivery",
        signal_rows_table,
        data_dir=data_dir,
        explicit_path=duckdb_path,
        require_exists=require_exists,
        fallback_layer="L7",
        label="strategy signal rows",
    )


def resolve_strategy_signal_files_duckdb_path(
    data_dir: str | Path | None = None,
    duckdb_path: str | Path | None = None,
    *,
    require_exists: bool = False,
    signal_files_table: str = DEFAULT_SIGNAL_FILES_TABLE,
) -> Path:
    return _resolve_active_strategy_table_duckdb_path(
        "L7_trading_delivery",
        signal_files_table,
        data_dir=data_dir,
        explicit_path=duckdb_path,
        require_exists=require_exists,
        fallback_layer="L7",
        label="strategy signal files",
    )


def resolve_strategy_signal_status_duckdb_path(
    data_dir: str | Path | None = None,
    duckdb_path: str | Path | None = None,
    *,
    require_exists: bool = False,
    signal_status_table: str = DEFAULT_SIGNAL_STATUS_TABLE,
) -> Path:
    return _resolve_active_strategy_table_duckdb_path(
        "L7_trading_delivery",
        signal_status_table,
        data_dir=data_dir,
        explicit_path=duckdb_path,
        require_exists=require_exists,
        fallback_layer="L7",
        label="strategy signal status",
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_duckdb_tables(conn, *tables: str) -> None:
    existing = {str(row[0]) for row in conn.execute("SHOW TABLES").fetchall()}
    missing = [table for table in tables if table not in existing]
    if missing:
        raise RuntimeError(f"DuckDB strategy asset missing required tables: {', '.join(missing)}")


def load_production_strategy_registry(
    *,
    registry_path: str | Path | None = None,
    data_dir: str | Path | None = None,
    duckdb_path: str | Path | None = None,
    backend: str | None = None,
    registry_table: str = DEFAULT_REGISTRY_TABLE,
) -> dict[str, Any]:
    selected = resolve_strategy_asset_backend(backend, data_dir=data_dir)
    if selected == BACKEND_LEGACY:
        return _load_json(resolve_strategy_registry_path(registry_path))

    path = resolve_strategy_registry_duckdb_path(
        data_dir=data_dir,
        duckdb_path=duckdb_path,
        require_exists=True,
        registry_table=registry_table,
    )
    conn = connect_duckdb_readonly(path)
    try:
        _require_duckdb_tables(conn, registry_table)
        rows = conn.execute(
            f"""
            SELECT strategy_id, is_current_production, registry_payload_json
            FROM "{registry_table}"
            ORDER BY is_current_production DESC, strategy_id
            """
        ).fetchall()
    finally:
        conn.close()

    strategies: list[dict[str, Any]] = []
    current = ""
    for strategy_id, is_current_production, payload_json in rows:
        payload = json.loads(payload_json) if payload_json else {"strategy_id": strategy_id}
        strategies.append(payload)
        if is_current_production and not current:
            current = str(strategy_id or "")
    return {"production": {"current": current, "strategies": strategies}}


def current_production_strategy(registry: dict[str, Any], strategy_id: str | None = None) -> dict[str, Any]:
    production = registry.get("production", {})
    selected_id = strategy_id or production.get("current")
    if not selected_id:
        raise ValueError("production.current missing in strategy registry")
    for item in production.get("strategies", []):
        if item.get("strategy_id") == selected_id:
            return item
    raise ValueError(f"strategy_id not found in production registry: {selected_id}")


def load_strategy_manifest_payload(
    strategy_id: str,
    *,
    strategy_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
    duckdb_path: str | Path | None = None,
    backend: str | None = None,
    manifest_table: str = DEFAULT_MANIFEST_TABLE,
) -> dict[str, Any]:
    selected = resolve_strategy_asset_backend(backend, data_dir=data_dir)
    if selected == BACKEND_LEGACY:
        if strategy_dir in (None, ""):
            raise ValueError("legacy strategy manifest loading requires strategy_dir")
        path = Path(strategy_dir) / "strategy_manifest.json"
        return _load_json(path)

    path = resolve_strategy_manifest_duckdb_path(
        data_dir=data_dir,
        duckdb_path=duckdb_path,
        require_exists=True,
        manifest_table=manifest_table,
    )
    conn = connect_duckdb_readonly(path)
    try:
        _require_duckdb_tables(conn, manifest_table)
        row = conn.execute(
            f"""
            SELECT manifest_payload_json
            FROM "{manifest_table}"
            WHERE strategy_id = ?
            LIMIT 1
            """,
            [strategy_id],
        ).fetchone()
    finally:
        conn.close()
    if not row or not row[0]:
        raise FileNotFoundError(f"strategy manifest not found for {strategy_id} in DuckDB asset")
    return json.loads(row[0])


def load_strategy_validation_payload(
    strategy_id: str,
    *,
    strategy_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
    duckdb_path: str | Path | None = None,
    backend: str | None = None,
    validation_table: str = DEFAULT_VALIDATION_TABLE,
) -> dict[str, Any]:
    selected = resolve_strategy_asset_backend(backend, data_dir=data_dir)
    if selected == BACKEND_LEGACY:
        if strategy_dir in (None, ""):
            raise ValueError("legacy strategy validation loading requires strategy_dir")
        path = Path(strategy_dir) / "validation.json"
        return _load_json(path)

    path = resolve_strategy_validation_duckdb_path(
        data_dir=data_dir,
        duckdb_path=duckdb_path,
        require_exists=True,
    )
    conn = connect_duckdb_readonly(path)
    try:
        _require_duckdb_tables(conn, validation_table)
        row = conn.execute(
            f"""
            SELECT validation_payload_json
            FROM "{validation_table}"
            WHERE strategy_id = ?
            LIMIT 1
            """,
            [strategy_id],
        ).fetchone()
    finally:
        conn.close()
    if not row or not row[0]:
        raise FileNotFoundError(f"strategy validation not found for {strategy_id} in DuckDB asset")
    return json.loads(row[0])


def resolve_latest_signal_file(
    strategy_id: str,
    *,
    signal_dir: str | Path | None = None,
    strategy_dir: str | Path | None = None,
) -> Path:
    candidates: list[Path] = []
    if signal_dir is not None:
        candidates.append(resolve_signal_dir(signal_dir) / f"{strategy_id}_latest.csv")
    if strategy_dir is not None:
        base = Path(strategy_dir)
        candidates.extend(
            [
                base / "signals" / "signals_latest.csv",
                base / "signals" / "production_signals.csv",
            ]
        )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"latest signal file not found for {strategy_id}; checked: "
        + ", ".join(str(path) for path in candidates)
    )


def load_latest_signal_batch(signal_path: str | Path) -> list[dict[str, str]]:
    path = Path(signal_path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"signal file is empty: {path}")
    latest_signal_date = max((row.get("signal_date") or "") for row in rows)
    latest_rows = [row for row in rows if (row.get("signal_date") or "") == latest_signal_date]
    latest_rows.sort(key=lambda row: int(row.get("rank") or "999999"))
    return latest_rows


def load_latest_signal_rows(
    strategy_id: str,
    *,
    signal_dir: str | Path | None = None,
    strategy_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
    duckdb_path: str | Path | None = None,
    backend: str | None = None,
    signal_rows_table: str = DEFAULT_SIGNAL_ROWS_TABLE,
    signal_files_table: str = DEFAULT_SIGNAL_FILES_TABLE,
) -> tuple[list[dict[str, Any]], str]:
    selected = resolve_strategy_asset_backend(backend, data_dir=data_dir)
    if selected == BACKEND_LEGACY:
        path = resolve_latest_signal_file(strategy_id, signal_dir=signal_dir, strategy_dir=strategy_dir)
        return load_latest_signal_batch(path), str(path)

    rows_path = resolve_strategy_signal_rows_duckdb_path(
        data_dir=data_dir,
        duckdb_path=duckdb_path,
        require_exists=True,
        signal_rows_table=signal_rows_table,
    )
    files_path = resolve_strategy_signal_files_duckdb_path(
        data_dir=data_dir,
        duckdb_path=duckdb_path,
        require_exists=True,
        signal_files_table=signal_files_table,
    )
    latest_file_conn = connect_duckdb_readonly(files_path)
    try:
        _require_duckdb_tables(latest_file_conn, signal_files_table)
        latest_file_row = latest_file_conn.execute(
            f"""
            SELECT source_path
            FROM "{signal_files_table}"
            WHERE strategy_id = ?
              AND lower(file_name) LIKE '%latest.csv'
            ORDER BY modified_at DESC, file_name DESC
            LIMIT 1
            """,
            [strategy_id],
        ).fetchone()
    finally:
        latest_file_conn.close()

    rows_conn = connect_duckdb_readonly(rows_path)
    try:
        _require_duckdb_tables(rows_conn, signal_rows_table)
        latest_signal_date_row = rows_conn.execute(
            f"""
            SELECT max(signal_date)
            FROM "{signal_rows_table}"
            WHERE strategy_id = ?
              AND lower(file_name) LIKE '%latest.csv'
            """,
            [strategy_id],
        ).fetchone()
        latest_signal_date = latest_signal_date_row[0] if latest_signal_date_row else None
        if latest_signal_date:
            fetched = rows_conn.execute(
                f"""
                SELECT row_payload_json
                FROM "{signal_rows_table}"
                WHERE strategy_id = ?
                  AND lower(file_name) LIKE '%latest.csv'
                  AND signal_date = ?
                ORDER BY coalesce(try_cast(rank AS BIGINT), 999999), row_index
                """,
                [strategy_id, latest_signal_date],
            ).fetchall()
        else:
            fetched = rows_conn.execute(
                f"""
                SELECT row_payload_json
                FROM "{signal_rows_table}"
                WHERE strategy_id = ?
                ORDER BY coalesce(signal_date, ''), coalesce(try_cast(rank AS BIGINT), 999999), row_index
                """,
                [strategy_id],
            ).fetchall()
    finally:
        rows_conn.close()

    rows = [json.loads(row[0]) for row in fetched if row and row[0]]
    if not rows:
        raise FileNotFoundError(f"latest signal rows not found for {strategy_id} in DuckDB asset")
    source = (
        str(latest_file_row[0])
        if latest_file_row and latest_file_row[0]
        else f"{rows_path}::{signal_rows_table}[strategy_id={strategy_id}]"
    )
    return rows, source


def load_current_production_strategy_context(
    *,
    strategy_id: str | None = None,
    registry_file: str | Path = DEFAULT_REGISTRY_FILE,
    signal_dir: str | Path = DEFAULT_SIGNAL_DIR,
    strategy_root: str | Path = DEFAULT_STRATEGY_ROOT,
    data_dir: str | Path | None = None,
    duckdb_path: str | Path | None = None,
    backend: str | None = None,
) -> dict[str, Any]:
    registry = load_production_strategy_registry(
        registry_path=registry_file,
        data_dir=data_dir,
        duckdb_path=duckdb_path,
        backend=backend,
    )
    strategy_entry = current_production_strategy(registry, strategy_id)
    resolved_strategy_dir = resolve_project_path(Path(strategy_root) / strategy_entry["strategy_id"])
    manifest = load_strategy_manifest_payload(
        str(strategy_entry["strategy_id"]),
        strategy_dir=resolved_strategy_dir,
        data_dir=data_dir,
        duckdb_path=duckdb_path,
        backend=backend,
    )
    signal_rows, signal_source = load_latest_signal_rows(
        str(strategy_entry["strategy_id"]),
        signal_dir=signal_dir,
        strategy_dir=resolved_strategy_dir,
        data_dir=data_dir,
        duckdb_path=duckdb_path,
        backend=backend,
    )
    return {
        "backend": resolve_strategy_asset_backend(backend, data_dir=data_dir),
        "registry": registry,
        "strategy_entry": strategy_entry,
        "strategy_manifest": manifest,
        "strategy_dir": str(resolved_strategy_dir),
        "signal_path": signal_source,
        "signal_rows": signal_rows,
    }
