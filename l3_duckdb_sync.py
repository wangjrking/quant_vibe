from __future__ import annotations

import os
from pathlib import Path

import pyarrow.parquet as pq

from adjustment_semantics import (
    FRONT_ADJUSTED_MARKET_PRICE_COLUMNS,
    NAKED_MARKET_PRICE_COLUMNS,
    prefer_explicit_qfq_columns,
)
from duckdb_asset_route import resolve_duckdb_path
from gtja_alpha_workflow import GTJA_ALPHA_COLUMNS
from l3_active_writer_lease import require_active_l3_writer_lease_for_path
from production_asset_registry import active_main_workflow_asset, split_asset_path
from project_paths import resolve_data_dir
from stock_daily_data_route import STOCK_DAILY_TABLE, resolve_stock_daily_duckdb_path


ENV_L3_DUCKDB_SYNC = "QUANT_L3_DUCKDB_SYNC"
ENV_L3_FEATURE_DUCKDB_TABLE = "QUANT_L3_FEATURE_DUCKDB_TABLE"
ENV_L3_LABEL_DUCKDB_TABLE = "QUANT_L3_LABEL_DUCKDB_TABLE"

DEFAULT_FEATURE_TABLE = "prod_l3_production_factor_parts_20260625"
DEFAULT_LABEL_TABLE = "prod_l3_prediction_label_parts_current"
GTJA_QFQ_COLUMN_MAP = {column: f"{column}_qfq" for column in GTJA_ALPHA_COLUMNS}


def l3_duckdb_sync_enabled() -> bool:
    value = str(os.environ.get(ENV_L3_DUCKDB_SYNC, "1")).strip().lower()
    return value not in {"0", "false", "no", "off"}


def _quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _connect_writable(path: Path):
    import duckdb

    path.parent.mkdir(parents=True, exist_ok=True)
    require_active_l3_writer_lease_for_path(path)
    return duckdb.connect(str(path))


def _resolve_default_table(
    layer: str,
    *,
    data_dir: str | Path | None,
    env_name: str,
    fallback: str,
) -> str:
    explicit = str(os.environ.get(env_name, "")).strip()
    if explicit:
        return explicit
    asset = active_main_workflow_asset(layer, data_dir=data_dir)
    if asset:
        _, table = split_asset_path(asset.get("asset_path"))
        if table:
            return table
        asset_id = str(asset.get("asset_id", "")).strip()
        if asset_id:
            return asset_id
    return fallback


def resolve_l3_feature_duckdb_table(
    data_dir: str | Path | None = None,
    *,
    table_name: str | None = None,
) -> str:
    return str(table_name or _resolve_default_table(
        "l3_features",
        data_dir=data_dir,
        env_name=ENV_L3_FEATURE_DUCKDB_TABLE,
        fallback=DEFAULT_FEATURE_TABLE,
    )).strip()


def resolve_l3_label_duckdb_table(
    data_dir: str | Path | None = None,
    *,
    table_name: str | None = None,
) -> str:
    return str(table_name or _resolve_default_table(
        "l3_labels",
        data_dir=data_dir,
        env_name=ENV_L3_LABEL_DUCKDB_TABLE,
        fallback=DEFAULT_LABEL_TABLE,
    )).strip()


def resolve_l3_feature_duckdb_path(
    data_dir: str | Path | None = None,
    *,
    duckdb_path: str | Path | None = None,
    require_exists: bool = False,
) -> Path:
    if duckdb_path:
        path = Path(duckdb_path)
    else:
        asset = active_main_workflow_asset("L3_features", data_dir=data_dir)
        asset_path, _table = split_asset_path(asset.get("asset_path")) if asset else (None, None)
        path = asset_path if asset_path and asset_path.suffix.lower() == ".duckdb" else resolve_duckdb_path("L3_FEATURES", data_dir=data_dir)
    if require_exists and not path.is_file():
        raise FileNotFoundError(f"L3 feature DuckDB asset not found: {path}")
    return path


def resolve_l3_label_duckdb_path(
    data_dir: str | Path | None = None,
    *,
    duckdb_path: str | Path | None = None,
    require_exists: bool = False,
) -> Path:
    if duckdb_path:
        path = Path(duckdb_path)
    else:
        asset = active_main_workflow_asset("L3_labels", data_dir=data_dir)
        asset_path, _table = split_asset_path(asset.get("asset_path")) if asset else (None, None)
        path = asset_path if asset_path and asset_path.suffix.lower() == ".duckdb" else resolve_duckdb_path("L3_LABELS", data_dir=data_dir)
    if require_exists and not path.is_file():
        raise FileNotFoundError(f"L3 label DuckDB asset not found: {path}")
    return path


def _parquet_glob(parts_dir: Path) -> str:
    if not parts_dir.exists():
        raise FileNotFoundError(parts_dir)
    if not any(parts_dir.glob("*.parquet")):
        raise RuntimeError(f"no parquet parts found: {parts_dir}")
    return str(parts_dir / "*.parquet").replace("\\", "/").replace("'", "''")


def _feature_schema_columns(parts_dir: Path) -> list[str]:
    first_part = next(iter(sorted(parts_dir.glob("*.parquet"))), None)
    if first_part is None:
        raise RuntimeError(f"no parquet parts found: {parts_dir}")
    return list(pq.ParquetFile(first_part).schema.names)


def _quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _feature_projection_columns(schema_columns: list[str]) -> list[str]:
    excluded = set(NAKED_MARKET_PRICE_COLUMNS) | set(FRONT_ADJUSTED_MARKET_PRICE_COLUMNS)
    preferred = prefer_explicit_qfq_columns(schema_columns)
    return [column for column in preferred if column not in excluded]


def _feature_select_exprs(schema_columns: list[str]) -> list[str]:
    select_exprs: list[str] = []
    for column in _feature_projection_columns(schema_columns):
        alias = GTJA_QFQ_COLUMN_MAP.get(column, column)
        select_exprs.append(f'p.{_quote_ident(column)} AS {_quote_ident(alias)}')
    return select_exprs


def _feature_sync_output_columns(schema_columns: list[str]) -> list[str]:
    output_columns = [GTJA_QFQ_COLUMN_MAP.get(column, column) for column in _feature_projection_columns(schema_columns)]
    output_columns.extend(FRONT_ADJUSTED_MARKET_PRICE_COLUMNS)
    return output_columns


def _migrate_feature_table_qfq_schema(conn, table_name: str, required_columns: list[str]) -> list[str]:
    existing_columns = set(_table_columns(conn, table_name))
    rename_map = {
        **GTJA_QFQ_COLUMN_MAP,
        "macdsignal": "macdsignal_qfq",
        "macdhist": "macdhist_qfq",
    }
    for old_name, new_name in rename_map.items():
        if old_name in existing_columns and new_name not in existing_columns:
            conn.execute(
                f"ALTER TABLE {_quote_ident(table_name)} RENAME COLUMN {_quote_ident(old_name)} TO {_quote_ident(new_name)}"
            )
            existing_columns.remove(old_name)
            existing_columns.add(new_name)

    for column in required_columns:
        if column not in existing_columns:
            conn.execute(f"ALTER TABLE {_quote_ident(table_name)} ADD COLUMN {_quote_ident(column)} DOUBLE")
            existing_columns.add(column)
    return _table_columns(conn, table_name)


def _build_feature_sync_select_sql(
    parquet_glob: str,
    schema_columns: list[str],
    *,
    l2_alias: str,
    target_date: str | None = None,
) -> str:
    projected_columns = _feature_projection_columns(schema_columns)
    select_exprs = _feature_select_exprs(schema_columns)
    select_exprs.extend(
        f'm.{_quote_ident(column)} AS {_quote_ident(column)}'
        for column in FRONT_ADJUSTED_MARKET_PRICE_COLUMNS
    )
    where_clauses = ["p.stock_code NOT LIKE '%.BJ'"]
    if target_date is not None:
        where_clauses.append(f"p.trade_date = {_quote_literal(target_date)}")
        where_clauses.append(f"m.trade_date = {_quote_literal(target_date)}")
    where_sql = " AND ".join(where_clauses)
    return (
        "SELECT "
        + ", ".join(select_exprs)
        + f" FROM read_parquet('{parquet_glob}', union_by_name=true) AS p "
        + f"LEFT JOIN {l2_alias}.{_quote_ident(STOCK_DAILY_TABLE)} AS m "
        + "ON p.stock_code = m.stock_code AND p.trade_date = m.trade_date "
        + f"WHERE {where_sql}"
    )


def _sync_feature_parts_with_l2_contract(
    data_dir: str | Path | None,
    parts_dir: Path,
    *,
    table_name: str,
    duckdb_path: str | Path | None = None,
    target_date: str | None = None,
) -> dict[str, int | str]:
    target_path = resolve_l3_feature_duckdb_path(data_dir=data_dir, duckdb_path=duckdb_path)
    l2_path = resolve_stock_daily_duckdb_path(data_dir=data_dir, require_exists=True)
    parquet_glob = _parquet_glob(parts_dir)
    schema_columns = _feature_schema_columns(parts_dir)
    select_sql = _build_feature_sync_select_sql(
        parquet_glob,
        schema_columns,
        l2_alias="l2src",
        target_date=target_date,
    )

    with _connect_writable(target_path) as conn:
        conn.execute(f"ATTACH {_quote_literal(str(l2_path))} AS l2src (READ_ONLY)")
        source_rows = int(conn.execute(f"SELECT COUNT(*) FROM ({select_sql}) AS src").fetchone()[0])
        if target_date is None:
            conn.execute(
                f"CREATE OR REPLACE TABLE {_quote_ident(table_name)} AS {select_sql}"
            )
            target_rows = int(conn.execute(f"SELECT COUNT(*) FROM {_quote_ident(table_name)}").fetchone()[0])
            return {
                "parts_dir": str(parts_dir),
                "duckdb_path": str(target_path),
                "table_name": table_name,
                "source_rows": source_rows,
                "target_rows": target_rows,
            }
        if not _table_exists(conn, table_name):
            conn.execute(
                f"CREATE TABLE {_quote_ident(table_name)} AS {select_sql}"
            )
        else:
            output_columns = _feature_sync_output_columns(schema_columns)
            existing_columns = set(_migrate_feature_table_qfq_schema(conn, table_name, output_columns))
            insert_columns = [column for column in output_columns if column in existing_columns]
            missing_in_target = [
                column for column in output_columns if column not in existing_columns
            ]
            if missing_in_target:
                raise RuntimeError(
                    f"target table {_quote_ident(table_name)} is missing feature columns required by source parts: "
                    f"{missing_in_target[:20]}"
                )
            conn.execute(
                f"DELETE FROM {_quote_ident(table_name)} WHERE trade_date = ?",
                [target_date],
            )
            conn.execute(
                f"INSERT INTO {_quote_ident(table_name)} "
                f"({', '.join(_quote_ident(column) for column in insert_columns)}) "
                f"{select_sql}"
            )
        target_rows = int(
            conn.execute(
                f"SELECT COUNT(*) FROM {_quote_ident(table_name)} WHERE trade_date = ?",
                [target_date],
            ).fetchone()[0]
        )
        total_rows = int(conn.execute(f"SELECT COUNT(*) FROM {_quote_ident(table_name)}").fetchone()[0])
        return {
            "parts_dir": str(parts_dir),
            "duckdb_path": str(target_path),
            "table_name": table_name,
            "target_date": target_date,
            "source_rows": source_rows,
            "target_rows": target_rows,
            "target_total_rows": total_rows,
        }


def _table_exists(conn, table_name: str) -> bool:
    return bool(
        conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            [table_name],
        ).fetchone()[0]
    )


def _table_columns(conn, table_name: str) -> list[str]:
    return [str(row[1]) for row in conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()]


def _rewrite_existing_feature_table_with_l2_contract(
    data_dir: str | Path | None,
    *,
    table_name: str,
    duckdb_path: str | Path | None = None,
) -> dict[str, int | str] | None:
    target_path = resolve_l3_feature_duckdb_path(data_dir=data_dir, duckdb_path=duckdb_path)
    if not target_path.exists():
        return None
    l2_path = resolve_stock_daily_duckdb_path(data_dir=data_dir, require_exists=True)

    with _connect_writable(target_path) as conn:
        if not _table_exists(conn, table_name):
            return None
        schema_columns = _table_columns(conn, table_name)
        select_exprs = _feature_select_exprs(schema_columns)
        select_exprs.extend(
            f'm.{_quote_ident(column)} AS {_quote_ident(column)}'
            for column in FRONT_ADJUSTED_MARKET_PRICE_COLUMNS
        )
        temp_table = f"{table_name}__qfq_rewrite_tmp"
        conn.execute(f"ATTACH {_quote_literal(str(l2_path))} AS l2src (READ_ONLY)")
        source_rows = int(conn.execute(f"SELECT COUNT(*) FROM {_quote_ident(table_name)}").fetchone()[0])
        conn.execute(f"DROP TABLE IF EXISTS {_quote_ident(temp_table)}")
        conn.execute(
            f"CREATE TABLE {_quote_ident(temp_table)} AS "
            + "SELECT "
            + ", ".join(select_exprs)
            + f" FROM {_quote_ident(table_name)} AS p "
            + f"LEFT JOIN l2src.{_quote_ident(STOCK_DAILY_TABLE)} AS m "
            + "ON p.stock_code = m.stock_code AND p.trade_date = m.trade_date "
            + "WHERE p.stock_code NOT LIKE '%.BJ'"
        )
        conn.execute(f"DROP TABLE {_quote_ident(table_name)}")
        conn.execute(
            f"ALTER TABLE {_quote_ident(temp_table)} RENAME TO {_quote_ident(table_name)}"
        )
        target_rows = int(conn.execute(f"SELECT COUNT(*) FROM {_quote_ident(table_name)}").fetchone()[0])

    return {
        "duckdb_path": str(target_path),
        "table_name": table_name,
        "source_rows": source_rows,
        "target_rows": target_rows,
        "mode": "rewrite_existing_duckdb_table",
    }


def sync_parquet_parts_full_to_duckdb(
    data_dir: str | Path | None,
    parts_dir: str | Path,
    *,
    table_name: str,
    duckdb_path: str | Path | None = None,
) -> dict[str, int | str]:
    target_path = resolve_l3_feature_duckdb_path(data_dir=data_dir, duckdb_path=duckdb_path)
    parts_path = Path(parts_dir)
    parquet_glob = _parquet_glob(parts_path)

    with _connect_writable(target_path) as conn:
        source_rows = int(
            conn.execute(
                f"SELECT COUNT(*) FROM read_parquet('{parquet_glob}', union_by_name=true)"
            ).fetchone()[0]
        )
        conn.execute(
            f"CREATE OR REPLACE TABLE {_quote_ident(table_name)} AS "
            f"SELECT * FROM read_parquet('{parquet_glob}', union_by_name=true)"
        )
        target_rows = int(conn.execute(f"SELECT COUNT(*) FROM {_quote_ident(table_name)}").fetchone()[0])

    return {
        "parts_dir": str(parts_path),
        "duckdb_path": str(target_path),
        "table_name": table_name,
        "source_rows": source_rows,
        "target_rows": target_rows,
    }


def sync_parquet_parts_trade_date_to_duckdb(
    data_dir: str | Path | None,
    parts_dir: str | Path,
    *,
    table_name: str,
    target_date: str,
    trade_col: str = "trade_date",
    duckdb_path: str | Path | None = None,
) -> dict[str, int | str]:
    target_path = resolve_l3_feature_duckdb_path(data_dir=data_dir, duckdb_path=duckdb_path)
    parts_path = Path(parts_dir)
    parquet_glob = _parquet_glob(parts_path)

    with _connect_writable(target_path) as conn:
        source_rows = int(
            conn.execute(
                f"SELECT COUNT(*) FROM read_parquet('{parquet_glob}', union_by_name=true) "
                f"WHERE {_quote_ident(trade_col)} = ?",
                [target_date],
            ).fetchone()[0]
        )
        if not _table_exists(conn, table_name):
            conn.execute(
                f"CREATE TABLE {_quote_ident(table_name)} AS "
                f"SELECT * FROM read_parquet('{parquet_glob}', union_by_name=true) "
                f"WHERE {_quote_ident(trade_col)} = ?",
                [target_date],
            )
        else:
            conn.execute(
                f"DELETE FROM {_quote_ident(table_name)} WHERE {_quote_ident(trade_col)} = ?",
                [target_date],
            )
            conn.execute(
                f"INSERT INTO {_quote_ident(table_name)} "
                f"SELECT * FROM read_parquet('{parquet_glob}', union_by_name=true) "
                f"WHERE {_quote_ident(trade_col)} = ?",
                [target_date],
            )
        target_rows = int(
            conn.execute(
                f"SELECT COUNT(*) FROM {_quote_ident(table_name)} WHERE {_quote_ident(trade_col)} = ?",
                [target_date],
            ).fetchone()[0]
        )
        total_rows = int(conn.execute(f"SELECT COUNT(*) FROM {_quote_ident(table_name)}").fetchone()[0])

    return {
        "parts_dir": str(parts_path),
        "duckdb_path": str(target_path),
        "table_name": table_name,
        "target_date": target_date,
        "source_rows": source_rows,
        "target_rows": target_rows,
        "target_total_rows": total_rows,
    }


def _sync_label_parts_full_to_duckdb(
    data_dir: str | Path | None,
    parts_dir: str | Path,
    *,
    table_name: str,
    duckdb_path: str | Path | None = None,
) -> dict[str, int | str]:
    target_path = resolve_l3_label_duckdb_path(data_dir=data_dir, duckdb_path=duckdb_path)
    parts_path = Path(parts_dir)
    parquet_glob = _parquet_glob(parts_path)

    with _connect_writable(target_path) as conn:
        source_rows = int(
            conn.execute(
                f"SELECT COUNT(*) FROM read_parquet('{parquet_glob}', union_by_name=true) "
                "WHERE stock_code NOT LIKE '%.BJ'"
            ).fetchone()[0]
        )
        conn.execute(
            f"CREATE OR REPLACE TABLE {_quote_ident(table_name)} AS "
            f"SELECT * FROM read_parquet('{parquet_glob}', union_by_name=true) "
            "WHERE stock_code NOT LIKE '%.BJ'"
        )
        target_rows = int(conn.execute(f"SELECT COUNT(*) FROM {_quote_ident(table_name)}").fetchone()[0])

    return {
        "parts_dir": str(parts_path),
        "duckdb_path": str(target_path),
        "table_name": table_name,
        "source_rows": source_rows,
        "target_rows": target_rows,
    }


def _sync_label_parts_trade_date_to_duckdb(
    data_dir: str | Path | None,
    parts_dir: str | Path,
    *,
    table_name: str,
    target_date: str,
    duckdb_path: str | Path | None = None,
) -> dict[str, int | str]:
    target_path = resolve_l3_label_duckdb_path(data_dir=data_dir, duckdb_path=duckdb_path)
    parts_path = Path(parts_dir)
    parquet_glob = _parquet_glob(parts_path)

    with _connect_writable(target_path) as conn:
        source_rows = int(
            conn.execute(
                f"SELECT COUNT(*) FROM read_parquet('{parquet_glob}', union_by_name=true) "
                "WHERE trade_date = ? AND stock_code NOT LIKE '%.BJ'",
                [target_date],
            ).fetchone()[0]
        )
        if not _table_exists(conn, table_name):
            conn.execute(
                f"CREATE TABLE {_quote_ident(table_name)} AS "
                f"SELECT * FROM read_parquet('{parquet_glob}', union_by_name=true) "
                "WHERE trade_date = ? AND stock_code NOT LIKE '%.BJ'",
                [target_date],
            )
        else:
            conn.execute(
                f"DELETE FROM {_quote_ident(table_name)} WHERE trade_date = ?",
                [target_date],
            )
            conn.execute(
                f"INSERT INTO {_quote_ident(table_name)} "
                f"SELECT * FROM read_parquet('{parquet_glob}', union_by_name=true) "
                "WHERE trade_date = ? AND stock_code NOT LIKE '%.BJ'",
                [target_date],
            )
        target_rows = int(
            conn.execute(
                f"SELECT COUNT(*) FROM {_quote_ident(table_name)} WHERE trade_date = ?",
                [target_date],
            ).fetchone()[0]
        )
        total_rows = int(conn.execute(f"SELECT COUNT(*) FROM {_quote_ident(table_name)}").fetchone()[0])

    return {
        "parts_dir": str(parts_path),
        "duckdb_path": str(target_path),
        "table_name": table_name,
        "target_date": target_date,
        "source_rows": source_rows,
        "target_rows": target_rows,
        "target_total_rows": total_rows,
    }


def sync_feature_parts_full_to_duckdb(
    data_dir: str | Path | None,
    parts_dir: str | Path | None = None,
    *,
    duckdb_path: str | Path | None = None,
    table_name: str | None = None,
) -> dict[str, int | str]:
    root = resolve_data_dir(data_dir)
    feature_parts_dir = Path(parts_dir or (root / "production_factor_parts"))
    rewritten = None
    if resolve_stock_daily_duckdb_path(root).exists():
        rewritten = _rewrite_existing_feature_table_with_l2_contract(
            root,
            table_name=resolve_l3_feature_duckdb_table(root, table_name=table_name),
            duckdb_path=duckdb_path,
        )
    if rewritten is not None:
        return rewritten
    if resolve_stock_daily_duckdb_path(root).exists():
        return _sync_feature_parts_with_l2_contract(
            root,
            feature_parts_dir,
            table_name=resolve_l3_feature_duckdb_table(root, table_name=table_name),
            duckdb_path=duckdb_path,
            target_date=None,
        )
    return sync_parquet_parts_full_to_duckdb(
        root,
        feature_parts_dir,
        table_name=resolve_l3_feature_duckdb_table(root, table_name=table_name),
        duckdb_path=duckdb_path,
    )


def sync_feature_parts_target_date_to_duckdb(
    data_dir: str | Path | None,
    target_date: str,
    parts_dir: str | Path | None = None,
    *,
    duckdb_path: str | Path | None = None,
    table_name: str | None = None,
) -> dict[str, int | str]:
    root = resolve_data_dir(data_dir)
    feature_parts_dir = Path(parts_dir or (root / "production_factor_parts"))
    if resolve_stock_daily_duckdb_path(root).exists():
        return _sync_feature_parts_with_l2_contract(
            root,
            feature_parts_dir,
            table_name=resolve_l3_feature_duckdb_table(root, table_name=table_name),
            duckdb_path=duckdb_path,
            target_date=target_date,
        )
    return sync_parquet_parts_trade_date_to_duckdb(
        root,
        feature_parts_dir,
        table_name=resolve_l3_feature_duckdb_table(root, table_name=table_name),
        target_date=target_date,
        duckdb_path=duckdb_path,
    )


def sync_label_parts_full_to_duckdb(
    data_dir: str | Path | None,
    parts_dir: str | Path | None = None,
    *,
    duckdb_path: str | Path | None = None,
    table_name: str | None = None,
) -> dict[str, int | str]:
    root = resolve_data_dir(data_dir)
    return _sync_label_parts_full_to_duckdb(
        root,
        parts_dir or (root / "prediction_label_parts"),
        table_name=resolve_l3_label_duckdb_table(root, table_name=table_name),
        duckdb_path=duckdb_path or resolve_l3_label_duckdb_path(root),
    )


def sync_label_parts_target_date_to_duckdb(
    data_dir: str | Path | None,
    target_date: str,
    parts_dir: str | Path | None = None,
    *,
    duckdb_path: str | Path | None = None,
    table_name: str | None = None,
) -> dict[str, int | str]:
    root = resolve_data_dir(data_dir)
    return _sync_label_parts_trade_date_to_duckdb(
        root,
        parts_dir or (root / "prediction_label_parts"),
        table_name=resolve_l3_label_duckdb_table(root, table_name=table_name),
        target_date=target_date,
        duckdb_path=duckdb_path or resolve_l3_label_duckdb_path(root),
    )
