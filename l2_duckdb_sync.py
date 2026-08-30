from __future__ import annotations

import os
from pathlib import Path

from duckdb_asset_route import connect_duckdb_readonly
from stock_daily_data_route import resolve_stock_daily_db_path, resolve_stock_daily_duckdb_path


ENV_L2_DUCKDB_SYNC = "QUANT_L2_DUCKDB_SYNC"
DEFAULT_TABLE = "STOCK_DAILY_DATA"


def l2_duckdb_sync_enabled() -> bool:
    value = str(os.environ.get(ENV_L2_DUCKDB_SYNC, "1")).strip().lower()
    return value not in {"0", "false", "no", "off"}


def _quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _duckdb_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _numeric_frame_columns(frame) -> set[str]:
    try:
        from pandas.api.types import is_numeric_dtype
    except Exception:
        return set()
    return {str(column) for column in frame.columns if is_numeric_dtype(frame[column])}


def _normalize_numeric_blob_columns(conn, table_name: str, frame) -> list[str]:
    table_ident = _quote_ident(table_name)
    numeric_columns = _numeric_frame_columns(frame)
    if not numeric_columns:
        return []

    schema_rows = conn.execute(f"DESCRIBE {table_ident}").fetchall()
    convert_columns = [
        str(row[0])
        for row in schema_rows
        if str(row[1]).upper() == "BLOB" and str(row[0]) in numeric_columns
    ]
    if not convert_columns:
        return []

    for column in convert_columns:
        column_ident = _quote_ident(column)
        bad_values = int(
            conn.execute(
                f"""
                SELECT COUNT(*)
                FROM {table_ident}
                WHERE {column_ident} IS NOT NULL
                  AND TRY_CAST(CAST({column_ident} AS VARCHAR) AS DOUBLE) IS NULL
                """
            ).fetchone()[0]
        )
        if bad_values:
            raise ValueError(
                f"cannot convert {table_name}.{column} from BLOB to DOUBLE; invalid values={bad_values}"
            )

    tmp_table = "__l2_stock_daily_retyped"
    tmp_ident = _quote_ident(tmp_table)
    conn.execute(f"DROP TABLE IF EXISTS {tmp_ident}")
    select_exprs = []
    for row in schema_rows:
        column = str(row[0])
        column_ident = _quote_ident(column)
        if column in convert_columns:
            select_exprs.append(
                f"TRY_CAST(CAST({column_ident} AS VARCHAR) AS DOUBLE) AS {column_ident}"
            )
        else:
            select_exprs.append(column_ident)
    conn.execute(f"CREATE TABLE {tmp_ident} AS SELECT {', '.join(select_exprs)} FROM {table_ident}")
    conn.execute(f"DROP TABLE {table_ident}")
    conn.execute(f"ALTER TABLE {tmp_ident} RENAME TO {table_ident}")
    return convert_columns


def sync_stock_daily_frame_to_duckdb(
    data_dir: str | Path,
    frame,
    start_date: str,
    end_date: str,
    *,
    duckdb_path: str | Path | None = None,
    table_name: str = DEFAULT_TABLE,
    trade_col: str = "trade_date",
) -> dict[str, int | str]:
    target_path = resolve_stock_daily_duckdb_path(data_dir=data_dir, duckdb_path=duckdb_path)
    if trade_col not in frame.columns:
        raise ValueError(f"frame is missing required trade column: {trade_col}")

    trade_values = frame[trade_col].astype(str)
    range_frame = frame[(trade_values >= str(start_date)) & (trade_values <= str(end_date))].copy()
    source_rows = int(len(range_frame))

    import duckdb

    target_path.parent.mkdir(parents=True, exist_ok=True)
    table_ident = _quote_ident(table_name)
    trade_ident = _quote_ident(trade_col)
    staging_name = "__l2_stock_daily_frame"
    staging_ident = _quote_ident(staging_name)

    with duckdb.connect(str(target_path)) as conn:
        conn.register(staging_name, range_frame)
        try:
            table_exists = bool(
                conn.execute(
                    "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
                    [table_name],
                ).fetchone()[0]
            )
            converted_blob_columns = []
            if not table_exists:
                conn.execute(f"CREATE TABLE {table_ident} AS SELECT * FROM {staging_ident}")
            else:
                converted_blob_columns = _normalize_numeric_blob_columns(conn, table_name, range_frame)
                target_columns = [
                    str(row[0])
                    for row in conn.execute(f"DESCRIBE {table_ident}").fetchall()
                ]
                frame_columns = set(str(column) for column in range_frame.columns)
                missing_columns = [column for column in target_columns if column not in frame_columns]
                if missing_columns:
                    raise ValueError(
                        f"frame is missing target columns for {table_name}: {missing_columns[:20]}"
                    )
                selected = ", ".join(_quote_ident(column) for column in target_columns)
                conn.execute(
                    f"DELETE FROM {table_ident} WHERE {trade_ident} >= ? AND {trade_ident} <= ?",
                    [str(start_date), str(end_date)],
                )
                if source_rows:
                    conn.execute(
                        f"INSERT INTO {table_ident} ({selected}) SELECT {selected} FROM {staging_ident}"
                    )
            target_rows = int(
                conn.execute(
                    f"SELECT COUNT(*) FROM {table_ident} WHERE {trade_ident} >= ? AND {trade_ident} <= ?",
                    [str(start_date), str(end_date)],
                ).fetchone()[0]
            )
            total_rows = int(conn.execute(f"SELECT COUNT(*) FROM {table_ident}").fetchone()[0])
        finally:
            conn.unregister(staging_name)
    return {
        "source_mode": "frame",
        "source_rows": source_rows,
        "target_rows": target_rows,
        "target_total_rows": total_rows,
        "converted_blob_columns": ",".join(converted_blob_columns) if table_exists else "",
        "start_date": str(start_date),
        "end_date": str(end_date),
        "duckdb_path": str(target_path),
    }


def sync_stock_daily_trade_range_to_duckdb(
    data_dir: str | Path,
    start_date: str,
    end_date: str,
    *,
    sqlite_db_path: str | Path | None = None,
    duckdb_path: str | Path | None = None,
    table_name: str = DEFAULT_TABLE,
    trade_col: str = "trade_date",
) -> dict[str, int | str]:
    sqlite_path = resolve_stock_daily_db_path(data_dir=data_dir, db_path=sqlite_db_path, require_exists=True)
    target_path = resolve_stock_daily_duckdb_path(data_dir=data_dir, duckdb_path=duckdb_path)

    import duckdb

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(target_path)) as conn:
        try:
            conn.execute("LOAD sqlite")
        except Exception:
            conn.execute("INSTALL sqlite")
            conn.execute("LOAD sqlite")
        source_expr = "SELECT * FROM sqlite_scan(?, ?) WHERE " + trade_col + " >= ? AND " + trade_col + " <= ?"
        source_rows = int(
            conn.execute(
                "SELECT COUNT(*) FROM (" + source_expr + ")",
                [str(sqlite_path), table_name, start_date, end_date],
            ).fetchone()[0]
        )
        table_exists = bool(
            conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
                [table_name],
            ).fetchone()[0]
        )
        if not table_exists:
            conn.execute(
                f'CREATE TABLE "{table_name}" AS ' + source_expr,
                [str(sqlite_path), table_name, start_date, end_date],
            )
        else:
            conn.execute(
                f'DELETE FROM "{table_name}" WHERE "{trade_col}" >= ? AND "{trade_col}" <= ?',
                [start_date, end_date],
            )
            conn.execute(
                f'INSERT INTO "{table_name}" ' + source_expr,
                [str(sqlite_path), table_name, start_date, end_date],
            )
        target_rows = int(
            conn.execute(
                f'SELECT COUNT(*) FROM "{table_name}" WHERE "{trade_col}" >= ? AND "{trade_col}" <= ?',
                [start_date, end_date],
            ).fetchone()[0]
        )
        total_rows = int(conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0])
    return {
        "source_rows": source_rows,
        "target_rows": target_rows,
        "target_total_rows": total_rows,
        "start_date": start_date,
        "end_date": end_date,
        "sqlite_db_path": str(sqlite_path),
        "duckdb_path": str(target_path),
    }


def sync_stock_daily_sqlite_full_to_duckdb(
    data_dir: str | Path,
    *,
    sqlite_db_path: str | Path | None = None,
    duckdb_path: str | Path | None = None,
    table_name: str = DEFAULT_TABLE,
    trade_col: str = "trade_date",
) -> dict[str, int | str]:
    sqlite_path = resolve_stock_daily_db_path(data_dir=data_dir, db_path=sqlite_db_path, require_exists=True)
    target_path = resolve_stock_daily_duckdb_path(data_dir=data_dir, duckdb_path=duckdb_path)

    import duckdb

    target_path.parent.mkdir(parents=True, exist_ok=True)
    table_ident = _quote_ident(table_name)
    trade_ident = _quote_ident(trade_col)
    with duckdb.connect(str(target_path)) as conn:
        try:
            conn.execute("LOAD sqlite")
        except Exception:
            conn.execute("INSTALL sqlite")
            conn.execute("LOAD sqlite")
        source_rows = int(
            conn.execute(
                "SELECT COUNT(*) FROM sqlite_scan(?, ?)",
                [str(sqlite_path), table_name],
            ).fetchone()[0]
        )
        conn.execute(f"DROP TABLE IF EXISTS {table_ident}")
        conn.execute(
            f"CREATE TABLE {table_ident} AS SELECT * FROM sqlite_scan(?, ?)",
            [str(sqlite_path), table_name],
        )
        target_rows = int(conn.execute(f"SELECT COUNT(*) FROM {table_ident}").fetchone()[0])
        min_trade_date, max_trade_date = conn.execute(
            f"SELECT MIN({trade_ident}), MAX({trade_ident}) FROM {table_ident}"
        ).fetchone()
    return {
        "source_mode": "sqlite_full_table",
        "source_rows": source_rows,
        "target_rows": target_rows,
        "target_total_rows": target_rows,
        "min_trade_date": min_trade_date or "",
        "max_trade_date": max_trade_date or "",
        "sqlite_db_path": str(sqlite_path),
        "duckdb_path": str(target_path),
    }


def sync_stock_daily_duckdb_table_full_to_duckdb(
    data_dir: str | Path,
    *,
    source_duckdb_path: str | Path,
    target_duckdb_path: str | Path | None = None,
    table_name: str = DEFAULT_TABLE,
    trade_col: str = "trade_date",
) -> dict[str, int | str]:
    import duckdb

    source_path = Path(source_duckdb_path)
    if not source_path.is_absolute():
        source_path = source_path.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"source DuckDB asset not found: {source_path}")
    target_path = resolve_stock_daily_duckdb_path(data_dir=data_dir, duckdb_path=target_duckdb_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    table_ident = _quote_ident(table_name)
    trade_ident = _quote_ident(trade_col)

    with duckdb.connect(str(target_path)) as conn:
        conn.execute(f"ATTACH {_duckdb_literal(str(source_path))} AS source_db (READ_ONLY)")
        try:
            source_rows = int(
                conn.execute(f"SELECT COUNT(*) FROM source_db.{table_ident}").fetchone()[0]
            )
            conn.execute(f"DROP TABLE IF EXISTS {table_ident}")
            conn.execute(
                f"CREATE TABLE {table_ident} AS SELECT * FROM source_db.{table_ident}"
            )
            target_rows = int(conn.execute(f"SELECT COUNT(*) FROM {table_ident}").fetchone()[0])
            min_trade_date, max_trade_date = conn.execute(
                f"SELECT MIN({trade_ident}), MAX({trade_ident}) FROM {table_ident}"
            ).fetchone()
        finally:
            conn.execute("DETACH source_db")
    return {
        "source_mode": "duckdb_full_table",
        "source_rows": source_rows,
        "target_rows": target_rows,
        "target_total_rows": target_rows,
        "min_trade_date": min_trade_date or "",
        "max_trade_date": max_trade_date or "",
        "source_duckdb_path": str(source_path),
        "duckdb_path": str(target_path),
    }


def count_stock_daily_duckdb_rows(
    data_dir: str | Path,
    *,
    duckdb_path: str | Path | None = None,
    table_name: str = DEFAULT_TABLE,
) -> int:
    path = resolve_stock_daily_duckdb_path(data_dir=data_dir, duckdb_path=duckdb_path, require_exists=True)
    conn = connect_duckdb_readonly(path)
    try:
        return int(conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0])
    finally:
        conn.close()
