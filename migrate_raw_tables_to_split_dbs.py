from __future__ import annotations

import argparse
import json
import sqlite3
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path


RAW_TABLES = [
    "daily_data",
    "daily_index_data",
    "stock_basic_data",
    "stk_factor",
    "moneyflow",
    "limit_list_data",
    "cyq_perf",
    "adj_factor",
    "stock_st",
    "index_daily",
    "top_list",
    "ths_hot",
    "dc_hot",
]


def now() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def log(message: str, log_path: Path) -> None:
    line = f"{now()} {message}"
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def fetch_one(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> tuple | None:
    cursor = conn.execute(sql, params)
    return cursor.fetchone()


def table_columns(conn: sqlite3.Connection, schema: str, table: str) -> list[str]:
    return [row[1] for row in conn.execute(f"PRAGMA {quote_ident(schema)}.table_info({quote_ident(table)})")]


def table_exists(conn: sqlite3.Connection, schema: str, table: str) -> bool:
    row = fetch_one(
        conn,
        f"SELECT name FROM {quote_ident(schema)}.sqlite_master WHERE type='table' AND name=?",
        (table,),
    )
    return row is not None


def table_schema(conn: sqlite3.Connection, schema: str, table: str) -> str:
    row = fetch_one(
        conn,
        f"SELECT sql FROM {quote_ident(schema)}.sqlite_master WHERE type='table' AND name=?",
        (table,),
    )
    if not row or not row[0]:
        raise RuntimeError(f"Cannot find CREATE TABLE SQL for {table}")
    return str(row[0])


def index_schemas(conn: sqlite3.Connection, schema: str, table: str) -> list[str]:
    rows = conn.execute(
        f"""
        SELECT sql
        FROM {quote_ident(schema)}.sqlite_master
        WHERE type='index' AND tbl_name=? AND sql IS NOT NULL
        ORDER BY name
        """,
        (table,),
    ).fetchall()
    return [str(row[0]) for row in rows if row[0]]


def trigger_schemas(conn: sqlite3.Connection, schema: str, table: str) -> list[str]:
    rows = conn.execute(
        f"""
        SELECT sql
        FROM {quote_ident(schema)}.sqlite_master
        WHERE type='trigger' AND tbl_name=? AND sql IS NOT NULL
        ORDER BY name
        """,
        (table,),
    ).fetchall()
    return [str(row[0]) for row in rows if row[0]]


def table_audit(conn: sqlite3.Connection, schema: str, table: str) -> dict:
    q_table = f"{quote_ident(schema)}.{quote_ident(table)}"
    columns = table_columns(conn, schema, table)
    count = int(fetch_one(conn, f"SELECT COUNT(*) FROM {q_table}")[0])
    audit = {
        "rows": count,
        "columns": len(columns),
        "min_trade_date": "",
        "max_trade_date": "",
        "trade_date_count": "",
        "duplicate_key_groups": "",
    }
    if "trade_date" in columns:
        min_date, max_date, date_count = fetch_one(
            conn,
            f"SELECT MIN(trade_date), MAX(trade_date), COUNT(DISTINCT trade_date) FROM {q_table}",
        )
        audit["min_trade_date"] = min_date or ""
        audit["max_trade_date"] = max_date or ""
        audit["trade_date_count"] = int(date_count or 0)
        key_col = None
        for candidate in ("ts_code", "stock_code"):
            if candidate in columns:
                key_col = candidate
                break
        if key_col:
            dup_groups = fetch_one(
                conn,
                f"""
                SELECT COUNT(*)
                FROM (
                    SELECT {quote_ident(key_col)}, trade_date, COUNT(*) AS n
                    FROM {q_table}
                    GROUP BY {quote_ident(key_col)}, trade_date
                    HAVING n > 1
                )
                """,
            )[0]
            audit["duplicate_key_groups"] = int(dup_groups or 0)
    return audit


def migrate_one(source_db: Path, output_dir: Path, table: str, log_path: Path) -> dict:
    target_db = output_dir / f"{table}.DB"
    temp_db = output_dir / f"{table}.tmp.DB"
    if temp_db.exists():
        temp_db.unlink()

    started = time.time()
    log(f"table_start table={table} target={target_db}", log_path)

    conn = sqlite3.connect(temp_db)
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA busy_timeout=120000")
    conn.execute(f"ATTACH DATABASE {str(source_db)!r} AS src")
    try:
        if not table_exists(conn, "src", table):
            raise RuntimeError(f"source table missing: {table}")

        create_sql = table_schema(conn, "src", table)
        conn.execute(create_sql)
        columns = table_columns(conn, "src", table)
        col_list = ", ".join(quote_ident(col) for col in columns)
        conn.execute("BEGIN")
        cursor = conn.execute(
            f"""
            INSERT INTO {quote_ident(table)} ({col_list})
            SELECT {col_list}
            FROM src.{quote_ident(table)}
            """
        )
        copied_rows = cursor.rowcount if cursor.rowcount is not None else None
        conn.commit()

        for sql in index_schemas(conn, "src", table):
            conn.execute(sql)
        for sql in trigger_schemas(conn, "src", table):
            conn.execute(sql)
        conn.commit()

        source_audit = table_audit(conn, "src", table)
        target_audit = table_audit(conn, "main", table)
        status = "completed" if source_audit == target_audit else "completed_with_audit_mismatch"
        conn.execute("DETACH DATABASE src")
        conn.close()
        temp_db.replace(target_db)

        elapsed = round(time.time() - started, 3)
        result = {
            "table": table,
            "status": status,
            "target_db": str(target_db),
            "copied_rows": copied_rows,
            "elapsed_seconds": elapsed,
            "source_audit": source_audit,
            "target_audit": target_audit,
        }
        log(
            f"table_done table={table} status={status} rows={target_audit['rows']} elapsed_seconds={elapsed}",
            log_path,
        )
        return result
    except Exception as exc:
        try:
            conn.rollback()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass
        result = {
            "table": table,
            "status": "failed",
            "target_db": str(target_db),
            "error": type(exc).__name__ + ": " + str(exc),
            "elapsed_seconds": round(time.time() - started, 3),
        }
        log(f"table_failed table={table} error={result['error']}", log_path)
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split raw ODB tables into one DB file per table.")
    parser.add_argument("--source-db", default=str(Path(__file__).resolve().parent.parent / "data_file" / "odb.db"))
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent.parent / "data_file" / "raw_table_dbs"))
    parser.add_argument("--reports-dir", default=str(Path(__file__).resolve().parent.parent / "data_file" / "reports"))
    parser.add_argument("--tables", default=",".join(RAW_TABLES))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_db = Path(args.source_db)
    output_dir = Path(args.output_dir)
    reports_dir = Path(args.reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    log_path = reports_dir / "raw_table_db_migration_20260616.log"
    summary_path = reports_dir / "raw_table_db_migration_20260616_summary.json"
    csv_path = reports_dir / "raw_table_db_migration_20260616_summary.csv"

    tables = [item.strip() for item in args.tables.split(",") if item.strip()]
    log(f"migration_start source_db={source_db} output_dir={output_dir} tables={len(tables)}", log_path)
    results = [migrate_one(source_db, output_dir, table, log_path) for table in tables]
    status = "completed" if all(result["status"] == "completed" for result in results) else "completed_with_errors"
    summary = {
        "time": now(),
        "status": status,
        "source_db": str(source_db),
        "output_dir": str(output_dir),
        "tables": tables,
        "results": results,
        "boundary": "Only raw data-ingestion tables were copied into split DB files. ODB.DB was not modified and no workflow was changed.",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    import csv

    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "table",
                "status",
                "target_db",
                "source_rows",
                "target_rows",
                "source_min_trade_date",
                "source_max_trade_date",
                "target_min_trade_date",
                "target_max_trade_date",
                "target_duplicate_key_groups",
                "elapsed_seconds",
                "error",
            ]
        )
        for result in results:
            source = result.get("source_audit", {})
            target = result.get("target_audit", {})
            writer.writerow(
                [
                    result.get("table", ""),
                    result.get("status", ""),
                    result.get("target_db", ""),
                    source.get("rows", ""),
                    target.get("rows", ""),
                    source.get("min_trade_date", ""),
                    source.get("max_trade_date", ""),
                    target.get("min_trade_date", ""),
                    target.get("max_trade_date", ""),
                    target.get("duplicate_key_groups", ""),
                    result.get("elapsed_seconds", ""),
                    result.get("error", ""),
                ]
            )
    log(f"migration_done status={status} summary={summary_path}", log_path)
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
