from __future__ import annotations

"""Historical one-off L1 governance remediation script.

This script was used for the 2026-06-17 auxiliary raw-table P3 cleanup
(`ths_hot` trade_date trimming plus auxiliary SQLite/split-DB date sync).
It is retained in place for audit traceability and possible historical
reproduction only. It is not part of the current default L1 standard chain.
"""

import argparse
import hashlib
import json
import shutil
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import pandas as pd


TABLES = ("top_list", "ths_hot", "dc_hot")
NATURAL_KEYS = {
    "top_list": ["trade_date", "ts_code", "reason"],
    "ths_hot": ["trade_date", "data_type", "ts_code", "rank_time"],
    "dc_hot": ["trade_date", "data_type", "ts_code", "rank_time"],
}


def now_tag() -> str:
    tz = timezone(timedelta(hours=8))
    return datetime.now(tz).strftime("%Y%m%d_%H%M%S")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def connect(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(path, timeout=30)


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "select 1 from sqlite_master where type='table' and name=?", (table,)
    ).fetchone()
    return row is not None


def db_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in conn.execute(f'pragma table_info("{table}")')]


def load_table(conn: sqlite3.Connection, table: str) -> pd.DataFrame:
    return pd.read_sql_query(f'select * from "{table}"', conn)


def metric_df(df: pd.DataFrame, table: str, store: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "store": store,
        "table": table,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": list(df.columns),
        "full_row_duplicate_count": int(df.duplicated().sum()) if len(df) else 0,
    }
    if "trade_date" in df.columns:
        raw = df["trade_date"].astype(str)
        trimmed = raw.str.strip()
        out.update(
            {
                "raw_min_trade_date": raw.min() if len(raw) else None,
                "raw_max_trade_date": raw.max() if len(raw) else None,
                "trim_min_trade_date": trimmed.min() if len(trimmed) else None,
                "trim_max_trade_date": trimmed.max() if len(trimmed) else None,
                "trim_distinct_trade_dates": int(trimmed.nunique()),
                "surrounding_space_rows": int((raw != trimmed).sum()),
            }
        )
    keys = [col for col in NATURAL_KEYS.get(table, []) if col in df.columns]
    if keys:
        key_df = df[keys].copy()
        if "trade_date" in key_df.columns:
            key_df["trade_date"] = key_df["trade_date"].astype(str).str.strip()
        out["natural_key"] = keys
        out["natural_key_duplicate_groups"] = int(
            key_df.groupby(keys, dropna=False).size().gt(1).sum()
        )
    return out


def read_parquet_table(data_dir: Path, table: str) -> pd.DataFrame:
    return pd.read_parquet(data_dir / f"{table}.parquet")


def write_parquet_table(data_dir: Path, table: str, df: pd.DataFrame) -> None:
    df.to_parquet(data_dir / f"{table}.parquet", index=False)


def backup_sqlite_table(
    src_db: Path, table: str, backup_db: Path, backup_table: str
) -> None:
    with connect(src_db) as src, connect(backup_db) as dst:
        if not table_exists(src, table):
            return
        df = load_table(src, table)
        df.to_sql(backup_table, dst, if_exists="replace", index=False)


def append_missing_trade_dates(
    db_path: Path, table: str, src_df: pd.DataFrame
) -> dict[str, Any]:
    with connect(db_path) as conn:
        if not table_exists(conn, table):
            raise RuntimeError(f"Missing table {table} in {db_path}")
        before_df = load_table(conn, table)
        before_dates = set(before_df["trade_date"].astype(str).str.strip())
        source = src_df.copy()
        source["trade_date"] = source["trade_date"].astype(str).str.strip()
        missing_dates = sorted(set(source["trade_date"]) - before_dates)
        append_df = source[source["trade_date"].isin(missing_dates)].copy()
        columns = db_columns(conn, table)
        append_df = append_df[columns]
        if len(append_df):
            append_df.to_sql(table, conn, if_exists="append", index=False)
        conn.commit()
        return {
            "db_path": str(db_path),
            "table": table,
            "missing_dates_appended": missing_dates,
            "rows_appended": int(len(append_df)),
        }


def trim_ths_hot_sqlite(db_path: Path) -> dict[str, Any]:
    with connect(db_path) as conn:
        before = conn.execute(
            'select count(*) from "ths_hot" where trade_date != trim(trade_date)'
        ).fetchone()[0]
        conn.execute(
            'update "ths_hot" set trade_date = trim(trade_date) '
            'where trade_date != trim(trade_date)'
        )
        conn.commit()
        after = conn.execute(
            'select count(*) from "ths_hot" where trade_date != trim(trade_date)'
        ).fetchone()[0]
    return {
        "db_path": str(db_path),
        "table": "ths_hot",
        "space_rows_before": int(before),
        "space_rows_after": int(after),
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# L1 Auxiliary P3 Governance Fix",
        "",
        f"- generated_at: {report['generated_at']}",
        f"- python: {report['python']}",
        "- scope: ths_hot trade_date trim; auxiliary SQLite coverage sync from raw parquet missing dates",
        "- boundary: no Tushare pull, no STOCK_DAILY_DATA, no factor/model/prediction/signal/backtest",
        "",
        "## Writes",
        "",
    ]
    for item in report["writes"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Backups", ""])
    for item in report["backups"]:
        lines.append(f"- {item}")
    lines.extend(["", "## SQLite Appends", ""])
    for item in report["sqlite_appends"]:
        dates = ",".join(item["missing_dates_appended"]) or "none"
        lines.append(
            f"- {item['table']} @ {item['db_path']}: rows_appended={item['rows_appended']}, dates={dates}"
        )
    lines.extend(["", "## ths_hot Space Rows", ""])
    for item in report["ths_hot_trim"]:
        lines.append(
            f"- {item['db_path']}: before={item['space_rows_before']}, after={item['space_rows_after']}"
        )
    lines.extend(["", "## Validation Summary", ""])
    for row in report["validation_summary"]:
        lines.append(
            "- {store}.{table}: rows {before_rows}->{after_rows}, columns {before_columns}->{after_columns}, "
            "space_rows {before_space}->{after_space}, trim_range {before_range}->{after_range}, "
            "trim_dates {before_distinct}->{after_distinct}, key_dups {before_key_dups}->{after_key_dups}".format(
                **row
            )
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default=str(Path(__file__).resolve().parents[1] / "data_file"),
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    reports_dir = data_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    raw_db_dir = data_dir / "raw_table_dbs"
    legacy_db = data_dir / "odb.db"
    tag = now_tag()
    backup_dir = data_dir / "backups" / f"l1_auxiliary_governance_p3_{tag}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    rollback_db = backup_dir / "legacy_and_split_table_backup.sqlite"

    before: list[dict[str, Any]] = []
    for table in TABLES:
        before.append(metric_df(read_parquet_table(data_dir, table), table, "raw_parquet"))
        with connect(legacy_db) as conn:
            before.append(metric_df(load_table(conn, table), table, "legacy_odb"))
        with connect(raw_db_dir / f"{table}.DB") as conn:
            before.append(metric_df(load_table(conn, table), table, "split_db"))

    backups: list[str] = []
    ths_parquet = data_dir / "ths_hot.parquet"
    ths_parquet_backup = backup_dir / "ths_hot.parquet.before"
    shutil.copy2(ths_parquet, ths_parquet_backup)
    backups.append(f"{ths_parquet_backup} sha256={sha256_file(ths_parquet_backup)}")

    for table in TABLES:
        backup_sqlite_table(legacy_db, table, rollback_db, f"legacy_odb__{table}")
        backups.append(f"{rollback_db}::{('legacy_odb__' + table)}")
        split_db = raw_db_dir / f"{table}.DB"
        split_backup = backup_dir / f"{table}.DB.before"
        shutil.copy2(split_db, split_backup)
        backups.append(f"{split_backup} sha256={sha256_file(split_backup)}")

    writes: list[str] = []
    ths_df = read_parquet_table(data_dir, "ths_hot")
    ths_df["trade_date"] = ths_df["trade_date"].astype(str).str.strip()
    write_parquet_table(data_dir, "ths_hot", ths_df)
    writes.append(str(ths_parquet))

    trim_results = []
    trim_results.append(trim_ths_hot_sqlite(legacy_db))
    trim_results.append(trim_ths_hot_sqlite(raw_db_dir / "ths_hot.DB"))
    writes.append(f"{legacy_db}::ths_hot")
    writes.append(f"{raw_db_dir / 'ths_hot.DB'}::ths_hot")

    append_results = []
    for table in TABLES:
        source_df = read_parquet_table(data_dir, table)
        source_df["trade_date"] = source_df["trade_date"].astype(str).str.strip()
        for db_path, store in (
            (legacy_db, "legacy_odb"),
            (raw_db_dir / f"{table}.DB", "split_db"),
        ):
            result = append_missing_trade_dates(db_path, table, source_df)
            result["store"] = store
            append_results.append(result)
            if result["rows_appended"]:
                writes.append(f"{db_path}::{table}")

    after: list[dict[str, Any]] = []
    for table in TABLES:
        after.append(metric_df(read_parquet_table(data_dir, table), table, "raw_parquet"))
        with connect(legacy_db) as conn:
            after.append(metric_df(load_table(conn, table), table, "legacy_odb"))
        with connect(raw_db_dir / f"{table}.DB") as conn:
            after.append(metric_df(load_table(conn, table), table, "split_db"))

    before_map = {(m["store"], m["table"]): m for m in before}
    after_map = {(m["store"], m["table"]): m for m in after}
    validation_summary = []
    for key, b in before_map.items():
        a = after_map[key]
        validation_summary.append(
            {
                "store": key[0],
                "table": key[1],
                "before_rows": b["rows"],
                "after_rows": a["rows"],
                "before_columns": b["columns"],
                "after_columns": a["columns"],
                "before_space": b.get("surrounding_space_rows"),
                "after_space": a.get("surrounding_space_rows"),
                "before_range": f"{b.get('trim_min_trade_date')}-{b.get('trim_max_trade_date')}",
                "after_range": f"{a.get('trim_min_trade_date')}-{a.get('trim_max_trade_date')}",
                "before_distinct": b.get("trim_distinct_trade_dates"),
                "after_distinct": a.get("trim_distinct_trade_dates"),
                "before_key_dups": b.get("natural_key_duplicate_groups"),
                "after_key_dups": a.get("natural_key_duplicate_groups"),
            }
        )

    report = {
        "generated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "python": str(Path(__import__("sys").executable).resolve()),
        "data_dir": str(data_dir),
        "backup_dir": str(backup_dir),
        "rollback_db": str(rollback_db),
        "writes": sorted(set(writes)),
        "backups": backups,
        "before": before,
        "after": after,
        "sqlite_appends": append_results,
        "ths_hot_trim": trim_results,
        "validation_summary": validation_summary,
        "boundary": {
            "tushare_pull": False,
            "stock_daily_data": False,
            "factor": False,
            "model": False,
            "prediction": False,
            "signal": False,
            "backtest": False,
        },
    }

    report_base = reports_dir / "l1_auxiliary_governance_p3_fix_20260617"
    json_path = report_base.with_suffix(".json")
    csv_path = report_base.with_suffix(".csv")
    md_path = report_base.with_suffix(".md")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(validation_summary).to_csv(csv_path, index=False, encoding="utf-8-sig")
    md_path.write_text(build_markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "md": str(md_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
