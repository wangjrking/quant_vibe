from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MAIN_DIR = PROJECT_ROOT / "quant" / "main"
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from l1_raw_data_route import BACKEND_DUCKDB, materialize_sqlite_temp_raw_table, read_raw_table_frame


DATA_DIR = PROJECT_ROOT / "quant" / "data_file"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate L1 DuckDB read contract without switching mainline.")
    parser.add_argument(
        "--duckdb-path",
        default=str(DATA_DIR / "production_assets" / "duckdb" / "l1_raw_tables"),
    )
    parser.add_argument("--report", required=True)
    parser.add_argument("--trade-date", default="20260626")
    args = parser.parse_args(argv)

    duckdb_path = Path(args.duckdb_path)
    report_path = Path(args.report)
    if not duckdb_path.is_absolute():
        duckdb_path = PROJECT_ROOT / duckdb_path
    if not report_path.is_absolute():
        report_path = PROJECT_ROOT / report_path

    daily_frame = read_raw_table_frame(
        DATA_DIR,
        "daily_data",
        backend=BACKEND_DUCKDB,
        duckdb_path=duckdb_path,
        where_sql="trade_date = ?",
        params=[args.trade_date],
    )
    stock_basic_frame = read_raw_table_frame(
        DATA_DIR,
        "stock_basic_data",
        backend=BACKEND_DUCKDB,
        duckdb_path=duckdb_path,
    )
    with sqlite3.connect(":memory:") as conn:
        materialize_sqlite_temp_raw_table(
            conn,
            temp_name="daily_data",
            table_name="daily_data",
            data_dir=DATA_DIR,
            backend=BACKEND_DUCKDB,
            duckdb_path=duckdb_path,
            where_sql="trade_date = ?",
            params=[args.trade_date],
        )
        temp_daily_rows = int(conn.execute("SELECT COUNT(*) FROM temp.daily_data").fetchone()[0])

    result = {
        "status": "ok",
        "duckdb_path": str(duckdb_path),
        "backend": BACKEND_DUCKDB,
        "trade_date": args.trade_date,
        "checks": {
            "daily_data_rows_for_trade_date": int(len(daily_frame)),
            "stock_basic_rows": int(len(stock_basic_frame)),
            "sqlite_temp_materialization_rows": temp_daily_rows,
        },
        "boundary": {
            "read_only_validation": True,
            "edits_production_registry": False,
            "switches_mainline_routes": False,
            "runs_business_pipeline": False,
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(str(report_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
