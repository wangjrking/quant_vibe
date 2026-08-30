from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import tempfile
from pathlib import Path
from contextlib import closing

import duckdb
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MAIN_DIR = PROJECT_ROOT / "quant" / "main"
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from raw_table_db_module import replace_raw_table_full, replace_raw_table_trade_range


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate L1 DuckDB update contract with isolated temp data.")
    parser.add_argument("--report", required=True)
    args = parser.parse_args(argv)

    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = PROJECT_ROOT / report_path

    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp) / "data_file"
        frame = pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ"],
                "trade_date": ["20260628", "20260628"],
                "close": [10.0, 20.0],
            }
        )
        replacement = pd.DataFrame(
            {
                "ts_code": ["000003.SZ"],
                "trade_date": ["20260629"],
                "close": [30.0],
            }
        )
        replace_raw_table_full(data_dir, "duckdb_update_contract_probe", frame)
        replace_raw_table_trade_range(
            data_dir,
            "duckdb_update_contract_probe",
            replacement,
            "20260629",
            "20260629",
        )
        split_path = data_dir / "raw_table_dbs" / "duckdb_update_contract_probe.DB"
        duckdb_path = data_dir / "production_assets" / "duckdb" / "l1_raw_tables" / "duckdb_update_contract_probe.duckdb"
        with sqlite3.connect(split_path) as conn:
            split_rows = int(conn.execute("SELECT COUNT(*) FROM duckdb_update_contract_probe").fetchone()[0])
        with duckdb.connect(str(duckdb_path), read_only=True) as conn:
            duck_rows = int(conn.execute("SELECT COUNT(*) FROM duckdb_update_contract_probe").fetchone()[0])
            target_rows = int(
                conn.execute(
                    "SELECT COUNT(*) FROM duckdb_update_contract_probe WHERE trade_date = '20260629'"
                ).fetchone()[0]
            )

        real_path_frame = pd.DataFrame(
            {
                "ts_code": ["000004.SZ", "000005.SZ"],
                "trade_date": ["20260630", "20260630"],
                "close": [40.0, 50.0],
            }
        )
        replace_raw_table_trade_range(
            data_dir,
            "daily_data",
            real_path_frame,
            "20260630",
            "20260630",
        )
        real_split_path = data_dir / "raw_table_dbs" / "daily_data.DB"
        with sqlite3.connect(real_split_path) as conn:
            real_split_rows = int(conn.execute("SELECT COUNT(*) FROM daily_data").fetchone()[0])
        with duckdb.connect(str(data_dir / "production_assets" / "duckdb" / "l1_raw_tables" / "daily_data.duckdb"), read_only=True) as conn:
            real_duck_rows = int(conn.execute("SELECT COUNT(*) FROM daily_data").fetchone()[0])
            real_target_rows = int(
                conn.execute("SELECT COUNT(*) FROM daily_data WHERE trade_date = '20260630'").fetchone()[0]
            )

    result = {
        "status": "ok"
        if split_rows == duck_rows
        and target_rows == 1
        and real_split_rows == real_duck_rows
        and real_target_rows == 2
        else "failed",
        "checks": {
            "split_rows": split_rows,
            "duckdb_rows": duck_rows,
            "duckdb_target_date_rows": target_rows,
            "real_write_path": {
                "entrypoint": "run_0609_dual_signals.replace_sqlite_trade_range",
                "split_rows": real_split_rows,
                "duckdb_rows": real_duck_rows,
                "duckdb_target_date_rows": real_target_rows,
            },
        },
        "boundary": {
            "uses_isolated_temp_data": True,
            "edits_production_registry": False,
            "switches_mainline_routes": False,
            "runs_business_pipeline": False,
            "touches_real_l1_assets": False,
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(str(report_path))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
