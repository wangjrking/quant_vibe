import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
import json

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from l4_duckdb_sync import sync_prediction_table_full_to_duckdb


class L4DuckDBSyncTests(unittest.TestCase):
    def _seed_sqlite(self, path: Path, rows: list[tuple[str, str, float]]) -> None:
        with sqlite3.connect(path) as conn:
            conn.execute("CREATE TABLE pred_table(trade_date TEXT, stock_code TEXT, pred_prob REAL)")
            conn.executemany("INSERT INTO pred_table VALUES (?, ?, ?)", rows)
            conn.commit()

    def test_full_sync_creates_prediction_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data_file"
            sqlite_path = data_dir / "model_predictions" / "MODEL_PREDICTIONS.db"
            duckdb_path = data_dir / "production_assets" / "duckdb" / "quant_production.duckdb"
            sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            self._seed_sqlite(
                sqlite_path,
                [
                    ("20260627", "000001.SZ", 0.1),
                    ("20260628", "000002.SZ", 0.2),
                ],
            )

            result = sync_prediction_table_full_to_duckdb(
                data_dir,
                "pred_table",
                sqlite_db_path=sqlite_path,
                duckdb_path=duckdb_path,
            )

            self.assertEqual(result["source_rows"], 2)
            self.assertEqual(result["target_rows"], 2)
            with duckdb.connect(str(duckdb_path), read_only=True) as conn:
                rows = conn.execute("SELECT COUNT(*) FROM pred_table").fetchone()[0]
            self.assertEqual(rows, 2)

    def test_full_sync_replaces_existing_duckdb_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data_file"
            sqlite_path = data_dir / "model_predictions" / "MODEL_PREDICTIONS.db"
            duckdb_path = data_dir / "production_assets" / "duckdb" / "quant_production.duckdb"
            sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            self._seed_sqlite(sqlite_path, [("20260627", "000001.SZ", 0.1)])
            sync_prediction_table_full_to_duckdb(
                data_dir,
                "pred_table",
                sqlite_db_path=sqlite_path,
                duckdb_path=duckdb_path,
            )

            with sqlite3.connect(sqlite_path) as conn:
                conn.execute("DELETE FROM pred_table")
                conn.executemany(
                    "INSERT INTO pred_table VALUES (?, ?, ?)",
                    [
                        ("20260628", "000003.SZ", 0.3),
                        ("20260628", "000004.SZ", 0.4),
                    ],
                )
                conn.commit()

            result = sync_prediction_table_full_to_duckdb(
                data_dir,
                "pred_table",
                sqlite_db_path=sqlite_path,
                duckdb_path=duckdb_path,
            )

            self.assertEqual(result["source_rows"], 2)
            self.assertEqual(result["target_rows"], 2)
            with duckdb.connect(str(duckdb_path), read_only=True) as conn:
                rows = conn.execute("SELECT COUNT(*) FROM pred_table").fetchone()[0]
                target_rows = conn.execute(
                    "SELECT COUNT(*) FROM pred_table WHERE trade_date = '20260628'"
                ).fetchone()[0]
            self.assertEqual(rows, 2)
            self.assertEqual(target_rows, 2)

    def test_full_sync_prefers_active_registry_duckdb_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data_file"
            sqlite_path = data_dir / "model_predictions" / "MODEL_PREDICTIONS.db"
            registry_dir = data_dir / "asset_registry"
            registry_dir.mkdir(parents=True, exist_ok=True)
            (data_dir / "reports").mkdir(parents=True, exist_ok=True)
            audit_record = data_dir / "reports" / "audit.md"
            audit_record.write_text("approved", encoding="utf-8")
            custom_duckdb_path = data_dir / "production_assets" / "duckdb" / "l4_predictions_current.duckdb"
            registry_dir.joinpath("production_assets.json").write_text(
                json.dumps(
                    {
                        "assets": [
                            {
                                "asset_id": "prod_l4_predictions_current",
                                "track": "production",
                                "layer": "L4_model_prediction",
                                "asset_type": "duckdb_table",
                                "status": "production_active",
                                "allowed_for_main_workflow": True,
                                "asset_path": f"{custom_duckdb_path}::pred_table",
                                "audit_record": str(audit_record),
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            self._seed_sqlite(
                sqlite_path,
                [
                    ("20260627", "000001.SZ", 0.1),
                ],
            )

            result = sync_prediction_table_full_to_duckdb(
                data_dir,
                "pred_table",
                sqlite_db_path=sqlite_path,
            )

            self.assertEqual(Path(result["duckdb_path"]), custom_duckdb_path)
            with duckdb.connect(str(custom_duckdb_path), read_only=True) as conn:
                rows = conn.execute("SELECT COUNT(*) FROM pred_table").fetchone()[0]
            self.assertEqual(rows, 1)


if __name__ == "__main__":
    unittest.main()
