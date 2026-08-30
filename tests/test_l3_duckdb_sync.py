import sys
import tempfile
import unittest
from pathlib import Path
import json

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from l3_duckdb_sync import (
    sync_feature_parts_full_to_duckdb,
    sync_feature_parts_target_date_to_duckdb,
    sync_label_parts_full_to_duckdb,
    sync_label_parts_target_date_to_duckdb,
)


class L3DuckDBSyncTests(unittest.TestCase):
    def _write_parquet(self, path: Path, frame: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)

    def test_feature_full_sync_creates_duckdb_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data_file"
            parts_dir = data_dir / "production_factor_parts"
            duckdb_path = data_dir / "production_assets" / "duckdb" / "quant_production.duckdb"
            self._write_parquet(
                parts_dir / "production_factor_part_0000.parquet",
                pd.DataFrame(
                    {
                        "trade_date": ["20260627", "20260628"],
                        "stock_code": ["000001.SZ", "000001.SZ"],
                        "close": [10.0, 11.0],
                    }
                ),
            )

            result = sync_feature_parts_full_to_duckdb(
                data_dir,
                parts_dir=parts_dir,
                duckdb_path=duckdb_path,
                table_name="prod_l3_feature_test",
            )

            self.assertEqual(result["source_rows"], 2)
            self.assertEqual(result["target_rows"], 2)
            with duckdb.connect(str(duckdb_path), read_only=True) as conn:
                count = conn.execute("SELECT COUNT(*) FROM prod_l3_feature_test").fetchone()[0]
            self.assertEqual(count, 2)

    def test_feature_full_sync_defaults_to_l3_feature_split_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data_file"
            parts_dir = data_dir / "production_factor_parts"
            expected_duckdb_path = data_dir / "production_assets" / "duckdb" / "l3_feature_current.duckdb"
            self._write_parquet(
                parts_dir / "production_factor_part_0000.parquet",
                pd.DataFrame(
                    {
                        "trade_date": ["20260627"],
                        "stock_code": ["000001.SZ"],
                        "close_qfq": [10.0],
                    }
                ),
            )

            result = sync_feature_parts_full_to_duckdb(
                data_dir,
                parts_dir=parts_dir,
                table_name="prod_l3_feature_test",
            )

            self.assertEqual(Path(result["duckdb_path"]), expected_duckdb_path)
            with duckdb.connect(str(expected_duckdb_path), read_only=True) as conn:
                count = conn.execute("SELECT COUNT(*) FROM prod_l3_feature_test").fetchone()[0]
            self.assertEqual(count, 1)

    def test_feature_full_sync_prefers_active_registry_duckdb_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data_file"
            parts_dir = data_dir / "production_factor_parts"
            registry_dir = data_dir / "asset_registry"
            registry_dir.mkdir(parents=True, exist_ok=True)
            custom_duckdb_path = data_dir / "production_assets" / "duckdb" / "l3_feature_current.duckdb"
            (data_dir / "reports").mkdir(parents=True, exist_ok=True)
            audit_record = data_dir / "reports" / "audit.md"
            audit_record.write_text("approved", encoding="utf-8")
            registry_dir.joinpath("production_assets.json").write_text(
                json.dumps(
                    {
                        "assets": [
                            {
                                "asset_id": "prod_l3_feature_current",
                                "track": "production",
                                "layer": "L3_features",
                                "asset_type": "duckdb_table",
                                "status": "production_active",
                                "allowed_for_main_workflow": True,
                                "asset_path": f"{custom_duckdb_path}::prod_l3_feature_test",
                                "audit_record": str(audit_record),
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            self._write_parquet(
                parts_dir / "production_factor_part_0000.parquet",
                pd.DataFrame(
                    {
                        "trade_date": ["20260627"],
                        "stock_code": ["000001.SZ"],
                        "close": [10.0],
                    }
                ),
            )

            result = sync_feature_parts_full_to_duckdb(
                data_dir,
                parts_dir=parts_dir,
                table_name="prod_l3_feature_test",
            )

            self.assertEqual(Path(result["duckdb_path"]), custom_duckdb_path)
            with duckdb.connect(str(custom_duckdb_path), read_only=True) as conn:
                count = conn.execute("SELECT COUNT(*) FROM prod_l3_feature_test").fetchone()[0]
            self.assertEqual(count, 1)

    def test_feature_target_date_sync_replaces_only_target_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data_file"
            parts_dir = data_dir / "production_factor_parts"
            duckdb_path = data_dir / "production_assets" / "duckdb" / "quant_production.duckdb"
            first = pd.DataFrame(
                {
                    "trade_date": ["20260627", "20260628"],
                    "stock_code": ["000001.SZ", "000001.SZ"],
                    "close": [10.0, 11.0],
                }
            )
            self._write_parquet(parts_dir / "production_factor_part_0000.parquet", first)
            sync_feature_parts_full_to_duckdb(
                data_dir,
                parts_dir=parts_dir,
                duckdb_path=duckdb_path,
                table_name="prod_l3_feature_test",
            )

            second = pd.DataFrame(
                {
                    "trade_date": ["20260627", "20260628", "20260628"],
                    "stock_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
                    "close": [10.0, 21.0, 31.0],
                }
            )
            self._write_parquet(parts_dir / "production_factor_part_0000.parquet", second)
            result = sync_feature_parts_target_date_to_duckdb(
                data_dir,
                "20260628",
                parts_dir=parts_dir,
                duckdb_path=duckdb_path,
                table_name="prod_l3_feature_test",
            )

            self.assertEqual(result["source_rows"], 2)
            self.assertEqual(result["target_rows"], 2)
            self.assertEqual(result["target_total_rows"], 3)
            with duckdb.connect(str(duckdb_path), read_only=True) as conn:
                old_rows = conn.execute(
                    "SELECT COUNT(*) FROM prod_l3_feature_test WHERE trade_date = '20260627'"
                ).fetchone()[0]
                new_rows = conn.execute(
                    "SELECT COUNT(*) FROM prod_l3_feature_test WHERE trade_date = '20260628'"
                ).fetchone()[0]
            self.assertEqual(old_rows, 1)
            self.assertEqual(new_rows, 2)

    def test_feature_full_sync_normalizes_qfq_contract_and_filters_bj(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data_file"
            parts_dir = data_dir / "production_factor_parts"
            duckdb_path = data_dir / "production_assets" / "duckdb" / "l3_feature_current.duckdb"
            l2_duckdb_path = data_dir / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
            self._write_parquet(
                parts_dir / "production_factor_part_0000.parquet",
                pd.DataFrame(
                    {
                        "trade_date": ["20260627", "20260627"],
                        "stock_code": ["000001.SZ", "920001.BJ"],
                        "industry": ["Bank", "BJ"],
                        "close": [10.0, 20.0],
                        "atr": [0.8, 0.9],
                        "atr_qfq": [0.4, 0.5],
                        "macd": [1.1, 1.2],
                        "macd_qfq": [0.6, 0.7],
                    }
                ),
            )
            l2_duckdb_path.parent.mkdir(parents=True, exist_ok=True)
            with duckdb.connect(str(l2_duckdb_path)) as conn:
                conn.execute(
                    """
                    CREATE TABLE STOCK_DAILY_DATA(
                        trade_date TEXT,
                        stock_code TEXT,
                        open_qfq DOUBLE,
                        high_qfq DOUBLE,
                        low_qfq DOUBLE,
                        close_qfq DOUBLE,
                        pre_close_qfq DOUBLE
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO STOCK_DAILY_DATA VALUES
                    ('20260627', '000001.SZ', 5.0, 5.2, 4.9, 5.1, 4.95),
                    ('20260627', '920001.BJ', 8.0, 8.2, 7.9, 8.1, 7.95)
                    """
                )

            result = sync_feature_parts_full_to_duckdb(
                data_dir,
                parts_dir=parts_dir,
                duckdb_path=duckdb_path,
                table_name="prod_l3_feature_test",
            )

            self.assertEqual(result["source_rows"], 1)
            self.assertEqual(result["target_rows"], 1)
            with duckdb.connect(str(duckdb_path), read_only=True) as conn:
                columns = [row[1] for row in conn.execute("PRAGMA table_info('prod_l3_feature_test')").fetchall()]
                rows = conn.execute(
                    "SELECT stock_code, open_qfq, high_qfq, low_qfq, close_qfq, pre_close_qfq, atr_qfq, macd_qfq FROM prod_l3_feature_test"
                ).fetchall()
            self.assertIn("open_qfq", columns)
            self.assertIn("high_qfq", columns)
            self.assertIn("low_qfq", columns)
            self.assertIn("close_qfq", columns)
            self.assertIn("pre_close_qfq", columns)
            self.assertNotIn("open", columns)
            self.assertNotIn("high", columns)
            self.assertNotIn("low", columns)
            self.assertNotIn("close", columns)
            self.assertNotIn("pre_close", columns)
            self.assertNotIn("atr", columns)
            self.assertNotIn("macd", columns)
            self.assertEqual(rows, [("000001.SZ", 5.0, 5.2, 4.9, 5.1, 4.95, 0.4, 0.6)])

    def test_feature_full_sync_renames_legacy_gtja_columns_to_qfq(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data_file"
            parts_dir = data_dir / "production_factor_parts"
            duckdb_path = data_dir / "production_assets" / "duckdb" / "l3_feature_current.duckdb"
            l2_duckdb_path = data_dir / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
            self._write_parquet(
                parts_dir / "production_factor_part_0000.parquet",
                pd.DataFrame(
                    {
                        "trade_date": ["20260627"],
                        "stock_code": ["000001.SZ"],
                        "industry": ["Bank"],
                        "gtja_alpha001": [0.123],
                    }
                ),
            )
            l2_duckdb_path.parent.mkdir(parents=True, exist_ok=True)
            with duckdb.connect(str(l2_duckdb_path)) as conn:
                conn.execute(
                    """
                    CREATE TABLE STOCK_DAILY_DATA(
                        trade_date TEXT,
                        stock_code TEXT,
                        open_qfq DOUBLE,
                        high_qfq DOUBLE,
                        low_qfq DOUBLE,
                        close_qfq DOUBLE,
                        pre_close_qfq DOUBLE
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO STOCK_DAILY_DATA VALUES
                    ('20260627', '000001.SZ', 5.0, 5.2, 4.9, 5.1, 4.95)
                    """
                )

            sync_feature_parts_full_to_duckdb(
                data_dir,
                parts_dir=parts_dir,
                duckdb_path=duckdb_path,
                table_name="prod_l3_feature_test",
            )

            with duckdb.connect(str(duckdb_path), read_only=True) as conn:
                columns = [row[1] for row in conn.execute("PRAGMA table_info('prod_l3_feature_test')").fetchall()]
                rows = conn.execute(
                    "SELECT gtja_alpha001_qfq FROM prod_l3_feature_test"
                ).fetchall()
            self.assertIn("gtja_alpha001_qfq", columns)
            self.assertNotIn("gtja_alpha001", columns)
            self.assertEqual(rows, [(0.123,)])

    def test_feature_full_sync_rewrites_existing_duckdb_table_with_l2_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data_file"
            duckdb_path = data_dir / "production_assets" / "duckdb" / "l3_feature_current.duckdb"
            l2_duckdb_path = data_dir / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
            duckdb_path.parent.mkdir(parents=True, exist_ok=True)
            with duckdb.connect(str(duckdb_path)) as conn:
                conn.execute(
                    """
                    CREATE TABLE prod_l3_feature_test(
                        trade_date TEXT,
                        stock_code TEXT,
                        industry TEXT,
                        close DOUBLE,
                        atr DOUBLE,
                        atr_qfq DOUBLE
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO prod_l3_feature_test VALUES
                    ('20260627', '000001.SZ', 'Bank', 10.0, 0.8, 0.4),
                    ('20260627', '920001.BJ', 'BJ', 20.0, 0.9, 0.5)
                    """
                )
            with duckdb.connect(str(l2_duckdb_path)) as conn:
                conn.execute(
                    """
                    CREATE TABLE STOCK_DAILY_DATA(
                        trade_date TEXT,
                        stock_code TEXT,
                        open_qfq DOUBLE,
                        high_qfq DOUBLE,
                        low_qfq DOUBLE,
                        close_qfq DOUBLE,
                        pre_close_qfq DOUBLE
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO STOCK_DAILY_DATA VALUES
                    ('20260627', '000001.SZ', 5.0, 5.2, 4.9, 5.1, 4.95),
                    ('20260627', '920001.BJ', 8.0, 8.2, 7.9, 8.1, 7.95)
                    """
                )

            result = sync_feature_parts_full_to_duckdb(
                data_dir,
                duckdb_path=duckdb_path,
                table_name="prod_l3_feature_test",
            )

            self.assertEqual(result["mode"], "rewrite_existing_duckdb_table")
            self.assertEqual(result["source_rows"], 2)
            self.assertEqual(result["target_rows"], 1)
            with duckdb.connect(str(duckdb_path), read_only=True) as conn:
                columns = [row[1] for row in conn.execute("PRAGMA table_info('prod_l3_feature_test')").fetchall()]
                rows = conn.execute(
                    "SELECT stock_code, open_qfq, high_qfq, low_qfq, close_qfq, pre_close_qfq, atr_qfq FROM prod_l3_feature_test"
                ).fetchall()
            self.assertNotIn("close", columns)
            self.assertNotIn("atr", columns)
            self.assertIn("close_qfq", columns)
            self.assertIn("atr_qfq", columns)
            self.assertEqual(rows, [("000001.SZ", 5.0, 5.2, 4.9, 5.1, 4.95, 0.4)])

    def test_label_full_and_target_date_sync(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data_file"
            parts_dir = data_dir / "prediction_label_parts"
            duckdb_path = data_dir / "production_assets" / "duckdb" / "quant_production.duckdb"
            self._write_parquet(
                parts_dir / "prediction_label_part_0000.parquet",
                pd.DataFrame(
                    {
                        "trade_date": ["20260627", "20260628"],
                        "stock_code": ["000001.SZ", "000001.SZ"],
                        "executable_5d_open_return": [0.1, 0.2],
                    }
                ),
            )
            full = sync_label_parts_full_to_duckdb(
                data_dir,
                parts_dir=parts_dir,
                duckdb_path=duckdb_path,
                table_name="prod_l3_label_test",
            )
            self.assertEqual(full["target_rows"], 2)

            self._write_parquet(
                parts_dir / "prediction_label_part_0000.parquet",
                pd.DataFrame(
                    {
                        "trade_date": ["20260627", "20260628", "20260628"],
                        "stock_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
                        "executable_5d_open_return": [0.1, 0.3, 0.4],
                    }
                ),
            )
            inc = sync_label_parts_target_date_to_duckdb(
                data_dir,
                "20260628",
                parts_dir=parts_dir,
                duckdb_path=duckdb_path,
                table_name="prod_l3_label_test",
            )
            self.assertEqual(inc["source_rows"], 2)
            self.assertEqual(inc["target_rows"], 2)
            self.assertEqual(inc["target_total_rows"], 3)
            with duckdb.connect(str(duckdb_path), read_only=True) as conn:
                total = conn.execute("SELECT COUNT(*) FROM prod_l3_label_test").fetchone()[0]
                target_rows = conn.execute(
                    "SELECT COUNT(*) FROM prod_l3_label_test WHERE trade_date = '20260628'"
                ).fetchone()[0]
            self.assertEqual(total, 3)
            self.assertEqual(target_rows, 2)

    def test_label_full_sync_defaults_to_l3_label_split_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data_file"
            parts_dir = data_dir / "prediction_label_parts"
            expected_duckdb_path = data_dir / "production_assets" / "duckdb" / "l3_label_current.duckdb"
            self._write_parquet(
                parts_dir / "prediction_label_part_0000.parquet",
                pd.DataFrame(
                    {
                        "trade_date": ["20260627"],
                        "stock_code": ["000001.SZ"],
                        "executable_5d_open_return": [0.1],
                    }
                ),
            )

            result = sync_label_parts_full_to_duckdb(
                data_dir,
                parts_dir=parts_dir,
                table_name="prod_l3_label_test",
            )

            self.assertEqual(Path(result["duckdb_path"]), expected_duckdb_path)
            with duckdb.connect(str(expected_duckdb_path), read_only=True) as conn:
                count = conn.execute("SELECT COUNT(*) FROM prod_l3_label_test").fetchone()[0]
            self.assertEqual(count, 1)

    def test_label_full_sync_filters_bj_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data_file"
            parts_dir = data_dir / "prediction_label_parts"
            duckdb_path = data_dir / "production_assets" / "duckdb" / "l3_label_current.duckdb"
            self._write_parquet(
                parts_dir / "prediction_label_part_0000.parquet",
                pd.DataFrame(
                    {
                        "trade_date": ["20260627", "20260627"],
                        "stock_code": ["000001.SZ", "920001.BJ"],
                        "executable_5d_open_return": [0.1, 0.2],
                    }
                ),
            )

            result = sync_label_parts_full_to_duckdb(
                data_dir,
                parts_dir=parts_dir,
                duckdb_path=duckdb_path,
                table_name="prod_l3_label_test",
            )

            self.assertEqual(result["source_rows"], 1)
            self.assertEqual(result["target_rows"], 1)
            with duckdb.connect(str(duckdb_path), read_only=True) as conn:
                rows = conn.execute("SELECT stock_code FROM prod_l3_label_test").fetchall()
            self.assertEqual(rows, [("000001.SZ",)])


if __name__ == "__main__":
    unittest.main()
