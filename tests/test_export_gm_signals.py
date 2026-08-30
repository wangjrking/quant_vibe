import json
import importlib.util
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from adjustment_semantics import default_adjustment_semantics, default_market_field_semantics
from export_gm_signals import main as export_gm_signals_main
from export_gm_signals import resolve_market_db_path, resolve_prediction_source


class ExportGmSignalsTests(unittest.TestCase):
    def test_resolve_prediction_source_requires_manifest_unless_legacy_mode(self):
        with self.assertRaisesRegex(ValueError, "prediction manifest"):
            resolve_prediction_source(
                prediction_manifest=None,
                legacy_reproduction=False,
                db_path=None,
                table=None,
            )

    def test_resolve_prediction_source_requires_explicit_db_and_table_in_legacy_mode(self):
        with self.assertRaisesRegex(ValueError, "db and table"):
            resolve_prediction_source(
                prediction_manifest=None,
                legacy_reproduction=True,
                db_path="data_file/odb.db",
                table=None,
            )

    def test_resolve_prediction_source_rejects_sqlite_manifest_in_production_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "prediction_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "asset_role": "l4_formal_prediction_asset",
                        "approval_status": "approved_for_l5",
                        "source_type": "sqlite_table",
                        "db_path": "predictions.db",
                        "table": "l4_prediction_table",
                        "market_db_path": "market.db",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "DuckDB-only"):
                resolve_prediction_source(
                    prediction_manifest=str(manifest_path),
                    legacy_reproduction=False,
                    db_path=None,
                    table=None,
                )

    def test_resolve_market_db_path_defaults_to_active_l2_route(self):
        source = {"market_db_path": None}
        expected = Path("D:/stub/quant_production.duckdb")
        with patch("prediction_manifest.resolve_stock_daily_backend", return_value="duckdb"):
            with patch("prediction_manifest.resolve_stock_daily_duckdb_path", return_value=expected):
                self.assertEqual(resolve_market_db_path(source, market_db_path=None), expected)

    def test_resolve_prediction_source_rejects_legacy_odb_manifest_for_production_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "prediction_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "asset_role": "l4_formal_prediction_asset",
                        "approval_status": "approved_for_l5",
                        "source_type": "sqlite_table",
                        "db_path": "odb.db",
                        "table": "stock_predict_data_legacy",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "legacy"):
                resolve_prediction_source(
                    prediction_manifest=str(manifest_path),
                    legacy_reproduction=False,
                    db_path=None,
                    table=None,
                )

    @unittest.skipIf(importlib.util.find_spec("duckdb") is None, "duckdb is not installed")
    def test_resolve_prediction_source_loads_audited_duckdb_manifest(self):
        import duckdb

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            db_path = root / "predictions.duckdb"
            audit_record = root / "audit.md"
            migration_manifest = root / "duckdb_migration_manifest.json"
            manifest_path = root / "prediction_manifest.json"
            audit_record.write_text("通过", encoding="utf-8")
            migration_manifest.write_text("{}", encoding="utf-8")
            with duckdb.connect(str(db_path)) as conn:
                conn.execute("CREATE TABLE pred_table(trade_date VARCHAR, stock_code VARCHAR, pred_prob DOUBLE)")
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "asset_role": "l4_formal_prediction_asset",
                        "approval_status": "approved_for_l5",
                        "source_type": "duckdb_table",
                        "db_path": "predictions.duckdb",
                        "table": "pred_table",
                        "audit_record": "audit.md",
                        "duckdb_migration_manifest": "duckdb_migration_manifest.json",
                        "adjustment_semantics": default_adjustment_semantics(),
                        "market_field_semantics": default_market_field_semantics(),
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            source = resolve_prediction_source(
                prediction_manifest=str(manifest_path),
                legacy_reproduction=False,
                db_path=None,
                table=None,
            )

        self.assertEqual(source["source_type"], "duckdb_table")
        self.assertEqual(source["db_path"], db_path)
        self.assertEqual(source["table"], "pred_table")

    def test_resolve_prediction_source_rejects_unaudited_duckdb_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            db_path = root / "predictions.duckdb"
            db_path.write_bytes(b"duckdb")
            manifest_path = root / "prediction_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "asset_role": "l4_formal_prediction_asset",
                        "approval_status": "approved_for_l5",
                        "source_type": "duckdb_table",
                        "db_path": "predictions.duckdb",
                        "table": "pred_table",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "audit_record"):
                resolve_prediction_source(
                    prediction_manifest=str(manifest_path),
                    legacy_reproduction=False,
                    db_path=None,
                    table=None,
                )

    def test_export_gm_signals_rejects_sqlite_prediction_manifest_in_production_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            prediction_db = root / "predictions.db"
            market_db = root / "market.db"
            output_csv = root / "signals.csv"
            manifest_path = root / "prediction_manifest.json"

            conn = sqlite3.connect(prediction_db)
            conn.execute(
                """
                CREATE TABLE pred_table (
                    trade_date TEXT,
                    stock_code TEXT,
                    pred_prob REAL
                )
                """
            )
            conn.executemany(
                "INSERT INTO pred_table (trade_date, stock_code, pred_prob) VALUES (?, ?, ?)",
                [
                    ("20240604", "000001.SZ", 0.91),
                    ("20240605", "000001.SZ", 0.10),
                ],
            )
            conn.commit()
            conn.close()

            conn = sqlite3.connect(market_db)
            conn.execute(
                """
                CREATE TABLE STOCK_DAILY_DATA (
                    trade_date TEXT,
                    stock_code TEXT,
                    name TEXT,
                    pre_close REAL,
                    open REAL,
                    close REAL,
                    amount REAL,
                    turnover_rate REAL,
                    total_mv REAL,
                    limit_times TEXT
                )
                """
            )
            conn.executemany(
                """
                INSERT INTO STOCK_DAILY_DATA (
                    trade_date, stock_code, name, pre_close, open, close, amount, turnover_rate, total_mv, limit_times
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    ("20240604", "000001.SZ", "平安银行", 10.0, 10.1, 10.2, 1200000.0, 3.5, 900000.0, None),
                    ("20240605", "000001.SZ", "平安银行", 10.1, 10.2, 10.3, 1200000.0, 3.5, 900000.0, None),
                ],
            )
            conn.commit()
            conn.close()

            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "asset_role": "l4_formal_prediction_asset",
                        "approval_status": "approved_for_l5",
                        "source_type": "sqlite_table",
                        "db_path": str(prediction_db),
                        "table": "pred_table",
                        "market_db_path": str(market_db),
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "DuckDB-only"):
                export_gm_signals_main(
                    [
                        "--prediction-manifest",
                        str(manifest_path),
                        "--market-db",
                        str(market_db),
                        "--start",
                        "20240604",
                        "--end",
                        "20240605",
                        "--top-k",
                        "1",
                        "--min-pred",
                        "none",
                        "--min-pred-quantile",
                        "0.95",
                        "--max-atr-ratio",
                        "none",
                        "--min-amount",
                        "800000",
                        "--min-turnover-rate",
                        "2.0",
                        "--max-total-mv",
                        "1000000",
                        "--max-positions",
                        "1",
                        "--holding-days",
                        "5",
                        "--weight-mode",
                        "equal",
                        "--target-total-pct",
                        "0.98",
                        "--output",
                        str(output_csv),
                    ]
                )

    @unittest.skipIf(importlib.util.find_spec("duckdb") is None, "duckdb is not installed")
    def test_export_gm_signals_supports_duckdb_prediction_and_market_assets(self):
        import duckdb

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            prediction_db = root / "predictions.duckdb"
            market_db = root / "market.duckdb"
            output_csv = root / "signals.csv"
            manifest_path = root / "prediction_manifest.json"
            audit_record = root / "audit.md"
            migration_manifest = root / "duckdb_migration_manifest.json"

            audit_record.write_text("通过", encoding="utf-8")
            migration_manifest.write_text("{}", encoding="utf-8")

            with duckdb.connect(str(prediction_db)) as conn:
                conn.execute(
                    """
                    CREATE TABLE pred_table (
                        trade_date VARCHAR,
                        stock_code VARCHAR,
                        pred_prob DOUBLE
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO pred_table VALUES
                    ('20240604', '000001.SZ', 0.91),
                    ('20240605', '000001.SZ', 0.10)
                    """
                )

            with duckdb.connect(str(market_db)) as conn:
                conn.execute(
                    """
                    CREATE TABLE STOCK_DAILY_DATA (
                        trade_date VARCHAR,
                        stock_code VARCHAR,
                        name VARCHAR,
                        pre_close DOUBLE,
                        open DOUBLE,
                        close DOUBLE,
                        close_qfq DOUBLE,
                        amount DOUBLE,
                        turnover_rate DOUBLE,
                        total_mv DOUBLE,
                        atr_qfq DOUBLE,
                        limit_times VARCHAR
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO STOCK_DAILY_DATA VALUES
                    ('20240604', '000001.SZ', '平安银行', 10.0, 10.1, 10.2, 10.2, 1200000.0, 3.5, 900000.0, 0.2, NULL),
                    ('20240605', '000001.SZ', '平安银行', 10.1, 10.2, 10.3, 10.3, 1200000.0, 3.5, 900000.0, 0.2, NULL)
                    """
                )

            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "asset_role": "l4_formal_prediction_asset",
                        "approval_status": "approved_for_l5",
                        "source_type": "duckdb_table",
                        "db_path": str(prediction_db),
                        "table": "pred_table",
                        "market_db_path": str(market_db),
                        "audit_record": str(audit_record),
                        "duckdb_migration_manifest": str(migration_manifest),
                        "adjustment_semantics": default_adjustment_semantics(),
                        "market_field_semantics": default_market_field_semantics(),
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            export_gm_signals_main(
                [
                    "--prediction-manifest",
                    str(manifest_path),
                    "--start",
                    "20240604",
                    "--end",
                    "20240605",
                    "--top-k",
                    "1",
                    "--min-pred",
                    "none",
                    "--min-pred-quantile",
                    "0.95",
                    "--max-atr-ratio",
                    "none",
                    "--min-amount",
                    "800000",
                    "--min-turnover-rate",
                    "2.0",
                    "--max-total-mv",
                    "1000000",
                    "--max-positions",
                    "1",
                    "--holding-days",
                    "5",
                    "--weight-mode",
                    "equal",
                    "--target-total-pct",
                    "0.98",
                    "--output",
                    str(output_csv),
                ]
            )

            self.assertTrue(output_csv.exists())
            content = output_csv.read_text(encoding="utf-8-sig")
            self.assertIn("SZSE.000001", content)

    @unittest.skipIf(importlib.util.find_spec("duckdb") is None, "duckdb is not installed")
    def test_export_gm_signals_duckdb_market_route_uses_explicit_close_qfq(self):
        import duckdb

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            prediction_db = root / "predictions.duckdb"
            market_db = root / "market.duckdb"
            output_csv = root / "signals.csv"
            manifest_path = root / "prediction_manifest.json"
            audit_record = root / "audit.md"
            migration_manifest = root / "duckdb_migration_manifest.json"

            audit_record.write_text("approved", encoding="utf-8")
            migration_manifest.write_text("{}", encoding="utf-8")

            with duckdb.connect(str(prediction_db)) as conn:
                conn.execute(
                    """
                    CREATE TABLE pred_table (
                        trade_date VARCHAR,
                        stock_code VARCHAR,
                        pred_prob DOUBLE
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO pred_table VALUES
                    ('20240604', '000001.SZ', 0.91)
                    """
                )

            with duckdb.connect(str(market_db)) as conn:
                conn.execute(
                    """
                    CREATE TABLE STOCK_DAILY_DATA (
                        trade_date VARCHAR,
                        stock_code VARCHAR,
                        name VARCHAR,
                        pre_close DOUBLE,
                        open DOUBLE,
                        close DOUBLE,
                        close_qfq DOUBLE,
                        amount DOUBLE,
                        turnover_rate DOUBLE,
                        total_mv DOUBLE,
                        atr_qfq DOUBLE,
                        limit_times VARCHAR
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO STOCK_DAILY_DATA VALUES
                    ('20240604', '000001.SZ', '平安银行', 10.0, 10.1, 99.9, 10.2, 1200000.0, 3.5, 900000.0, 0.2, NULL),
                    ('20240605', '000001.SZ', '平安银行', 10.1, 10.2, 88.8, 10.3, 1200000.0, 3.5, 900000.0, 0.2, NULL)
                    """
                )

            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "asset_role": "l4_formal_prediction_asset",
                        "approval_status": "approved_for_l5",
                        "source_type": "duckdb_table",
                        "db_path": str(prediction_db),
                        "table": "pred_table",
                        "market_db_path": str(market_db),
                        "audit_record": str(audit_record),
                        "duckdb_migration_manifest": str(migration_manifest),
                        "adjustment_semantics": default_adjustment_semantics(),
                        "market_field_semantics": default_market_field_semantics(),
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            export_gm_signals_main(
                [
                    "--prediction-manifest",
                    str(manifest_path),
                    "--start",
                    "20240604",
                    "--end",
                    "20240605",
                    "--top-k",
                    "1",
                    "--min-pred",
                    "none",
                    "--min-pred-quantile",
                    "0.0",
                    "--max-atr-ratio",
                    "0.03",
                    "--min-amount",
                    "800000",
                    "--min-turnover-rate",
                    "2.0",
                    "--max-total-mv",
                    "1000000",
                    "--max-positions",
                    "1",
                    "--holding-days",
                    "5",
                    "--weight-mode",
                    "equal",
                    "--target-total-pct",
                    "0.98",
                    "--output",
                    str(output_csv),
                ]
            )

            self.assertTrue(output_csv.exists())
            content = output_csv.read_text(encoding="utf-8-sig")
            self.assertIn("SZSE.000001", content)


if __name__ == "__main__":
    unittest.main()
