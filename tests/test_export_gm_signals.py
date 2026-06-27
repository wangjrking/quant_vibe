import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

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

    def test_resolve_prediction_source_loads_approved_manifest(self):
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

            source = resolve_prediction_source(
                prediction_manifest=str(manifest_path),
                legacy_reproduction=False,
                db_path=None,
                table=None,
            )

            market_db = resolve_market_db_path(source, market_db_path=None)

        self.assertEqual(source["table"], "l4_prediction_table")
        self.assertEqual(source["db_path"], root / "predictions.db")
        self.assertEqual(market_db, root / "market.db")

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

    def test_export_gm_signals_enriches_missing_liquidity_fields_from_market_db(self):
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
                    trade_date, stock_code, name, pre_close, open, amount, turnover_rate, total_mv, limit_times
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    ("20240604", "000001.SZ", "平安银行", 10.0, 10.1, 1200000.0, 3.5, 900000.0, None),
                    ("20240605", "000001.SZ", "平安银行", 10.1, 10.2, 1200000.0, 3.5, 900000.0, None),
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

            self.assertTrue(output_csv.exists())
            content = output_csv.read_text(encoding="utf-8-sig")
            self.assertIn("SZSE.000001", content)


if __name__ == "__main__":
    unittest.main()
