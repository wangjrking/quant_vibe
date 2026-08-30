import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

MAIN_DIR = Path(__file__).resolve().parents[1]
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from l5_duckdb_sync import resolve_l5_duckdb_path, resolve_l5_table_duckdb_path, sync_strategy_registry_to_duckdb
from l7_duckdb_sync import resolve_l7_duckdb_path, resolve_l7_table_duckdb_path, sync_production_signal_artifacts_to_duckdb


@unittest.skipIf(importlib.util.find_spec("duckdb") is None, "duckdb is not installed")
class L5L7DuckDBSyncTests(unittest.TestCase):
    def test_l5_manifest_sync_falls_back_to_signal_date_and_buy_date(self):
        import duckdb

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            strategy_dir = root / "strategy_library" / "production" / "prod_a"
            strategy_dir.mkdir(parents=True)
            registry_path = root / "strategy_library" / "registry.json"
            registry_path.write_text(
                json.dumps(
                    {
                        "production": {
                            "current": "prod_a",
                            "strategies": [
                                {
                                    "strategy_id": "prod_a",
                                    "name": "Prod A",
                                    "path": "strategy_library/production/prod_a",
                                    "status": "production",
                                    "published_at": "2026-06-28",
                                }
                            ],
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (strategy_dir / "strategy_manifest.json").write_text(
                json.dumps(
                    {
                        "strategy_id": "prod_a",
                        "name": "Prod A",
                        "status": "production",
                        "published_at": "2026-06-28",
                        "current_signal": {
                            "signal_date": "20260630",
                            "buy_date": "20260701",
                            "latest_file": "signals/prod_a_latest.csv",
                        },
                        "input_contract": {},
                        "validation": {},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            duckdb_path = root / "data_file" / "production_assets" / "duckdb" / "quant_production.duckdb"

            sync_strategy_registry_to_duckdb(
                project_dir=root,
                data_dir=root / "data_file",
                registry_path=registry_path,
                duckdb_path=duckdb_path,
            )

            with duckdb.connect(str(duckdb_path), read_only=True) as conn:
                manifest_row = conn.execute(
                    'SELECT latest_signal_date, latest_buy_date FROM "prod_l5_strategy_manifest_current"'
                ).fetchone()

            self.assertEqual(manifest_row, ("20260630", "20260701"))

    def test_l5_manifest_sync_backfills_formal_manifest_from_formal_manifest_1d(self):
        import duckdb

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            strategy_dir = root / "strategy_library" / "production" / "prod_a"
            strategy_dir.mkdir(parents=True)
            registry_path = root / "strategy_library" / "registry.json"
            registry_path.write_text(
                json.dumps(
                    {
                        "production": {
                            "current": "prod_a",
                            "strategies": [
                                {
                                    "strategy_id": "prod_a",
                                    "name": "Prod A",
                                    "path": "strategy_library/production/prod_a",
                                    "status": "production",
                                }
                            ],
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (strategy_dir / "strategy_manifest.json").write_text(
                json.dumps(
                    {
                        "strategy_id": "prod_a",
                        "name": "Prod A",
                        "status": "production",
                        "current_signal": {
                            "latest_signal_date": "20260717",
                            "latest_buy_date": "20260720",
                            "latest_file": "signals/prod_a_latest.csv",
                        },
                        "input_contract": {
                            "formal_manifest_1d": "config/prediction_manifests/prod_a_1d.json",
                            "formal_manifest_3d": "config/prediction_manifests/prod_a_3d.json",
                            "formal_manifest_5d": "config/prediction_manifests/prod_a_5d.json",
                            "formal_manifest_10d": "config/prediction_manifests/prod_a_10d.json",
                        },
                        "validation": {},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            duckdb_path = root / "data_file" / "production_assets" / "duckdb" / "quant_production.duckdb"

            sync_strategy_registry_to_duckdb(
                project_dir=root,
                data_dir=root / "data_file",
                registry_path=registry_path,
                duckdb_path=duckdb_path,
            )

            with duckdb.connect(str(duckdb_path), read_only=True) as conn:
                manifest_row = conn.execute(
                    'SELECT formal_manifest, formal_manifest_3d, formal_manifest_5d, formal_manifest_10d FROM "prod_l5_strategy_manifest_current"'
                ).fetchone()

            self.assertEqual(
                manifest_row,
                (
                    "config/prediction_manifests/prod_a_1d.json",
                    "config/prediction_manifests/prod_a_3d.json",
                    "config/prediction_manifests/prod_a_5d.json",
                    "config/prediction_manifests/prod_a_10d.json",
                ),
            )

    def test_l5_resolve_default_paths_use_single_table_duckdb_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data_file"
            expected_registry = (
                data_dir
                / "production_assets"
                / "duckdb"
                / "production"
                / "l5"
                / "prod_l5_strategy_registry_current.duckdb"
            )
            expected_manifest = (
                data_dir
                / "production_assets"
                / "duckdb"
                / "production"
                / "l5"
                / "prod_l5_strategy_manifest_current.duckdb"
            )
            self.assertEqual(resolve_l5_duckdb_path(data_dir=data_dir), expected_registry)
            self.assertEqual(
                resolve_l5_table_duckdb_path("prod_l5_strategy_manifest_current", data_dir=data_dir),
                expected_manifest,
            )

    def test_l7_resolve_default_paths_use_single_table_duckdb_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data_file"
            expected_rows = (
                data_dir
                / "production_assets"
                / "duckdb"
                / "production"
                / "l7"
                / "prod_l7_signal_rows_current.duckdb"
            )
            expected_status = (
                data_dir
                / "production_assets"
                / "duckdb"
                / "production"
                / "l7"
                / "prod_l7_signal_status_current.duckdb"
            )
            self.assertEqual(resolve_l7_duckdb_path(data_dir=data_dir), expected_rows)
            self.assertEqual(
                resolve_l7_table_duckdb_path("prod_l7_signal_status_current", data_dir=data_dir),
                expected_status,
            )

    def test_sync_strategy_registry_to_single_table_duckdb_files(self):
        import duckdb

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "data_file"
            registry_dir = data_dir / "asset_registry"
            registry_dir.mkdir(parents=True)
            registry_duckdb = data_dir / "production_assets" / "duckdb" / "production" / "l5" / "prod_l5_strategy_registry_current.duckdb"
            manifest_duckdb = data_dir / "production_assets" / "duckdb" / "production" / "l5" / "prod_l5_strategy_manifest_current.duckdb"
            registry_duckdb.parent.mkdir(parents=True, exist_ok=True)
            (registry_dir / "production_assets.json").write_text(
                json.dumps(
                    {
                        "assets": [
                            {
                                "asset_id": "prod_l5_registry_current",
                                "layer": "l5_strategy_signal",
                                "asset_type": "duckdb_strategy_registry",
                                "status": "production_active",
                                "allowed_for_main_workflow": True,
                                "asset_path": f"{registry_duckdb}::prod_l5_strategy_registry_current",
                            },
                            {
                                "asset_id": "prod_l5_manifest_current",
                                "layer": "l5_strategy_signal",
                                "asset_type": "duckdb_strategy_manifest",
                                "status": "production_active",
                                "allowed_for_main_workflow": True,
                                "asset_path": f"{manifest_duckdb}::prod_l5_strategy_manifest_current",
                            },
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            strategy_dir = root / "strategy_library" / "production" / "prod_a"
            strategy_dir.mkdir(parents=True)
            registry_path = root / "strategy_library" / "registry.json"
            registry_path.write_text(
                json.dumps(
                    {
                        "production": {
                            "current": "prod_a",
                            "strategies": [
                                {
                                    "strategy_id": "prod_a",
                                    "name": "Prod A",
                                    "path": "strategy_library/production/prod_a",
                                    "status": "production",
                                }
                            ],
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (strategy_dir / "strategy_manifest.json").write_text(
                json.dumps(
                    {
                        "strategy_id": "prod_a",
                        "name": "Prod A",
                        "status": "production",
                        "current_signal": {},
                        "input_contract": {},
                        "validation": {},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            result = sync_strategy_registry_to_duckdb(
                project_dir=root,
                data_dir=data_dir,
                registry_path=registry_path,
            )

            self.assertEqual(Path(result["registry_duckdb_path"]), registry_duckdb)
            self.assertEqual(Path(result["manifest_duckdb_path"]), manifest_duckdb)
            with duckdb.connect(str(registry_duckdb), read_only=True) as conn:
                registry_value = conn.execute('SELECT strategy_id FROM "prod_l5_strategy_registry_current"').fetchone()[0]
            with duckdb.connect(str(manifest_duckdb), read_only=True) as conn:
                manifest_value = conn.execute('SELECT strategy_id FROM "prod_l5_strategy_manifest_current"').fetchone()[0]
            self.assertEqual(registry_value, "prod_a")
            self.assertEqual(manifest_value, "prod_a")

    def test_sync_strategy_registry_to_duckdb_materializes_registry_and_manifests(self):
        import duckdb

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            strategy_dir = root / "strategy_library" / "production" / "prod_a"
            strategy_dir.mkdir(parents=True)
            registry_path = root / "strategy_library" / "registry.json"
            registry_path.write_text(
                json.dumps(
                    {
                        "production": {
                            "current": "prod_a",
                            "strategies": [
                                {
                                    "strategy_id": "prod_a",
                                    "name": "Prod A",
                                    "path": "strategy_library/production/prod_a",
                                    "status": "production",
                                    "published_at": "2026-06-28",
                                    "annual_return": 1.23,
                                    "sharpe": 2.34,
                                    "max_drawdown": 0.45,
                                    "validation_platform": "juejin",
                                }
                            ],
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (strategy_dir / "strategy_manifest.json").write_text(
                json.dumps(
                    {
                        "strategy_id": "prod_a",
                        "name": "Prod A",
                        "status": "production",
                        "published_at": "2026-06-28",
                        "production_version": "v20260628",
                        "current_signal": {
                            "latest_file": "signals/prod_a_latest.csv",
                            "latest_signal_date": "20260627",
                            "latest_buy_date": "20260628",
                        },
                        "input_contract": {
                            "formal_manifest": "config/prediction_manifests/prod_a.json",
                            "formal_manifest_3d": "config/prediction_manifests/prod_a_3d.json",
                            "formal_manifest_5d": "config/prediction_manifests/prod_a_5d.json",
                            "formal_manifest_10d": "config/prediction_manifests/prod_a_10d.json",
                        },
                        "validation": {
                            "annual_return": 1.23,
                            "sharpe": 2.34,
                            "max_drawdown": 0.45,
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            duckdb_path = root / "data_file" / "production_assets" / "duckdb" / "quant_production.duckdb"

            result = sync_strategy_registry_to_duckdb(
                project_dir=root,
                data_dir=root / "data_file",
                registry_path=registry_path,
                duckdb_path=duckdb_path,
            )

            self.assertEqual(result["registry_rows"], 1)
            self.assertEqual(result["manifest_rows"], 1)
            with duckdb.connect(str(duckdb_path), read_only=True) as conn:
                registry_row = conn.execute(
                    'SELECT strategy_id, is_current_production FROM "prod_l5_strategy_registry_current"'
                ).fetchone()
                manifest_row = conn.execute(
                    'SELECT strategy_id, latest_signal_date FROM "prod_l5_strategy_manifest_current"'
                ).fetchone()

            self.assertEqual(registry_row, ("prod_a", True))
            self.assertEqual(manifest_row, ("prod_a", "20260627"))

    def test_sync_production_signal_artifacts_to_duckdb_materializes_csv_and_json(self):
        import duckdb

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            registry_path = root / "strategy_library" / "registry.json"
            registry_path.parent.mkdir(parents=True)
            registry_path.write_text(
                json.dumps(
                    {
                        "production": {
                            "current": "prod_a",
                            "strategies": [
                                {
                                    "strategy_id": "prod_a",
                                    "name": "Prod A",
                                    "path": "strategy_library/production/prod_a",
                                    "status": "production",
                                }
                            ],
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            signal_dir = root / "data_file" / "production_signals"
            signal_dir.mkdir(parents=True)
            csv_path = signal_dir / "prod_a_latest.csv"
            with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=["signal_date", "buy_date", "stock_code", "symbol", "pred_prob", "target_pct"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "signal_date": "20260627",
                        "buy_date": "20260630",
                        "stock_code": "000001.SZ",
                        "symbol": "SZSE.000001",
                        "pred_prob": "0.91",
                        "target_pct": "0.98",
                    }
                )
            status_path = signal_dir / "prod_a_latest_status.json"
            status_path.write_text(
                json.dumps(
                    {
                        "status": "ok",
                        "latest_signal_date": "20260627",
                        "latest_buy_date": "20260630",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            duckdb_path = root / "data_file" / "production_assets" / "duckdb" / "quant_production.duckdb"

            result = sync_production_signal_artifacts_to_duckdb(
                data_dir=root / "data_file",
                signal_dir=signal_dir,
                duckdb_path=duckdb_path,
                project_dir=root,
                registry_path=registry_path,
            )

            self.assertEqual(result["signal_file_count"], 2)
            self.assertEqual(result["signal_row_count"], 1)
            self.assertEqual(result["status_row_count"], 1)
            with duckdb.connect(str(duckdb_path), read_only=True) as conn:
                signal_row = conn.execute(
                    'SELECT strategy_id, stock_code, buy_date FROM "prod_l7_signal_rows_current"'
                ).fetchone()
                status_row = conn.execute(
                    'SELECT strategy_id, latest_signal_date, status FROM "prod_l7_signal_status_current"'
                ).fetchone()

            self.assertEqual(signal_row, ("prod_a", "000001.SZ", "20260630"))
            self.assertEqual(status_row, ("prod_a", "20260627", "ok"))

    def test_sync_production_signal_artifacts_to_single_table_duckdb_files(self):
        import duckdb

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "data_file"
            registry_dir = data_dir / "asset_registry"
            registry_dir.mkdir(parents=True)
            signal_rows_duckdb = data_dir / "production_assets" / "duckdb" / "production" / "l7" / "prod_l7_signal_rows_current.duckdb"
            signal_status_duckdb = data_dir / "production_assets" / "duckdb" / "production" / "l7" / "prod_l7_signal_status_current.duckdb"
            signal_files_duckdb = data_dir / "production_assets" / "duckdb" / "production" / "l7" / "prod_l7_signal_files_current.duckdb"
            signal_rows_duckdb.parent.mkdir(parents=True, exist_ok=True)
            (registry_dir / "production_assets.json").write_text(
                json.dumps(
                    {
                        "assets": [
                            {
                                "asset_id": "prod_l7_signal_rows_current",
                                "layer": "l7_trading_delivery",
                                "asset_type": "duckdb_signal_rows",
                                "status": "production_active",
                                "allowed_for_main_workflow": True,
                                "asset_path": f"{signal_rows_duckdb}::prod_l7_signal_rows_current",
                            },
                            {
                                "asset_id": "prod_l7_signal_status_current",
                                "layer": "l7_trading_delivery",
                                "asset_type": "duckdb_signal_status",
                                "status": "production_active",
                                "allowed_for_main_workflow": True,
                                "asset_path": f"{signal_status_duckdb}::prod_l7_signal_status_current",
                            },
                            {
                                "asset_id": "prod_l7_signal_files_current",
                                "layer": "l7_trading_delivery",
                                "asset_type": "duckdb_signal_files",
                                "status": "production_active",
                                "allowed_for_main_workflow": True,
                                "asset_path": f"{signal_files_duckdb}::prod_l7_signal_files_current",
                            },
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            registry_path = root / "strategy_library" / "registry.json"
            registry_path.parent.mkdir(parents=True)
            registry_path.write_text(
                json.dumps(
                    {
                        "production": {
                            "current": "prod_a",
                            "strategies": [{"strategy_id": "prod_a", "status": "production"}],
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            signal_dir = data_dir / "production_signals"
            signal_dir.mkdir(parents=True)
            with (signal_dir / "prod_a_latest.csv").open("w", encoding="utf-8-sig", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=["signal_date", "buy_date", "stock_code", "symbol", "pred_prob", "target_pct"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "signal_date": "20260627",
                        "buy_date": "20260630",
                        "stock_code": "000001.SZ",
                        "symbol": "SZSE.000001",
                        "pred_prob": "0.91",
                        "target_pct": "0.98",
                    }
                )
            (signal_dir / "prod_a_latest_status.json").write_text(
                json.dumps({"status": "ok", "latest_signal_date": "20260627", "latest_buy_date": "20260630"}, ensure_ascii=False),
                encoding="utf-8",
            )

            result = sync_production_signal_artifacts_to_duckdb(
                data_dir=data_dir,
                signal_dir=signal_dir,
                project_dir=root,
                registry_path=registry_path,
            )

            self.assertEqual(Path(result["signal_rows_duckdb_path"]), signal_rows_duckdb)
            self.assertEqual(Path(result["signal_status_duckdb_path"]), signal_status_duckdb)
            self.assertEqual(Path(result["signal_files_duckdb_path"]), signal_files_duckdb)
            with duckdb.connect(str(signal_rows_duckdb), read_only=True) as conn:
                row_value = conn.execute('SELECT stock_code FROM "prod_l7_signal_rows_current"').fetchone()[0]
            with duckdb.connect(str(signal_status_duckdb), read_only=True) as conn:
                status_value = conn.execute('SELECT status FROM "prod_l7_signal_status_current"').fetchone()[0]
            with duckdb.connect(str(signal_files_duckdb), read_only=True) as conn:
                file_value = conn.execute('SELECT file_name FROM "prod_l7_signal_files_current"').fetchone()[0]
            self.assertEqual(row_value, "000001.SZ")
            self.assertEqual(status_value, "ok")
            self.assertEqual(file_value, "prod_a_latest.csv")

    def test_l7_sync_preserves_header_only_no_signal_batch_without_action_rows(self):
        import duckdb

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "data_file"
            registry_path = root / "strategy_library" / "registry.json"
            registry_path.parent.mkdir(parents=True)
            registry_path.write_text(
                json.dumps(
                    {
                        "production": {
                            "current": "prod_a",
                            "strategies": [{"strategy_id": "prod_a", "status": "production"}],
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            signal_dir = data_dir / "production_signals"
            signal_dir.mkdir(parents=True)
            (signal_dir / "prod_a_latest.csv").write_text(
                "strategy_id,signal_date,buy_date,action,stock_code,target_pct,status\n",
                encoding="utf-8-sig",
            )
            (signal_dir / "prod_a_latest_status.json").write_text(
                json.dumps(
                    {
                        "strategy_id": "prod_a",
                        "status": "pending_buy_day_hard_gate",
                        "signal_semantics": "no_signal_hold_only",
                        "signal_date": "20260729",
                        "buy_date": "20260730",
                        "latest_signal_date": "20260729",
                        "latest_buy_date": "20260730",
                        "no_signal": True,
                        "hold_only": True,
                        "action_count": 0,
                        "row_count": 0,
                        "buy_count": 0,
                        "sell_count": 0,
                        "l7_execution_allowed": False,
                        "execution_allowed": False,
                        "approved_for_execution": False,
                        "auto_execution_allowed": False,
                        "live_execution_allowed": False,
                        "formal_batch_generated": True,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            signal_rows_duckdb = (
                data_dir
                / "production_assets"
                / "duckdb"
                / "production"
                / "l7"
                / "prod_l7_signal_rows_current.duckdb"
            )
            signal_rows_duckdb.parent.mkdir(parents=True)
            with duckdb.connect(str(signal_rows_duckdb)) as conn:
                conn.execute(
                    """
                    CREATE TABLE prod_l7_signal_rows_current (
                        strategy_id VARCHAR,
                        file_name VARCHAR,
                        source_path VARCHAR,
                        row_index BIGINT,
                        signal_date VARCHAR,
                        buy_date VARCHAR,
                        stock_code VARCHAR,
                        symbol INTEGER,
                        name VARCHAR,
                        rank INTEGER,
                        pred_prob INTEGER,
                        target_pct VARCHAR,
                        row_payload_json VARCHAR
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO prod_l7_signal_rows_current
                    VALUES ('old', 'old.csv', 'old.csv', 1, '20260728', '20260729',
                            '000001.SZ', NULL, 'old', 1, NULL, '0.1', '{}')
                    """
                )
                expected_schema = conn.execute(
                    'PRAGMA table_info("prod_l7_signal_rows_current")'
                ).fetchall()

            result = sync_production_signal_artifacts_to_duckdb(
                data_dir=data_dir,
                signal_dir=signal_dir,
                project_dir=root,
                registry_path=registry_path,
                status_payload_overrides={
                    "ready_for_human_confirmation_execution": False,
                    "pending_user_approval": False,
                    "auto_executable": False,
                    "live_trading_ready": False,
                },
            )

            self.assertEqual(result["signal_file_count"], 2)
            self.assertEqual(result["signal_row_count"], 0)
            self.assertEqual(result["status_row_count"], 1)
            with duckdb.connect(str(result["signal_rows_duckdb_path"]), read_only=True) as conn:
                self.assertEqual(
                    conn.execute('SELECT COUNT(*) FROM "prod_l7_signal_rows_current"').fetchone()[0],
                    0,
                )
                self.assertEqual(
                    conn.execute('PRAGMA table_info("prod_l7_signal_rows_current")').fetchall(),
                    expected_schema,
                )
            with duckdb.connect(str(result["signal_status_duckdb_path"]), read_only=True) as conn:
                status, payload = conn.execute(
                    'SELECT status, payload_json FROM "prod_l7_signal_status_current"'
                ).fetchone()
            self.assertEqual(status, "pending_buy_day_hard_gate")
            payload = json.loads(payload)
            self.assertTrue(payload["no_signal"])
            self.assertFalse(payload["ready_for_human_confirmation_execution"])
            self.assertFalse(payload["pending_user_approval"])
            self.assertFalse(payload["auto_executable"])
            self.assertFalse(payload["live_trading_ready"])

    def test_l7_sync_fails_closed_when_production_current_missing(self):
        import duckdb

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "data_file"
            registry_path = root / "strategy_library" / "registry.json"
            registry_path.parent.mkdir(parents=True)
            registry_path.write_text(
                json.dumps(
                    {
                        "production": {
                            "current": "",
                            "state": "no_available_production_strategy",
                            "strategies": [],
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            signal_dir = data_dir / "production_signals"
            signal_dir.mkdir(parents=True)
            (signal_dir / "withdrawn_latest.csv").write_text(
                "signal_date,buy_date,stock_code\n20260720,20260721,000001.SZ\n",
                encoding="utf-8",
            )
            rows_path = (
                data_dir
                / "production_assets"
                / "duckdb"
                / "production"
                / "l7"
                / "prod_l7_signal_rows_current.duckdb"
            )
            rows_path.parent.mkdir(parents=True)
            with duckdb.connect(str(rows_path)) as connection:
                connection.execute("CREATE TABLE prod_l7_signal_rows_current(marker VARCHAR)")
                connection.execute("INSERT INTO prod_l7_signal_rows_current VALUES ('sentinel')")

            with self.assertRaisesRegex(ValueError, "production.current missing"):
                sync_production_signal_artifacts_to_duckdb(
                    data_dir=data_dir,
                    signal_dir=signal_dir,
                    project_dir=root,
                    registry_path=registry_path,
                    duckdb_path=rows_path,
                )

            with duckdb.connect(str(rows_path), read_only=True) as connection:
                self.assertEqual(
                    connection.execute("SELECT marker FROM prod_l7_signal_rows_current").fetchall(),
                    [("sentinel",)],
                )

    def test_l5_sync_prefers_active_registry_duckdb_path(self):
        import duckdb

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "data_file"
            registry_dir = data_dir / "asset_registry"
            registry_dir.mkdir(parents=True)
            active_duckdb = (
                data_dir
                / "production_assets"
                / "duckdb"
                / "production"
                / "l5"
                / "prod_l5_strategy_registry_current.duckdb"
            )
            active_duckdb.parent.mkdir(parents=True)
            (registry_dir / "production_assets.json").write_text(
                json.dumps(
                    {
                        "assets": [
                            {
                                "asset_id": "prod_l5_strategy_registry_current",
                                "layer": "l5_strategy_signal",
                                "asset_type": "duckdb_table",
                                "status": "production_active",
                                "allowed_for_main_workflow": True,
                                "asset_path": f"{active_duckdb}::prod_l5_strategy_registry_current",
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            strategy_dir = root / "strategy_library" / "production" / "prod_a"
            strategy_dir.mkdir(parents=True)
            registry_path = root / "strategy_library" / "registry.json"
            registry_path.write_text(
                json.dumps(
                    {
                        "production": {
                            "current": "prod_a",
                            "strategies": [
                                {
                                    "strategy_id": "prod_a",
                                    "name": "Prod A",
                                    "path": "strategy_library/production/prod_a",
                                    "status": "production",
                                }
                            ],
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (strategy_dir / "strategy_manifest.json").write_text(
                json.dumps(
                    {
                        "strategy_id": "prod_a",
                        "name": "Prod A",
                        "status": "production",
                        "current_signal": {},
                        "input_contract": {},
                        "validation": {},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            result = sync_strategy_registry_to_duckdb(
                project_dir=root,
                data_dir=data_dir,
                registry_path=registry_path,
            )

            self.assertEqual(Path(result["duckdb_path"]), active_duckdb)
            with duckdb.connect(str(active_duckdb), read_only=True) as conn:
                value = conn.execute('SELECT strategy_id FROM "prod_l5_strategy_registry_current"').fetchone()[0]
            self.assertEqual(value, "prod_a")

    def test_l7_sync_prefers_active_registry_duckdb_path(self):
        import duckdb

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "data_file"
            registry_dir = data_dir / "asset_registry"
            registry_dir.mkdir(parents=True)
            active_duckdb = (
                data_dir
                / "production_assets"
                / "duckdb"
                / "production"
                / "l7"
                / "prod_l7_signal_rows_current.duckdb"
            )
            active_duckdb.parent.mkdir(parents=True)
            (registry_dir / "production_assets.json").write_text(
                json.dumps(
                    {
                        "assets": [
                            {
                                "asset_id": "prod_l7_signal_rows_current",
                                "layer": "l7_trading_delivery",
                                "asset_type": "duckdb_table",
                                "status": "production_active",
                                "allowed_for_main_workflow": True,
                                "asset_path": f"{active_duckdb}::prod_l7_signal_rows_current",
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            registry_path = root / "strategy_library" / "registry.json"
            registry_path.parent.mkdir(parents=True)
            registry_path.write_text(
                json.dumps(
                    {
                        "production": {
                            "current": "prod_a",
                            "strategies": [{"strategy_id": "prod_a", "status": "production"}],
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            signal_dir = data_dir / "production_signals"
            signal_dir.mkdir(parents=True)
            with (signal_dir / "prod_a_latest.csv").open("w", encoding="utf-8-sig", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=["signal_date", "buy_date", "stock_code", "symbol", "pred_prob", "target_pct"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "signal_date": "20260627",
                        "buy_date": "20260630",
                        "stock_code": "000001.SZ",
                        "symbol": "SZSE.000001",
                        "pred_prob": "0.91",
                        "target_pct": "0.98",
                    }
                )
            (signal_dir / "prod_a_latest_status.json").write_text(
                json.dumps({"status": "ok", "latest_signal_date": "20260627", "latest_buy_date": "20260630"}, ensure_ascii=False),
                encoding="utf-8",
            )

            result = sync_production_signal_artifacts_to_duckdb(
                data_dir=data_dir,
                signal_dir=signal_dir,
                project_dir=root,
                registry_path=registry_path,
            )

            self.assertEqual(Path(result["duckdb_path"]), active_duckdb)
            with duckdb.connect(str(active_duckdb), read_only=True) as conn:
                value = conn.execute('SELECT stock_code FROM "prod_l7_signal_rows_current"').fetchone()[0]
            self.assertEqual(value, "000001.SZ")


if __name__ == "__main__":
    unittest.main()
