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

from l6_duckdb_sync import resolve_l6_duckdb_path, resolve_l6_table_duckdb_path, sync_strategy_backtests_to_duckdb


@unittest.skipIf(importlib.util.find_spec("duckdb") is None, "duckdb is not installed")
class L6DuckDBSyncTests(unittest.TestCase):
    def test_l6_validation_prefers_latest_signal_status_auto_over_stale_validation_block(self):
        import duckdb

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            strategy_dir = root / "strategy_library" / "production" / "prod_a"
            backtest_dir = strategy_dir / "backtests"
            signals_dir = strategy_dir / "signals"
            backtest_dir.mkdir(parents=True)
            signals_dir.mkdir(parents=True)
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
                        "strategy_name": "Prod A",
                        "status": "production",
                        "published_at": "2026-06-28",
                        "validation_platform": "juejin",
                        "current_signal": {
                            "signal_date": "20260630",
                            "buy_date": "20260701",
                            "buy_day_hard_gate_complete": False,
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (strategy_dir / "validation.json").write_text(
                json.dumps(
                    {
                        "metrics": {"annual_return": 1.23},
                        "latest_signal_status": {
                            "signal_date": "20260629",
                            "buy_date": "20260630",
                            "buy_day_hard_gate_complete": False,
                            "selected_stock_code": "000001.SZ",
                        },
                        "hard_gate_audit": {"failed_files": 0, "signal_rows": 10},
                        "validation_platform": "juejin",
                        "status": "production",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (signals_dir / "latest_signal_status_auto.json").write_text(
                json.dumps(
                    {
                        "status": "pending_buy_day_hard_gate",
                        "signal_date": "20260630",
                        "buy_date": "20260701",
                        "latest_signal_date": "20260630",
                        "latest_buy_date": "20260701",
                        "buy_day_hard_gate_complete": False,
                        "selected_stock_code": "000636.SZ",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (backtest_dir / "hard_gate_audit.json").write_text(
                json.dumps({"status": "ok"}, ensure_ascii=False),
                encoding="utf-8",
            )

            duckdb_path = root / "data_file" / "production_assets" / "duckdb" / "production" / "l6" / "prod_l6_strategy_validation_current.duckdb"
            result = sync_strategy_backtests_to_duckdb(
                project_dir=root,
                data_dir=root / "data_file",
                registry_path=registry_path,
                duckdb_path=duckdb_path,
            )

            self.assertEqual(result["validation_row_count"], 1)
            with duckdb.connect(str(duckdb_path), read_only=True) as conn:
                row = conn.execute(
                    'SELECT latest_signal_date, latest_buy_date, selected_stock_code FROM "prod_l6_strategy_validation_current"'
                ).fetchone()
            self.assertEqual(row, ("20260630", "20260701", "000636.SZ"))

    def test_l6_resolve_default_paths_use_single_table_duckdb_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data_file"
            expected_validation = (
                data_dir
                / "production_assets"
                / "duckdb"
                / "production"
                / "l6"
                / "prod_l6_strategy_validation_current.duckdb"
            )
            expected_json = (
                data_dir
                / "production_assets"
                / "duckdb"
                / "production"
                / "l6"
                / "prod_l6_backtest_json_current.duckdb"
            )
            self.assertEqual(resolve_l6_duckdb_path(data_dir=data_dir), expected_validation)
            self.assertEqual(
                resolve_l6_table_duckdb_path("prod_l6_backtest_json_current", data_dir=data_dir),
                expected_json,
            )

    def test_sync_strategy_backtests_to_single_table_duckdb_files(self):
        import duckdb

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "data_file"
            registry_dir = data_dir / "asset_registry"
            registry_dir.mkdir(parents=True)
            validation_duckdb = data_dir / "production_assets" / "duckdb" / "production" / "l6" / "prod_l6_strategy_validation_current.duckdb"
            files_duckdb = data_dir / "production_assets" / "duckdb" / "production" / "l6" / "prod_l6_backtest_files_current.duckdb"
            json_duckdb = data_dir / "production_assets" / "duckdb" / "production" / "l6" / "prod_l6_backtest_json_current.duckdb"
            csv_duckdb = data_dir / "production_assets" / "duckdb" / "production" / "l6" / "prod_l6_backtest_csv_rows_current.duckdb"
            validation_duckdb.parent.mkdir(parents=True, exist_ok=True)
            (registry_dir / "production_assets.json").write_text(
                json.dumps(
                    {
                        "assets": [
                            {
                                "asset_id": "prod_l6_strategy_validation_current",
                                "layer": "l6_backtest",
                                "asset_type": "duckdb_strategy_validation",
                                "status": "production_active",
                                "allowed_for_main_workflow": True,
                                "asset_path": f"{validation_duckdb}::prod_l6_strategy_validation_current",
                            },
                            {
                                "asset_id": "prod_l6_backtest_files_current",
                                "layer": "l6_backtest",
                                "asset_type": "duckdb_backtest_files",
                                "status": "production_active",
                                "allowed_for_main_workflow": True,
                                "asset_path": f"{files_duckdb}::prod_l6_backtest_files_current",
                            },
                            {
                                "asset_id": "prod_l6_backtest_json_current",
                                "layer": "l6_backtest",
                                "asset_type": "duckdb_backtest_json",
                                "status": "production_active",
                                "allowed_for_main_workflow": True,
                                "asset_path": f"{json_duckdb}::prod_l6_backtest_json_current",
                            },
                            {
                                "asset_id": "prod_l6_backtest_csv_rows_current",
                                "layer": "l6_backtest",
                                "asset_type": "duckdb_backtest_csv_rows",
                                "status": "production_active",
                                "allowed_for_main_workflow": True,
                                "asset_path": f"{csv_duckdb}::prod_l6_backtest_csv_rows_current",
                            },
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            strategy_dir = root / "strategy_library" / "production" / "prod_a"
            backtest_dir = strategy_dir / "backtests"
            backtest_dir.mkdir(parents=True)
            registry_path = root / "strategy_library" / "registry.json"
            registry_path.write_text(
                json.dumps(
                    {
                        "production": {
                            "current": "prod_a",
                            "strategies": [{"strategy_id": "prod_a", "path": "strategy_library/production/prod_a", "status": "production"}],
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (strategy_dir / "strategy_manifest.json").write_text(
                json.dumps({"strategy_id": "prod_a", "status": "production"}, ensure_ascii=False),
                encoding="utf-8",
            )
            (strategy_dir / "validation.json").write_text(
                json.dumps({"metrics": {}, "latest_signal_status": {}, "hard_gate_audit": {}, "status": "production"}, ensure_ascii=False),
                encoding="utf-8",
            )
            (backtest_dir / "hard_gate_audit.json").write_text(json.dumps({"status": "ok"}, ensure_ascii=False), encoding="utf-8")
            with (backtest_dir / "nearby_summary.csv").open("w", encoding="utf-8-sig", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=["trade_date", "annual", "sharpe", "max_drawdown", "signal_file", "log_file"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "trade_date": "20260627",
                        "annual": "1.23",
                        "sharpe": "2.34",
                        "max_drawdown": "0.45",
                        "signal_file": "signals.csv",
                        "log_file": "backtest.log",
                    }
                )

            result = sync_strategy_backtests_to_duckdb(
                project_dir=root,
                data_dir=data_dir,
                registry_path=registry_path,
            )

            self.assertEqual(Path(result["validation_duckdb_path"]), validation_duckdb)
            self.assertEqual(Path(result["backtest_file_duckdb_path"]), files_duckdb)
            self.assertEqual(Path(result["backtest_json_duckdb_path"]), json_duckdb)
            self.assertEqual(Path(result["backtest_csv_duckdb_path"]), csv_duckdb)
            with duckdb.connect(str(validation_duckdb), read_only=True) as conn:
                validation_value = conn.execute('SELECT strategy_id FROM "prod_l6_strategy_validation_current"').fetchone()[0]
            with duckdb.connect(str(files_duckdb), read_only=True) as conn:
                files_value = conn.execute('SELECT COUNT(*) FROM "prod_l6_backtest_files_current"').fetchone()[0]
            with duckdb.connect(str(json_duckdb), read_only=True) as conn:
                json_value = conn.execute('SELECT file_name FROM "prod_l6_backtest_json_current"').fetchone()[0]
            with duckdb.connect(str(csv_duckdb), read_only=True) as conn:
                csv_value = conn.execute('SELECT strategy_id FROM "prod_l6_backtest_csv_rows_current"').fetchone()[0]
            self.assertEqual(validation_value, "prod_a")
            self.assertEqual(files_value, 2)
            self.assertEqual(json_value, "hard_gate_audit.json")
            self.assertEqual(csv_value, "prod_a")

    def test_sync_strategy_backtests_to_duckdb_materializes_validation_and_backtests(self):
        import duckdb

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            strategy_dir = root / "strategy_library" / "production" / "prod_a"
            backtest_dir = strategy_dir / "backtests"
            backtest_dir.mkdir(parents=True)
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
                        "strategy_name": "Prod A",
                        "status": "production",
                        "published_at": "2026-06-28",
                        "validation_platform": "juejin",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (strategy_dir / "validation.json").write_text(
                json.dumps(
                    {
                        "metrics": {
                            "annual_return": 1.23,
                            "pnl_ratio": 4.56,
                            "sharpe": 2.34,
                            "max_drawdown": 0.45,
                            "avg_invested_pct": 0.67,
                            "max_active_positions": 1,
                        },
                        "latest_signal_status": {
                            "signal_date": "20260627",
                            "buy_date": "20260630",
                            "buy_day_hard_gate_complete": True,
                            "selected_stock_code": "000001.SZ",
                        },
                        "hard_gate_audit": {
                            "failed_files": 0,
                            "signal_rows": 10,
                        },
                        "validation_platform": "juejin",
                        "status": "production",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (backtest_dir / "hard_gate_audit.json").write_text(
                json.dumps({"status": "ok"}, ensure_ascii=False),
                encoding="utf-8",
            )
            with (backtest_dir / "nearby_summary.csv").open("w", encoding="utf-8-sig", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=["trade_date", "annual", "sharpe", "max_drawdown", "signal_file", "log_file"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "trade_date": "20260627",
                        "annual": "1.23",
                        "sharpe": "2.34",
                        "max_drawdown": "0.45",
                        "signal_file": "signals.csv",
                        "log_file": "backtest.log",
                    }
                )

            duckdb_path = root / "data_file" / "production_assets" / "duckdb" / "quant_production.duckdb"
            result = sync_strategy_backtests_to_duckdb(
                project_dir=root,
                data_dir=root / "data_file",
                registry_path=registry_path,
                duckdb_path=duckdb_path,
            )

            self.assertEqual(result["validation_row_count"], 1)
            self.assertEqual(result["backtest_file_count"], 2)
            self.assertEqual(result["backtest_json_count"], 1)
            self.assertEqual(result["backtest_csv_row_count"], 1)
            with duckdb.connect(str(duckdb_path), read_only=True) as conn:
                validation_row = conn.execute(
                    'SELECT strategy_id, annual_return, latest_signal_date FROM "prod_l6_strategy_validation_current"'
                ).fetchone()
                file_row_count = conn.execute(
                    'SELECT COUNT(*) FROM "prod_l6_backtest_files_current"'
                ).fetchone()[0]
                csv_row = conn.execute(
                    'SELECT strategy_id, signal_file FROM "prod_l6_backtest_csv_rows_current"'
                ).fetchone()

            self.assertEqual(validation_row, ("prod_a", 1.23, "20260627"))
            self.assertEqual(file_row_count, 2)
            self.assertEqual(csv_row, ("prod_a", "signals.csv"))

    def test_l6_sync_prefers_active_registry_duckdb_path(self):
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
                / "l6"
                / "prod_l6_strategy_validation_current.duckdb"
            )
            active_duckdb.parent.mkdir(parents=True)
            (registry_dir / "production_assets.json").write_text(
                json.dumps(
                    {
                        "assets": [
                            {
                                "asset_id": "prod_l6_strategy_validation_current",
                                "layer": "l6_backtest",
                                "asset_type": "duckdb_table",
                                "status": "production_active",
                                "allowed_for_main_workflow": True,
                                "asset_path": f"{active_duckdb}::prod_l6_strategy_validation_current",
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            strategy_dir = root / "strategy_library" / "production" / "prod_a"
            backtest_dir = strategy_dir / "backtests"
            backtest_dir.mkdir(parents=True)
            registry_path = root / "strategy_library" / "registry.json"
            registry_path.write_text(
                json.dumps(
                    {
                        "production": {
                            "current": "prod_a",
                            "strategies": [{"strategy_id": "prod_a", "path": "strategy_library/production/prod_a", "status": "production"}],
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (strategy_dir / "strategy_manifest.json").write_text(
                json.dumps({"strategy_id": "prod_a", "status": "production"}, ensure_ascii=False),
                encoding="utf-8",
            )
            (strategy_dir / "validation.json").write_text(
                json.dumps({"metrics": {}, "latest_signal_status": {}, "hard_gate_audit": {}, "status": "production"}, ensure_ascii=False),
                encoding="utf-8",
            )
            (backtest_dir / "hard_gate_audit.json").write_text(json.dumps({"status": "ok"}, ensure_ascii=False), encoding="utf-8")

            result = sync_strategy_backtests_to_duckdb(
                project_dir=root,
                data_dir=data_dir,
                registry_path=registry_path,
            )

            self.assertEqual(Path(result["duckdb_path"]), active_duckdb)
            with duckdb.connect(str(active_duckdb), read_only=True) as conn:
                value = conn.execute('SELECT strategy_id FROM "prod_l6_strategy_validation_current"').fetchone()[0]
            self.assertEqual(value, "prod_a")


if __name__ == "__main__":
    unittest.main()
