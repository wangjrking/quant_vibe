import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


MAIN_DIR = Path(__file__).resolve().parents[1]
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))


class StrategyAssetRouteTests(unittest.TestCase):
    def test_backend_defaults_to_duckdb_without_registry(self):
        from strategy_asset_route import resolve_strategy_asset_backend

        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(resolve_strategy_asset_backend(), "duckdb")

    def test_deprecated_strategy_backend_aliases_are_rejected(self):
        from strategy_asset_route import resolve_strategy_asset_backend

        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "deprecated strategy asset backend alias"):
                resolve_strategy_asset_backend("sqlite")

    def test_load_current_context_from_legacy_files(self):
        from strategy_asset_route import load_current_production_strategy_context

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            strategy_dir = root / "strategy_library" / "production" / "prod_a"
            signal_dir = root / "data_file" / "production_signals"
            strategy_dir.mkdir(parents=True)
            signal_dir.mkdir(parents=True)

            (root / "strategy_library" / "registry.json").write_text(
                json.dumps(
                    {
                        "production": {
                            "current": "prod_a",
                            "strategies": [
                                {
                                    "strategy_id": "prod_a",
                                    "name": "测试策略",
                                    "status": "production",
                                    "path": "strategy_library/production/prod_a",
                                }
                            ],
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (strategy_dir / "strategy_manifest.json").write_text(
                json.dumps({"strategy_id": "prod_a", "name": "测试策略", "production_version": "v1"}, ensure_ascii=False),
                encoding="utf-8",
            )
            with (signal_dir / "prod_a_latest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["signal_date", "buy_date", "stock_code", "name", "rank", "target_pct"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "signal_date": "20260627",
                        "buy_date": "20260630",
                        "stock_code": "000001.SZ",
                        "name": "A",
                        "rank": "1",
                        "target_pct": "0.5",
                    }
                )

            old_cwd = Path.cwd()
            try:
                os.chdir(root / "quant" / "main")
            except FileNotFoundError:
                (root / "quant" / "main").mkdir(parents=True)
                os.chdir(root / "quant" / "main")
            try:
                with patch.dict("os.environ", {"QUANT_ALLOW_LEGACY_SQLITE_PARQUET_ROLLBACK": "1"}, clear=False):
                    context = load_current_production_strategy_context(
                        registry_file=root / "strategy_library" / "registry.json",
                        signal_dir=signal_dir,
                        strategy_root=root / "strategy_library" / "production",
                        backend="legacy",
                    )
            finally:
                os.chdir(old_cwd)

        self.assertEqual(context["backend"], "legacy")
        self.assertEqual(context["strategy_entry"]["strategy_id"], "prod_a")
        self.assertEqual(context["signal_rows"][0]["stock_code"], "000001.SZ")

    def test_load_current_context_from_duckdb_tables(self):
        import duckdb
        import pandas as pd

        from strategy_asset_route import load_current_production_strategy_context

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            duckdb_path = root / "quant_production.duckdb"
            registry_payload = {
                "strategy_id": "prod_a",
                "name": "测试策略",
                "status": "production",
                "path": "strategy_library/production/prod_a",
            }
            manifest_payload = {
                "strategy_id": "prod_a",
                "name": "测试策略",
                "production_version": "v2",
            }
            signal_payload = {
                "signal_date": "20260628",
                "buy_date": "20260630",
                "stock_code": "000001.SZ",
                "name": "A",
                "rank": "1",
                "target_pct": "0.4",
            }
            with duckdb.connect(str(duckdb_path)) as conn:
                conn.register(
                    "_registry_df",
                    pd.DataFrame(
                        [
                            {
                                "strategy_id": "prod_a",
                                "is_current_production": True,
                                "registry_payload_json": json.dumps(registry_payload, ensure_ascii=False),
                            }
                        ]
                    ),
                )
                conn.execute('CREATE TABLE "prod_l5_strategy_registry_current" AS SELECT * FROM _registry_df')
                conn.unregister("_registry_df")
                conn.register(
                    "_manifest_df",
                    pd.DataFrame(
                        [
                            {
                                "strategy_id": "prod_a",
                                "manifest_payload_json": json.dumps(manifest_payload, ensure_ascii=False),
                            }
                        ]
                    ),
                )
                conn.execute('CREATE TABLE "prod_l5_strategy_manifest_current" AS SELECT * FROM _manifest_df')
                conn.unregister("_manifest_df")
                conn.register(
                    "_signal_rows_df",
                    pd.DataFrame(
                        [
                            {
                                "strategy_id": "prod_a",
                                "file_name": "prod_a_latest.csv",
                                "row_index": 1,
                                "signal_date": "20260628",
                                "rank": "1",
                                "row_payload_json": json.dumps(signal_payload, ensure_ascii=False),
                            }
                        ]
                    ),
                )
                conn.execute('CREATE TABLE "prod_l7_signal_rows_current" AS SELECT * FROM _signal_rows_df')
                conn.unregister("_signal_rows_df")
                conn.register(
                    "_signal_files_df",
                    pd.DataFrame(
                        [
                            {
                                "strategy_id": "prod_a",
                                "file_name": "prod_a_latest.csv",
                                "source_path": "quant/data_file/production_signals/prod_a_latest.csv",
                                "modified_at": "2026-06-28T10:00:00",
                            }
                        ]
                    ),
                )
                conn.execute('CREATE TABLE "prod_l7_signal_files_current" AS SELECT * FROM _signal_files_df')
                conn.unregister("_signal_files_df")

            context = load_current_production_strategy_context(
                strategy_id="prod_a",
                duckdb_path=duckdb_path,
                backend="duckdb",
            )

        self.assertEqual(context["backend"], "duckdb")
        self.assertEqual(context["strategy_manifest"]["production_version"], "v2")
        self.assertEqual(context["signal_rows"][0]["signal_date"], "20260628")
        self.assertIn("prod_a_latest.csv", context["signal_path"])

    def test_load_current_context_from_split_l5_l6_l7_duckdb_files(self):
        import duckdb
        import pandas as pd

        from strategy_asset_route import (
            load_current_production_strategy_context,
            load_strategy_validation_payload,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "data_file"
            registry_dir = data_dir / "asset_registry"
            registry_dir.mkdir(parents=True)
            l5_path = data_dir / "production_assets" / "duckdb" / "production" / "l5" / "prod_l5_strategy_registry_current.duckdb"
            l6_path = data_dir / "production_assets" / "duckdb" / "production" / "l6" / "prod_l6_strategy_validation_current.duckdb"
            l7_path = data_dir / "production_assets" / "duckdb" / "production" / "l7" / "prod_l7_signal_rows_current.duckdb"
            l5_path.parent.mkdir(parents=True)
            l6_path.parent.mkdir(parents=True)
            l7_path.parent.mkdir(parents=True)

            (registry_dir / "production_assets.json").write_text(
                json.dumps(
                    {
                        "assets": [
                            {
                                "asset_id": "prod_l5_registry_current",
                                "layer": "L5_strategy_signal",
                                "asset_type": "duckdb_table",
                                "status": "production_active",
                                "allowed_for_main_workflow": True,
                                "asset_path": f"{l5_path}::prod_l5_strategy_registry_current",
                            },
                            {
                                "asset_id": "prod_l6_validation_current",
                                "layer": "L6_backtest",
                                "asset_type": "duckdb_table",
                                "status": "production_active",
                                "allowed_for_main_workflow": True,
                                "asset_path": f"{l6_path}::prod_l6_strategy_validation_current",
                            },
                            {
                                "asset_id": "prod_l7_signal_rows_current",
                                "layer": "L7_trading_delivery",
                                "asset_type": "duckdb_table",
                                "status": "production_active",
                                "allowed_for_main_workflow": True,
                                "asset_path": f"{l7_path}::prod_l7_signal_rows_current",
                            },
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            registry_payload = {
                "strategy_id": "prod_a",
                "name": "测试策略",
                "status": "production",
                "path": "strategy_library/production/prod_a",
            }
            manifest_payload = {
                "strategy_id": "prod_a",
                "name": "测试策略",
                "production_version": "v3",
            }
            validation_payload = {
                "strategy_id": "prod_a",
                "status": "production",
                "validation_platform": "juejin",
            }
            signal_payload = {
                "signal_date": "20260629",
                "buy_date": "20260630",
                "stock_code": "000001.SZ",
                "name": "A",
                "rank": "1",
                "target_pct": "0.4",
            }

            with duckdb.connect(str(l5_path)) as conn:
                conn.register(
                    "_registry_df",
                    pd.DataFrame(
                        [
                            {
                                "strategy_id": "prod_a",
                                "is_current_production": True,
                                "registry_payload_json": json.dumps(registry_payload, ensure_ascii=False),
                            }
                        ]
                    ),
                )
                conn.execute('CREATE TABLE "prod_l5_strategy_registry_current" AS SELECT * FROM _registry_df')
                conn.unregister("_registry_df")
                conn.register(
                    "_manifest_df",
                    pd.DataFrame(
                        [
                            {
                                "strategy_id": "prod_a",
                                "manifest_payload_json": json.dumps(manifest_payload, ensure_ascii=False),
                            }
                        ]
                    ),
                )
                conn.execute('CREATE TABLE "prod_l5_strategy_manifest_current" AS SELECT * FROM _manifest_df')
                conn.unregister("_manifest_df")

            with duckdb.connect(str(l6_path)) as conn:
                conn.register(
                    "_validation_df",
                    pd.DataFrame(
                        [
                            {
                                "strategy_id": "prod_a",
                                "validation_payload_json": json.dumps(validation_payload, ensure_ascii=False),
                            }
                        ]
                    ),
                )
                conn.execute('CREATE TABLE "prod_l6_strategy_validation_current" AS SELECT * FROM _validation_df')
                conn.unregister("_validation_df")

            with duckdb.connect(str(l7_path)) as conn:
                conn.register(
                    "_signal_rows_df",
                    pd.DataFrame(
                        [
                            {
                                "strategy_id": "prod_a",
                                "file_name": "prod_a_latest.csv",
                                "row_index": 1,
                                "signal_date": "20260629",
                                "rank": "1",
                                "row_payload_json": json.dumps(signal_payload, ensure_ascii=False),
                            }
                        ]
                    ),
                )
                conn.execute('CREATE TABLE "prod_l7_signal_rows_current" AS SELECT * FROM _signal_rows_df')
                conn.unregister("_signal_rows_df")
                conn.register(
                    "_signal_files_df",
                    pd.DataFrame(
                        [
                            {
                                "strategy_id": "prod_a",
                                "file_name": "prod_a_latest.csv",
                                "source_path": "quant/data_file/production_signals/prod_a_latest.csv",
                                "modified_at": "2026-06-29T10:00:00",
                            }
                        ]
                    ),
                )
                conn.execute('CREATE TABLE "prod_l7_signal_files_current" AS SELECT * FROM _signal_files_df')
                conn.unregister("_signal_files_df")

            context = load_current_production_strategy_context(
                strategy_id="prod_a",
                data_dir=data_dir,
                backend="duckdb",
            )
            validation = load_strategy_validation_payload(
                "prod_a",
                data_dir=data_dir,
                backend="duckdb",
            )

        self.assertEqual(context["backend"], "duckdb")
        self.assertEqual(context["strategy_manifest"]["production_version"], "v3")
        self.assertEqual(context["signal_rows"][0]["signal_date"], "20260629")
        self.assertEqual(validation["validation_platform"], "juejin")

    def test_load_current_context_from_single_table_duckdb_files(self):
        import duckdb
        import pandas as pd

        from strategy_asset_route import (
            load_current_production_strategy_context,
            load_strategy_validation_payload,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "data_file"
            registry_dir = data_dir / "asset_registry"
            registry_dir.mkdir(parents=True)
            l5_registry_path = data_dir / "production_assets" / "duckdb" / "production" / "l5" / "prod_l5_strategy_registry_current.duckdb"
            l5_manifest_path = data_dir / "production_assets" / "duckdb" / "production" / "l5" / "prod_l5_strategy_manifest_current.duckdb"
            l6_validation_path = data_dir / "production_assets" / "duckdb" / "production" / "l6" / "prod_l6_strategy_validation_current.duckdb"
            l7_rows_path = data_dir / "production_assets" / "duckdb" / "production" / "l7" / "prod_l7_signal_rows_current.duckdb"
            l7_files_path = data_dir / "production_assets" / "duckdb" / "production" / "l7" / "prod_l7_signal_files_current.duckdb"
            for path in (
                l5_registry_path,
                l5_manifest_path,
                l6_validation_path,
                l7_rows_path,
                l7_files_path,
            ):
                path.parent.mkdir(parents=True, exist_ok=True)

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
                                "asset_path": f"{l5_registry_path}::prod_l5_strategy_registry_current",
                            },
                            {
                                "asset_id": "prod_l5_manifest_current",
                                "layer": "l5_strategy_signal",
                                "asset_type": "duckdb_strategy_manifest",
                                "status": "production_active",
                                "allowed_for_main_workflow": True,
                                "asset_path": f"{l5_manifest_path}::prod_l5_strategy_manifest_current",
                            },
                            {
                                "asset_id": "prod_l6_validation_current",
                                "layer": "l6_backtest",
                                "asset_type": "duckdb_strategy_validation",
                                "status": "production_active",
                                "allowed_for_main_workflow": True,
                                "asset_path": f"{l6_validation_path}::prod_l6_strategy_validation_current",
                            },
                            {
                                "asset_id": "prod_l7_rows_current",
                                "layer": "l7_trading_delivery",
                                "asset_type": "duckdb_signal_rows",
                                "status": "production_active",
                                "allowed_for_main_workflow": True,
                                "asset_path": f"{l7_rows_path}::prod_l7_signal_rows_current",
                            },
                            {
                                "asset_id": "prod_l7_files_current",
                                "layer": "l7_trading_delivery",
                                "asset_type": "duckdb_signal_files",
                                "status": "production_active",
                                "allowed_for_main_workflow": True,
                                "asset_path": f"{l7_files_path}::prod_l7_signal_files_current",
                            },
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            registry_payload = {
                "strategy_id": "prod_a",
                "name": "测试策略",
                "status": "production",
                "path": "strategy_library/production/prod_a",
            }
            manifest_payload = {
                "strategy_id": "prod_a",
                "name": "测试策略",
                "production_version": "v4",
            }
            validation_payload = {
                "strategy_id": "prod_a",
                "status": "production",
                "validation_platform": "juejin",
            }
            signal_payload = {
                "signal_date": "20260630",
                "buy_date": "20260701",
                "stock_code": "000001.SZ",
                "name": "A",
                "rank": "1",
                "target_pct": "0.4",
            }

            with duckdb.connect(str(l5_registry_path)) as conn:
                conn.register(
                    "_registry_df",
                    pd.DataFrame(
                        [
                            {
                                "strategy_id": "prod_a",
                                "is_current_production": True,
                                "registry_payload_json": json.dumps(registry_payload, ensure_ascii=False),
                            }
                        ]
                    ),
                )
                conn.execute('CREATE TABLE "prod_l5_strategy_registry_current" AS SELECT * FROM _registry_df')
                conn.unregister("_registry_df")

            with duckdb.connect(str(l5_manifest_path)) as conn:
                conn.register(
                    "_manifest_df",
                    pd.DataFrame(
                        [
                            {
                                "strategy_id": "prod_a",
                                "manifest_payload_json": json.dumps(manifest_payload, ensure_ascii=False),
                            }
                        ]
                    ),
                )
                conn.execute('CREATE TABLE "prod_l5_strategy_manifest_current" AS SELECT * FROM _manifest_df')
                conn.unregister("_manifest_df")

            with duckdb.connect(str(l6_validation_path)) as conn:
                conn.register(
                    "_validation_df",
                    pd.DataFrame(
                        [
                            {
                                "strategy_id": "prod_a",
                                "validation_payload_json": json.dumps(validation_payload, ensure_ascii=False),
                            }
                        ]
                    ),
                )
                conn.execute('CREATE TABLE "prod_l6_strategy_validation_current" AS SELECT * FROM _validation_df')
                conn.unregister("_validation_df")

            with duckdb.connect(str(l7_rows_path)) as conn:
                conn.register(
                    "_signal_rows_df",
                    pd.DataFrame(
                        [
                            {
                                "strategy_id": "prod_a",
                                "file_name": "prod_a_latest.csv",
                                "row_index": 1,
                                "signal_date": "20260630",
                                "rank": "1",
                                "row_payload_json": json.dumps(signal_payload, ensure_ascii=False),
                            }
                        ]
                    ),
                )
                conn.execute('CREATE TABLE "prod_l7_signal_rows_current" AS SELECT * FROM _signal_rows_df')
                conn.unregister("_signal_rows_df")

            with duckdb.connect(str(l7_files_path)) as conn:
                conn.register(
                    "_signal_files_df",
                    pd.DataFrame(
                        [
                            {
                                "strategy_id": "prod_a",
                                "file_name": "prod_a_latest.csv",
                                "source_path": "quant/data_file/production_signals/prod_a_latest.csv",
                                "modified_at": "2026-06-30T10:00:00",
                            }
                        ]
                    ),
                )
                conn.execute('CREATE TABLE "prod_l7_signal_files_current" AS SELECT * FROM _signal_files_df')
                conn.unregister("_signal_files_df")

            context = load_current_production_strategy_context(
                strategy_id="prod_a",
                data_dir=data_dir,
                backend="duckdb",
            )
            validation = load_strategy_validation_payload(
                "prod_a",
                data_dir=data_dir,
                backend="duckdb",
            )

        self.assertEqual(context["backend"], "duckdb")
        self.assertEqual(context["strategy_manifest"]["production_version"], "v4")
        self.assertEqual(context["signal_rows"][0]["signal_date"], "20260630")
        self.assertEqual(validation["validation_platform"], "juejin")
