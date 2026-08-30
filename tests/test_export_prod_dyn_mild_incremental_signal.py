from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


MAIN_DIR = Path(__file__).resolve().parents[1]
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from adjustment_semantics import (
    default_adjustment_semantics,
    default_market_field_semantics,
    validate_strategy_output_field_names,
)
import export_prod_dyn_mild_incremental_signal as target


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class ExportProdDynMildIncrementalSignalTests(unittest.TestCase):
    def test_strategy_output_field_validator_rejects_naked_market_price_names(self):
        with self.assertRaisesRegex(ValueError, "naked market price fields"):
            validate_strategy_output_field_names(
                ["stock_code", "open", "close", "atr_qfq"],
                context="unit",
            )

    def test_strategy_output_field_validator_rejects_naked_front_adjusted_indicator_names(self):
        with self.assertRaisesRegex(ValueError, "naked front-adjusted indicator"):
            validate_strategy_output_field_names(
                ["stock_code", "open_raw", "close_raw", "atr"],
                context="unit",
            )

    def test_strategy_output_field_validator_rejects_naked_raw_derived_market_names(self):
        with self.assertRaisesRegex(ValueError, "naked raw-derived market fields"):
            validate_strategy_output_field_names(
                ["stock_code", "open_raw", "close_raw", "pct_chg"],
                context="unit",
            )

    def test_load_sources_rejects_mismatched_adjustment_semantics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            duckdb_path = root / "predictions_3d.duckdb"
            audit_record = root / "audit.md"
            migration_manifest = root / "duckdb_manifest.json"
            duckdb_path.write_bytes(b"duckdb")
            audit_record.write_text("approved", encoding="utf-8")
            migration_manifest.write_text("{}", encoding="utf-8")

            good = default_adjustment_semantics()
            bad = dict(good)
            bad["contract_rule"] = "mismatched contract rule"

            manifests: dict[str, Path] = {}
            for label, semantics in {"3d": good, "5d": bad, "10d": good}.items():
                path = root / f"{label}.json"
                _write_json(
                    path,
                    {
                        "schema_version": 1,
                        "asset_role": "l4_formal_prediction_asset",
                        "approval_status": "approved_for_l5",
                        "source_type": "duckdb_table",
                        "db_path": str(duckdb_path),
                        "table": f"pred_{label}",
                        "market_db_path": str(duckdb_path),
                        "audit_record": str(audit_record),
                        "duckdb_migration_manifest": str(migration_manifest),
                        "adjustment_semantics": semantics,
                        "market_field_semantics": default_market_field_semantics(),
                    },
                )
                manifests[label] = path

            strategy_manifest = {
                "input_contract": {
                    "formal_manifest_3d": str(manifests["3d"]),
                    "formal_manifest_5d": str(manifests["5d"]),
                    "formal_manifest_10d": str(manifests["10d"]),
                    "market_db_path": str(duckdb_path),
                    "allow_legacy": False,
                    "market_field_semantics": default_market_field_semantics(),
                    "adjustment_semantics": good,
                }
            }

            with self.assertRaisesRegex(RuntimeError, "5d manifest adjustment_semantics does not match"):
                target._load_sources(strategy_manifest)

    def test_load_sources_rejects_mismatched_market_field_semantics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            duckdb_path = root / "predictions_3d.duckdb"
            audit_record = root / "audit.md"
            migration_manifest = root / "duckdb_manifest.json"
            duckdb_path.write_bytes(b"duckdb")
            audit_record.write_text("approved", encoding="utf-8")
            migration_manifest.write_text("{}", encoding="utf-8")

            good = default_market_field_semantics()
            bad = dict(good)
            bad["contract_rule"] = "mismatched market field semantics"

            manifests: dict[str, Path] = {}
            for label, semantics in {"3d": good, "5d": bad, "10d": good}.items():
                path = root / f"{label}.json"
                _write_json(
                    path,
                    {
                        "schema_version": 1,
                        "asset_role": "l4_formal_prediction_asset",
                        "approval_status": "approved_for_l5",
                        "source_type": "duckdb_table",
                        "db_path": str(duckdb_path),
                        "table": f"pred_{label}",
                        "market_db_path": str(duckdb_path),
                        "audit_record": str(audit_record),
                        "duckdb_migration_manifest": str(migration_manifest),
                        "adjustment_semantics": default_adjustment_semantics(),
                        "market_field_semantics": semantics,
                    },
                )
                manifests[label] = path

            strategy_manifest = {
                "input_contract": {
                    "formal_manifest_3d": str(manifests["3d"]),
                    "formal_manifest_5d": str(manifests["5d"]),
                    "formal_manifest_10d": str(manifests["10d"]),
                    "market_db_path": str(duckdb_path),
                    "allow_legacy": False,
                    "market_field_semantics": good,
                    "adjustment_semantics": default_adjustment_semantics(),
                }
            }

            with self.assertRaisesRegex(RuntimeError, "5d manifest market_field_semantics does not match"):
                target._load_sources(strategy_manifest)

    def test_load_sources_allows_distinct_prediction_duckdb_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            market_db_path = root / "market.duckdb"
            market_db_path.write_bytes(b"duckdb")
            audit_record = root / "audit.md"
            migration_manifest = root / "duckdb_manifest.json"
            audit_record.write_text("approved", encoding="utf-8")
            migration_manifest.write_text("{}", encoding="utf-8")
            semantics = default_adjustment_semantics()

            manifests: dict[str, Path] = {}
            prediction_paths = {
                "3d": root / "predictions_3d.duckdb",
                "5d": root / "predictions_5d.duckdb",
                "10d": root / "predictions_10d.duckdb",
            }
            for label, path in prediction_paths.items():
                path.write_bytes(b"duckdb")
                manifest_path = root / f"{label}.json"
                _write_json(
                    manifest_path,
                    {
                        "schema_version": 1,
                        "asset_role": "l4_formal_prediction_asset",
                        "approval_status": "approved_for_l5",
                        "source_type": "duckdb_table",
                        "db_path": str(path),
                        "table": f"pred_{label}",
                        "market_db_path": str(market_db_path),
                        "audit_record": str(audit_record),
                        "duckdb_migration_manifest": str(migration_manifest),
                        "adjustment_semantics": semantics,
                        "market_field_semantics": default_market_field_semantics(),
                    },
                )
                manifests[label] = manifest_path

            strategy_manifest = {
                "input_contract": {
                    "formal_manifest_3d": str(manifests["3d"]),
                    "formal_manifest_5d": str(manifests["5d"]),
                    "formal_manifest_10d": str(manifests["10d"]),
                    "market_db_path": str(market_db_path),
                    "allow_legacy": False,
                    "market_field_semantics": default_market_field_semantics(),
                    "adjustment_semantics": semantics,
                }
            }

            sources = target._load_sources(strategy_manifest)
            self.assertEqual(Path(sources["3d"]["db_path"]), prediction_paths["3d"])
            self.assertEqual(Path(sources["5d"]["db_path"]), prediction_paths["5d"])
            self.assertEqual(Path(sources["10d"]["db_path"]), prediction_paths["10d"])
            self.assertEqual(Path(sources["3d"]["market_db_path"]), market_db_path)

    def test_resolve_market_duckdb_path_requires_active_l2_route_match(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            active_market = root / "l2_stock_daily_data.duckdb"
            stale_market = root / "quant_production.duckdb"
            active_market.write_bytes(b"duckdb")
            stale_market.write_bytes(b"duckdb")

            strategy_manifest = {
                "input_contract": {
                    "market_db_path": str(stale_market),
                    "market_field_semantics": default_market_field_semantics(),
                    "adjustment_semantics": default_adjustment_semantics(),
                }
            }

            with patch.object(target, "resolve_stock_daily_duckdb_path", return_value=active_market):
                with self.assertRaisesRegex(RuntimeError, "must match current active L2 DuckDB route"):
                    target._resolve_market_duckdb_path(strategy_manifest, {})

    def test_resolve_market_duckdb_path_accepts_active_l2_route(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            active_market = root / "l2_stock_daily_data.duckdb"
            active_market.write_bytes(b"duckdb")

            strategy_manifest = {
                "input_contract": {
                    "market_db_path": str(active_market),
                    "market_field_semantics": default_market_field_semantics(),
                    "adjustment_semantics": default_adjustment_semantics(),
                }
            }

            with patch.object(target, "resolve_stock_daily_duckdb_path", return_value=active_market):
                resolved = target._resolve_market_duckdb_path(strategy_manifest, {})

            self.assertEqual(resolved, active_market.resolve())

    def test_load_sources_rejects_missing_market_field_semantics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            duckdb_path = root / "predictions.duckdb"
            audit_record = root / "audit.md"
            migration_manifest = root / "duckdb_manifest.json"
            duckdb_path.write_bytes(b"duckdb")
            audit_record.write_text("approved", encoding="utf-8")
            migration_manifest.write_text("{}", encoding="utf-8")

            semantics = default_adjustment_semantics()
            manifests: dict[str, Path] = {}
            for label in ("3d", "5d", "10d"):
                path = root / f"{label}.json"
                _write_json(
                    path,
                    {
                        "schema_version": 1,
                        "asset_role": "l4_formal_prediction_asset",
                        "approval_status": "approved_for_l5",
                        "source_type": "duckdb_table",
                        "db_path": str(duckdb_path),
                        "table": f"pred_{label}",
                        "market_db_path": str(duckdb_path),
                        "audit_record": str(audit_record),
                        "duckdb_migration_manifest": str(migration_manifest),
                        "adjustment_semantics": semantics,
                        "market_field_semantics": default_market_field_semantics(),
                    },
                )
                manifests[label] = path

            strategy_manifest = {
                "input_contract": {
                    "formal_manifest_3d": str(manifests["3d"]),
                    "formal_manifest_5d": str(manifests["5d"]),
                    "formal_manifest_10d": str(manifests["10d"]),
                    "market_db_path": str(duckdb_path),
                    "allow_legacy": False,
                    "adjustment_semantics": semantics,
                }
            }

            with self.assertRaisesRegex(ValueError, "missing market_field_semantics"):
                target._load_sources(strategy_manifest)

    def test_format_signal_row_marks_raw_market_prices_and_returns_explicitly(self):
        row = {
            "signal_date": "20260630",
            "stock_code": "000001.SZ",
            "name": "Ping An",
            "entry_score": 0.91,
            "pred_3d": 0.11,
            "pred_5d": 0.22,
            "pred_10d": 0.33,
            "rank_3d": 0.70,
            "rank_5d": 0.65,
            "rank_10d": 0.90,
            "amount": 1.2e8,
            "turnover_rate": 3.2,
            "total_mv": 9.8e9,
            "open_raw": 10.1,
            "close_raw": 10.2,
            "pre_close_raw": 10.0,
            "atr_qfq": 0.5,
            "pct_chg": 2.0,
            "prev_pct_chg": -1.0,
            "two_day_ret": 0.0098,
        }

        formatted = target._format_signal_row(row, 1, buy_date="20260701", latest_market_date="20260630")

        self.assertEqual(formatted["signal_open_raw"], "10.1")
        self.assertEqual(formatted["signal_close_raw"], "10.2")
        self.assertEqual(formatted["signal_pre_close_raw"], "10")
        self.assertEqual(formatted["signal_pct_chg_raw"], "2")
        self.assertEqual(formatted["signal_prev_pct_chg_raw"], "-1")
        self.assertEqual(formatted["signal_two_day_ret_raw"], "0.0098")
        self.assertIn("atr_qfq", formatted)
        self.assertNotIn("open", formatted)
        self.assertNotIn("close", formatted)
        self.assertNotIn("pre_close", formatted)
        self.assertNotIn("pct_chg", formatted)
        self.assertNotIn("prev_pct_chg", formatted)
        self.assertNotIn("two_day_ret", formatted)


if __name__ == "__main__":
    unittest.main()
