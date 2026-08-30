from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


MAIN_DIR = Path(__file__).resolve().parents[1]
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from adjustment_semantics import (
    MARKET_FIELD_SCHEMA_KIND_STRATEGY,
    validate_adjustment_semantics,
    validate_market_field_semantics,
)
from production_asset_registry import load_production_registry, split_asset_path
from stock_daily_data_route import resolve_stock_daily_duckdb_path


REGISTRY_DUCKDB_LAYERS = {
    "l2_stock_daily_base",
    "l3_features",
    "l3_labels",
    "l5_strategy_signal",
    "l6_backtest",
    "l7_trading_delivery",
}


class ActiveDuckdbOnlyCurrentContractsTests(unittest.TestCase):
    def test_active_l1_route_points_to_full_no_bj_table_root_not_partial_directory(self):
        registry = load_production_registry()
        self.assertIsNotNone(registry, "production asset registry must exist")

        failures: list[str] = []
        for asset in registry.get("assets", []):
            if asset.get("allowed_for_main_workflow") is not True:
                continue
            if str(asset.get("layer", "")).strip().lower() != "l1_raw_data":
                continue
            path, _table = split_asset_path(asset.get("asset_path"))
            if path is None:
                failures.append(f"{asset.get('asset_id')}: active L1 path missing")
                continue
            normalized = path.as_posix().rstrip("/")
            if not normalized.endswith("l1_raw_tables"):
                failures.append(
                    f"{asset.get('asset_id')}: active L1 path must point to l1_raw_tables, got {path}"
                )
            if "partial" in normalized.lower():
                failures.append(
                    f"{asset.get('asset_id')}: active L1 path must not point to partial rebuild directory: {path}"
                )

        if failures:
            self.fail("\n".join(failures))

    def test_active_duckdb_registry_assets_do_not_point_to_shared_or_sqlite_paths(self):
        registry = load_production_registry()
        self.assertIsNotNone(registry, "production asset registry must exist")

        failures: list[str] = []
        for asset in registry.get("assets", []):
            if asset.get("allowed_for_main_workflow") is not True:
                continue
            layer = str(asset.get("layer", "")).strip().lower()
            if layer not in REGISTRY_DUCKDB_LAYERS:
                continue
            path, table = split_asset_path(asset.get("asset_path"))
            if path is None:
                failures.append(f"{asset.get('asset_id')}: missing asset_path")
                continue
            if path.suffix.lower() != ".duckdb":
                failures.append(f"{asset.get('asset_id')}: active path must be .duckdb, got {path}")
                continue
            if path.name.lower() == "quant_production.duckdb":
                failures.append(f"{asset.get('asset_id')}: active path must not point to shared quant_production.duckdb")
            if path.suffix.lower() == ".db":
                failures.append(f"{asset.get('asset_id')}: active path must not point to SQLite db")
            if table is None:
                failures.append(f"{asset.get('asset_id')}: active DuckDB asset must keep explicit table binding")
            if layer == "l5_strategy_signal" and "/production/l5/" not in path.as_posix():
                failures.append(f"{asset.get('asset_id')}: active L5 path must live under production/l5 split route")
            if layer == "l6_backtest" and "/production/l6/" not in path.as_posix():
                failures.append(f"{asset.get('asset_id')}: active L6 path must live under production/l6 split route")
            if layer == "l7_trading_delivery" and "/production/l7/" not in path.as_posix():
                failures.append(f"{asset.get('asset_id')}: active L7 path must live under production/l7 split route")

        if failures:
            self.fail("\n".join(failures))

    def test_active_l4_formal_manifests_use_split_duckdb_and_explicit_semantics(self):
        registry = load_production_registry()
        self.assertIsNotNone(registry, "production asset registry must exist")
        active_market_db_path = str(resolve_stock_daily_duckdb_path(require_exists=False).resolve())

        failures: list[str] = []
        for asset in registry.get("assets", []):
            if asset.get("allowed_for_main_workflow") is not True:
                continue
            if str(asset.get("layer", "")).strip().lower() != "l4_model_prediction":
                continue
            if str(asset.get("asset_type", "")).strip().lower() != "formal_prediction_manifest":
                failures.append(
                    f"{asset.get('asset_id')}: active L4 mainline must be formal_prediction_manifest, got {asset.get('asset_type')}"
                )
                continue
            manifest_path, _table = split_asset_path(asset.get("asset_path"))
            if manifest_path is None or manifest_path.suffix.lower() != ".json":
                failures.append(f"{asset.get('asset_id')}: manifest path invalid: {asset.get('asset_path')}")
                continue
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_text = json.dumps(payload, ensure_ascii=False)
            if str(payload.get("source_type") or "").strip().lower() != "duckdb_table":
                failures.append(f"{manifest_path}: active L4 manifest source_type must be duckdb_table")
            db_path = str(payload.get("db_path") or "").strip()
            if not db_path.endswith(".duckdb"):
                failures.append(f"{manifest_path}: db_path must point to .duckdb")
            if db_path.lower().endswith("quant_production.duckdb"):
                failures.append(f"{manifest_path}: db_path must not point to shared quant_production.duckdb")
            market_db_path = str((payload.get("market_db_path") or "")).strip()
            resolved_market = str((manifest_path.parent / market_db_path).resolve()) if market_db_path else ""
            if resolved_market != active_market_db_path:
                failures.append(
                    f"{manifest_path}: market_db_path must equal active split L2 DuckDB route {active_market_db_path}"
                )
            try:
                validate_adjustment_semantics(
                    payload.get("adjustment_semantics"),
                    context=str(manifest_path),
                )
            except ValueError as exc:
                failures.append(str(exc))
            try:
                validate_market_field_semantics(
                    payload.get("market_field_semantics"),
                    context=str(manifest_path),
                    expected_schema_kind=MARKET_FIELD_SCHEMA_KIND_STRATEGY,
                )
            except ValueError as exc:
                failures.append(str(exc))
            for forbidden in ("quant_production.duckdb", "MODEL_PREDICTIONS.db", "STOCK_DAILY_DATA.db", "odb.db"):
                if forbidden in manifest_text:
                    failures.append(f"{manifest_path}: active manifest must not retain legacy/shared path token {forbidden}")
            latest_incremental_update = payload.get("latest_incremental_update")
            if isinstance(latest_incremental_update, dict):
                feature_input = str(latest_incremental_update.get("production_factor_input") or "")
                label_input = str(latest_incremental_update.get("label_input") or "")
                if feature_input and "l3_feature_current.duckdb::" not in feature_input:
                    failures.append(f"{manifest_path}: production_factor_input must point to split L3 feature DuckDB")
                if label_input and "l3_label_current.duckdb::" not in label_input:
                    failures.append(f"{manifest_path}: label_input must point to split L3 label DuckDB")

        if failures:
            self.fail("\n".join(failures))


if __name__ == "__main__":
    unittest.main()
