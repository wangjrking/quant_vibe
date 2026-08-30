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
from stock_daily_data_route import resolve_stock_daily_duckdb_path


PRODUCTION_STRATEGY_ROOT = MAIN_DIR / "strategy_library" / "production"


class StrategyManifestAdjustmentSemanticsTests(unittest.TestCase):
    def test_all_production_strategy_manifests_have_explicit_adjustment_and_market_semantics(self):
        failures: list[str] = []
        active_market_db_path = str(resolve_stock_daily_duckdb_path(require_exists=False).resolve())
        for manifest_path in sorted(PRODUCTION_STRATEGY_ROOT.glob("*/strategy_manifest.json")):
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            contract = payload.get("input_contract")
            if contract is None:
                continue
            if not isinstance(contract, dict):
                failures.append(f"{manifest_path}: input_contract must be object")
                continue
            try:
                validate_adjustment_semantics(
                    contract.get("adjustment_semantics"),
                    context=str(manifest_path),
                )
            except ValueError as exc:
                failures.append(str(exc))
            try:
                validate_market_field_semantics(
                    contract.get("market_field_semantics"),
                    context=str(manifest_path),
                    expected_schema_kind=MARKET_FIELD_SCHEMA_KIND_STRATEGY,
                )
            except ValueError as exc:
                failures.append(str(exc))
            market_db_path = str(contract.get("market_db_path") or "").strip()
            if market_db_path != active_market_db_path:
                failures.append(
                    f"{manifest_path} market_db_path must equal active split L2 DuckDB route: "
                    f"{active_market_db_path}"
                )

        if failures:
            self.fail("\n".join(failures))


if __name__ == "__main__":
    unittest.main()
