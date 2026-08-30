import sys
import unittest
from pathlib import Path

MAIN_DIR = Path(__file__).resolve().parents[1]
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from workflow_contract import (
    build_layer_handoff_contract,
    next_layer_for,
    validate_layer_handoff_contract,
)


class WorkflowContractTests(unittest.TestCase):
    def test_next_layer_for(self):
        self.assertEqual(next_layer_for("L1"), "L2")
        self.assertEqual(next_layer_for("L7"), "L8")
        self.assertIsNone(next_layer_for("L8"))

    def test_build_and_validate_contract(self):
        payload = build_layer_handoff_contract(
            workflow_run_id="incremental-trading-signal-20260711",
            layer="L3",
            target_trade_date="20260711",
            status="audit_passed",
            ready_for_audit_review=True,
            allow_next_layer_continue=True,
            active_input_assets=["l2_stock_daily_data.duckdb::STOCK_DAILY_DATA"],
            active_output_assets=["l3_feature_current.duckdb::prod_l3_production_factor_parts_20260625"],
            gate_checks=[{"name": "no_bj_enforced", "passed": True}],
            handoff_constraints=["不得回退 legacy parquet"],
            evidence_paths=["quant/data_file/reports/l3_full_delivery_20260711.json"],
            residual_risk=[{"level": "P2", "summary": "target-date only residue"}],
            boundaries={"no_training": True},
            layer_payload={"max_trade_date": "20260711"},
        )
        errors = validate_layer_handoff_contract(payload, expected_layer="L3", expected_owner_agent="factor-agent")
        self.assertEqual(errors, [])

    def test_validate_rejects_missing_required_structure(self):
        payload = {
            "schema_version": 999,
            "contract_type": "bad",
            "layer": "L2",
            "owner_agent": "data-integration-agent",
            "workflow_template": "",
            "workflow_run_id": "",
            "target_trade_date": "",
            "status": "unknown",
            "ready_for_audit_review": "yes",
            "allow_next_layer_continue": "no",
            "next_layer": "L9",
            "hard_rules": [],
            "contract": {},
            "layer_payload": [],
        }
        errors = validate_layer_handoff_contract(payload, expected_layer="L2")
        self.assertTrue(errors)
        self.assertIn("schema_version must be 1", errors)
        self.assertIn("contract_type must be workflow_layer_handoff", errors)
        self.assertIn("status is not in allowed terminal statuses", errors)


if __name__ == "__main__":
    unittest.main()
