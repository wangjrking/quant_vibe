import sys
import unittest
from pathlib import Path

MAIN_DIR = Path(__file__).resolve().parents[1]
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from workflow_contract_adapters import (
    build_l1_contract_from_report,
    build_l2_contract_from_report,
    build_l3_contract_from_report,
    build_l4_contract_from_report,
    build_l5_contract_from_report,
    build_l6_contract_from_report,
    build_l7_contract_from_report,
    build_l8_contract_from_report,
    validate_built_contract,
)


class WorkflowContractAdapterTests(unittest.TestCase):
    def test_build_l1_contract_from_report(self):
        report = {
            "target_trade_date": "20260706",
            "status": "l1_complete_ready_for_audit",
            "allow_l2_continue": False,
            "ready_for_audit_review": True,
            "universe_rule": "no_bj",
            "active_l1_asset": "quant/data_file/production_assets/duckdb/l1_raw_tables/[table].duckdb",
            "source_gate": {"gate_pass": True, "daily_rows": 5194, "stk_factor_rows": 5194},
            "alignment": {
                "daily_data": {
                    "source_minus_parquet_count": 0,
                    "parquet_minus_source_count": 0,
                    "source_minus_duckdb_count": 0,
                    "duckdb_minus_source_count": 0,
                    "duplicate_key_groups": 0,
                }
            },
            "governance_notes": ["daily_data drives calendar", "adj_factor does not drive calendar"],
        }
        payload = build_l1_contract_from_report(
            report,
            workflow_run_id="incremental-trading-signal-20260706",
            evidence_paths=["report.json"],
        )
        self.assertEqual(payload["layer"], "L1")
        self.assertEqual(payload["status"], "ready_for_audit_review")
        self.assertFalse(payload["allow_next_layer_continue"])
        self.assertEqual(payload["contract"]["active_output_assets"][0], report["active_l1_asset"])
        self.assertEqual(validate_built_contract(payload, "L1"), [])

    def test_build_l2_contract_from_report(self):
        report = {
            "target_trade_date": "20260709",
            "l1_report": "D:/work/quant/quant_mcp/quant/data_file/reports/l1_incremental_raw_ingest_20260709_complete.json",
            "l2_asset": "D:/work/quant/quant_mcp/quant/data_file/production_assets/duckdb/l2_stock_daily_data.duckdb",
            "table": "STOCK_DAILY_DATA",
            "duplicate_key_groups": 0,
            "target_qfq_price_nulls": {
                "open_qfq": 0,
                "high_qfq": 0,
                "low_qfq": 0,
                "close_qfq": 0,
                "pre_close_qfq": 0,
            },
            "full_price_qfq_formula_mismatch_count": 0,
            "target_indicator_sample_mismatch": {"atr_qfq": 0, "macd_qfq": 0},
            "scope": {"non_qfq_fields": "target_date_incremental_or_controlled_overwrite"},
            "governance": {
                "no_bj": True,
                "duckdb_only": True,
            },
        }
        payload = build_l2_contract_from_report(
            report,
            workflow_run_id="incremental-trading-signal-20260709",
            evidence_paths=["validation.json"],
        )
        self.assertEqual(payload["layer"], "L2")
        self.assertEqual(payload["status"], "ready_for_audit_review")
        self.assertEqual(
            payload["contract"]["active_output_assets"][0],
            "D:/work/quant/quant_mcp/quant/data_file/production_assets/duckdb/l2_stock_daily_data.duckdb::STOCK_DAILY_DATA",
        )
        self.assertEqual(validate_built_contract(payload, "L2"), [])

    def test_build_l3_contract_from_report(self):
        report = {
            "target_date": "20260709",
            "workspace_dir": "runtime/agent_workspaces/factor-agent/work/l3_full_factor_delivery_20260709",
            "feature_target_part_path": "feature_target_parts/production_factor_part_target_20260709.parquet",
            "legacy_staging_opt_in": {"env_name": "QUANT_ALLOW_LEGACY_L3_PARQUET_PARTS", "value": "1"},
            "feature_rewrite": {"mode": "skipped_full_rewrite_for_target_date_delivery"},
            "feature_audit": {
                "duckdb_path": "l3_feature_current.duckdb",
                "table_name": "prod_l3_production_factor_parts_20260625",
                "duplicate_rows": 0,
                "bj_rows": 0,
                "target_date_rows": 5194,
                "target_date_bj_rows": 0,
                "has_naked_price_columns": [],
                "has_qfq_price_columns": ["open_qfq", "close_qfq"],
                "has_legacy_gtja_columns": [],
                "has_qfq_gtja_columns": ["gtja_alpha001_qfq"],
            },
            "label_audit": {
                "duckdb_path": "l3_label_current.duckdb",
                "table_name": "prod_l3_prediction_label_parts_current",
                "duplicate_rows": 0,
                "bj_rows": 0,
                "max_trade_date": "20260616",
            },
        }
        payload = build_l3_contract_from_report(report, workflow_run_id="wf", evidence_paths=["l3.json"])
        self.assertEqual(payload["layer"], "L3")
        self.assertIn("l3_feature_current.duckdb::prod_l3_production_factor_parts_20260625", payload["contract"]["active_output_assets"])
        self.assertEqual(validate_built_contract(payload, "L3"), [])

    def test_build_l4_contract_from_report(self):
        report = {
            "target_date": "20260709",
            "inputs": {
                "factor_asset": "l3_feature_current.duckdb::prod_l3_production_factor_parts_20260625",
                "label_asset": "l3_label_current.duckdb::prod_l3_prediction_label_parts_current",
                "factor_status": {"label_max_trade_date": "20260616"},
            },
            "outputs": [
                {
                    "duckdb_path": "l4_1d.duckdb",
                    "table": "pred_1d",
                    "stats": {
                        "latest_day_rows": 5194,
                        "duplicate_key_groups": 0,
                        "null_pred_prob": 0,
                        "latest_day_bj_rows": 0,
                    },
                }
            ],
            "allow_l5_continue": False,
            "boundaries": {"no_training": True, "no_tuning": True, "no_signal": True, "no_backtest": True},
        }
        payload = build_l4_contract_from_report(report, workflow_run_id="wf", evidence_paths=["l4.json"])
        self.assertEqual(payload["layer"], "L4")
        self.assertEqual(payload["status"], "ready_for_audit_review")
        self.assertEqual(validate_built_contract(payload, "L4"), [])

    def test_build_l5_contract_from_report(self):
        report = {
            "signal_date": "20260709",
            "input_manifests": {"3d": "manifest_3d.json", "5d": "manifest_5d.json"},
            "signal_row_count": 3,
            "signal_stock_count": 3,
            "duplicate_signal_stock_keys": 0,
            "old_chain_read_check": {
                "legacy_odb_used": False,
                "model_predictions_sqlite_used": False,
                "research_only_used": False,
            },
            "output_paths": {
                "latest_signal_csv": "production_signals/latest.csv",
                "latest_status_json": "production_signals/latest_status.json",
            },
            "residual_risks": [],
        }
        payload = build_l5_contract_from_report(report, workflow_run_id="wf", evidence_paths=["l5.json"])
        self.assertEqual(payload["layer"], "L5")
        self.assertIn("production_signals/latest.csv", payload["contract"]["active_output_assets"])
        self.assertEqual(validate_built_contract(payload, "L5"), [])

    def test_build_l6_contract_from_report(self):
        report = {
            "signal_date": "20260709",
            "signal_row_count": 3,
            "buy_day_hard_gate_complete": False,
            "strategy_manifest_path": "strategy_manifest.json",
            "trading_rules_path": "trading_rules.json",
            "l7_handoff": {"allowed_now": False, "reason": "pending gate"},
            "output_paths": {
                "latest_signal_csv": "production_signals/latest.csv",
                "latest_status_json": "production_signals/latest_status.json",
                "archive_latest_signal_csv": "strategy_library/signals/latest.csv",
                "full_history_signal_csv": "strategy_library/signals/full_history.csv",
            },
            "residual_risks": [],
        }
        payload = build_l6_contract_from_report(report, workflow_run_id="wf", evidence_paths=["l6.json"])
        self.assertEqual(payload["layer"], "L6")
        self.assertEqual(payload["status"], "pending_buy_day_hard_gate")
        self.assertTrue(payload["allow_next_layer_continue"])
        self.assertEqual(validate_built_contract(payload, "L6"), [])

    def test_build_l7_contract_from_report(self):
        report = {
            "signal_date": "20260709",
            "source_assets": {"latest_csv": "latest.csv", "latest_status_json": "latest_status.json"},
            "duplicate_signal_stock_keys": 0,
            "integrity_checks": {
                "row_count_positive": True,
                "duplicate_keys_is_0": True,
                "all_symbol_mapping_valid": True,
            },
            "buy_day_hard_gate": {
                "status": "pending_buy_day_hard_gate",
                "ready_for_human_confirmation_execution": False,
                "buy_day_hard_gate_complete": False,
                "summary_path": "buy_day_hard_gate_summary.json",
            },
        }
        payload = build_l7_contract_from_report(report, workflow_run_id="wf", evidence_paths=["l7.json"])
        self.assertEqual(payload["layer"], "L7")
        self.assertEqual(payload["status"], "pending_buy_day_hard_gate")
        self.assertEqual(validate_built_contract(payload, "L7"), [])

    def test_build_l8_contract_from_report(self):
        report = {
            "workflow_id": "incremental-trading-signal-20260709",
            "status": "in_progress",
            "requires_user_approval": True,
            "agents": [{"name": f"L{i}", "status": "pending"} for i in range(1, 9)],
            "open_risks": [],
        }
        payload = build_l8_contract_from_report(report, workflow_run_id="incremental-trading-signal-20260709", evidence_paths=["monitor.json"])
        self.assertEqual(payload["layer"], "L8")
        self.assertEqual(payload["status"], "in_progress")
        self.assertEqual(validate_built_contract(payload, "L8"), [])


if __name__ == "__main__":
    unittest.main()
