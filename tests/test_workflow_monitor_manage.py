import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

MAIN_DIR = Path(__file__).resolve().parents[1]
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from tools.workflow_monitor_manage import (
    advance_record,
    create_record_from_template,
    list_workflow_templates,
    load_record,
    record_path,
)


class WorkflowMonitorManageTests(unittest.TestCase):
    def test_list_workflow_templates_includes_standard_incremental_chain(self):
        templates = list_workflow_templates(Path(MAIN_DIR).parents[1])
        template_ids = {item["template_id"] for item in templates}
        self.assertIn("standard_incremental_trading_signal_l1_l8", template_ids)

    def test_create_record_from_template_populates_agents_and_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("tools.workflow_monitor_manage.now_iso", return_value="2026-07-11T12:00:00+08:00"):
                args = SimpleNamespace(
                    project_root=str(root),
                    template_id="standard_incremental_trading_signal_l1_l8",
                    workflow_id="incremental-trading-signal-20260711",
                    target_trade_date="20260711",
                    workflow_name=None,
                    user_goal="补齐 20260711 的 L1-L8 标准增量链路",
                    overwrite=False,
                )
                code = create_record_from_template(args)

            self.assertEqual(code, 0)
            path = record_path(root, "incremental-trading-signal-20260711")
            record = load_record(path)

            self.assertEqual(record["workflow_template"], "standard_incremental_trading_signal_l1_l8")
            self.assertEqual(record["workflow_name"], "标准增量交易信号工作流")
            self.assertEqual(
                record["matched_workflow"],
                "standard_incremental_trading_signal: L1/L2/L3/L4/L5/L6/L7/L8 target-date incremental, audit-gated handoff",
            )
            self.assertEqual(record["workflow_contract"]["hard_rules"][0], "no-BJ")
            self.assertEqual(record["workflow_contract"]["execution_scope"], "target_trade_date_only")
            self.assertEqual(record["workflow_contract"]["execution_profile"], "routine_target_date_incremental_fast_path_v1")
            self.assertFalse(record["workflow_contract"]["full_history_rebuild"])
            self.assertFalse(record["workflow_contract"]["authorization_policy"]["per_layer_commander_grant"])
            self.assertTrue(record["autopilot"]["enabled"])
            self.assertEqual(record["autopilot"]["state"], "ready_to_dispatch")
            self.assertEqual(record["workflow_contract"]["execution_route"]["target_trade_date"], "20260711")
            self.assertEqual(record["agents"][0]["agent_id"], "data-ingestion-agent")
            self.assertEqual(record["agents"][0]["layer"], "L1")
            self.assertEqual(record["agents"][-1]["agent_id"], "mcp-agent")
            self.assertEqual(record["agents"][-1]["layer"], "L8")
            self.assertEqual(record["agents"][2]["owner_contract"]["template_id"], "standard_incremental_trading_signal_l1_l8")
            self.assertEqual(len(record["workflow_contract"]["gates"]), 7)

    def test_create_record_from_template_respects_custom_workflow_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("tools.workflow_monitor_manage.now_iso", return_value="2026-07-11T12:00:00+08:00"):
                args = SimpleNamespace(
                    project_root=str(root),
                    template_id="standard_incremental_trading_signal_l1_l8",
                    workflow_id="wf-custom-name",
                    target_trade_date="20260711",
                    workflow_name="20260711 标准增量跑批",
                    user_goal="补齐 20260711",
                    overwrite=False,
                )
                code = create_record_from_template(args)

            self.assertEqual(code, 0)
            record = load_record(record_path(root, "wf-custom-name"))
            self.assertEqual(record["workflow_name"], "20260711 标准增量跑批")

    def test_audit_pass_auto_dispatches_next_layer_without_commander_grant(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_args = SimpleNamespace(
                project_root=str(root),
                template_id="standard_incremental_trading_signal_l1_l8",
                workflow_id="wf-auto-advance",
                target_trade_date="20260711",
                workflow_name=None,
                user_goal="补齐 20260711",
                overwrite=False,
            )
            self.assertEqual(create_record_from_template(create_args), 0)
            path = record_path(root, "wf-auto-advance")
            advance_args = SimpleNamespace(
                project_root=str(root),
                file=str(path),
                layer="L2",
                event="audit_completed",
                contract_valid=False,
                audit_passed=True,
                failure_class=None,
            )
            self.assertEqual(advance_record(advance_args), 0)
            record = load_record(path)
            self.assertEqual(record["autopilot"]["current_layer"], "L3")
            self.assertEqual(record["autopilot"]["last_decision"]["action"], "dispatch_next_layer")
            l3 = next(item for item in record["agents"] if item.get("layer") == "L3")
            self.assertEqual(l3["status"], "in_progress")

    def test_control_timeout_does_not_block_monitor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_args = SimpleNamespace(
                project_root=str(root),
                template_id="standard_incremental_trading_signal_l1_l8",
                workflow_id="wf-control-timeout",
                target_trade_date="20260711",
                workflow_name=None,
                user_goal="补齐 20260711",
                overwrite=False,
            )
            self.assertEqual(create_record_from_template(create_args), 0)
            path = record_path(root, "wf-control-timeout")
            advance_args = SimpleNamespace(
                project_root=str(root),
                file=str(path),
                layer="L3",
                event="control_thread_timeout",
                contract_valid=False,
                audit_passed=False,
                failure_class=None,
            )
            self.assertEqual(advance_record(advance_args), 0)
            record = load_record(path)
            self.assertEqual(record["status"], "in_progress")
            self.assertEqual(record["autopilot"]["last_decision"]["action"], "redispatch_control_message")
            self.assertEqual(record["autopilot"]["business_retries_used"], {})


    def test_create_record_from_template_writes_utf8_bom_for_powershell_compat(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("tools.workflow_monitor_manage.now_iso", return_value="2026-07-11T12:00:00+08:00"):
                args = SimpleNamespace(
                    project_root=str(root),
                    template_id="standard_incremental_trading_signal_l1_l8",
                    workflow_id="wf-bom",
                    target_trade_date="20260711",
                    workflow_name=None,
                    user_goal="BOM compatibility",
                    overwrite=False,
                )
                code = create_record_from_template(args)

            self.assertEqual(code, 0)
            path = record_path(root, "wf-bom")
            self.assertTrue(path.read_bytes().startswith(b"\xef\xbb\xbf"))
            record = load_record(path)
            self.assertEqual(record["workflow_id"], "wf-bom")


if __name__ == "__main__":
    unittest.main()
