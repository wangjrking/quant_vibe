import sys
import unittest
from pathlib import Path

MAIN_DIR = Path(__file__).resolve().parents[1]
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from workflow_route_guard import (  # noqa: E402
    build_target_date_incremental_route,
    validate_incremental_route,
)


class WorkflowRouteGuardTests(unittest.TestCase):
    def test_standard_route_is_target_date_only(self):
        route = build_target_date_incremental_route("20260805")
        self.assertEqual(validate_incremental_route(route), [])
        self.assertFalse(route["full_history_rebuild"])
        self.assertFalse(route["requires_candidate_grant"])
        self.assertFalse(route["requires_architect_review"])
        self.assertEqual(
            route["authorization_policy"]["policy_id"],
            "owner_approved_once_auto_advance_v1",
        )
        self.assertFalse(route["authorization_policy"]["per_layer_commander_grant"])
        self.assertTrue(route["authorization_policy"]["audit_pass_auto_dispatches_next_layer"])
        self.assertFalse(route["authorization_policy"]["control_thread_timeout_is_blocking"])
        self.assertFalse(route["evidence_policy"]["repeat_package_parser_checks"])
        self.assertEqual(len(route["stages"]), 8)
        l2_stage = next(stage for stage in route["stages"] if stage["layer"] == "L2")
        self.assertIn(
            "integrate_l2_stock_risk_events_daily.py",
            l2_stage["companion_entrypoints"],
        )

    def test_missing_risk_event_companion_is_blocked(self):
        route = build_target_date_incremental_route("20260805")
        l2_stage = next(stage for stage in route["stages"] if stage["layer"] == "L2")
        l2_stage["companion_entrypoints"] = []

        errors = validate_incremental_route(route)

        self.assertIn(
            "L2 missing companion entrypoint: integrate_l2_stock_risk_events_daily.py",
            errors,
        )

    def test_full_history_entrypoint_is_blocked(self):
        route = build_target_date_incremental_route("20260805")
        route["stages"][2]["entrypoint"] = "rebuild_l3_full_duckdb_mainline.py"
        errors = validate_incremental_route(route)
        self.assertTrue(any("forbidden full-history entrypoint" in error for error in errors))

    def test_scope_drift_is_blocked(self):
        route = build_target_date_incremental_route("20260805")
        route["execution_scope"] = "full_history"
        route["stages"][1]["scope"] = "full_history"
        errors = validate_incremental_route(route)
        self.assertIn("execution_scope must be target_trade_date_only", errors)
        self.assertIn("L2 scope must be target_trade_date_only", errors)

    def test_governance_overhead_drift_is_blocked(self):
        route = build_target_date_incremental_route("20260805")
        route["requires_candidate_grant"] = True
        route["evidence_policy"]["repeat_package_parser_checks"] = True
        errors = validate_incremental_route(route)
        self.assertTrue(any("candidate grant" in error for error in errors))
        self.assertTrue(any("repeat_package_parser_checks" in error for error in errors))

    def test_per_layer_grant_and_blocking_control_timeout_are_blocked(self):
        route = build_target_date_incremental_route("20260805")
        route["authorization_policy"]["per_layer_commander_grant"] = True
        route["authorization_policy"]["control_thread_timeout_is_blocking"] = True
        errors = validate_incremental_route(route)
        self.assertTrue(any("per_layer_commander_grant" in error for error in errors))
        self.assertTrue(any("control_thread_timeout_is_blocking" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
