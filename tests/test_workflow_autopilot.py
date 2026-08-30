import sys
import unittest
from pathlib import Path

MAIN_DIR = Path(__file__).resolve().parents[1]
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from workflow_autopilot import decide_next_action  # noqa: E402


class WorkflowAutopilotTests(unittest.TestCase):
    def test_valid_layer_completion_requests_audit_without_new_grant(self):
        decision = decide_next_action(layer="L3", event="layer_completed", contract_valid=True)
        self.assertEqual(decision.action, "request_audit")
        self.assertEqual(decision.next_layer, "L4")
        self.assertEqual(decision.workflow_status, "waiting_for_audit")

    def test_audit_pass_dispatches_next_layer(self):
        decision = decide_next_action(layer="L3", event="audit_completed", audit_passed=True)
        self.assertEqual(decision.action, "dispatch_next_layer")
        self.assertEqual(decision.next_layer, "L4")

    def test_control_thread_timeout_does_not_consume_business_retry(self):
        decision = decide_next_action(layer="L3", event="control_thread_timeout")
        self.assertEqual(decision.action, "redispatch_control_message")
        self.assertFalse(decision.consume_business_retry)
        self.assertEqual(decision.workflow_status, "in_progress")

    def test_infrastructure_failure_gets_one_bounded_retry(self):
        first = decide_next_action(
            layer="L3", event="stage_failed", failure_class="stale_lease", retries_used=0
        )
        second = decide_next_action(
            layer="L3", event="stage_failed", failure_class="stale_lease", retries_used=1
        )
        self.assertEqual(first.action, "remediate_and_retry_same_layer")
        self.assertTrue(first.consume_business_retry)
        self.assertEqual(second.action, "hard_stop")

    def test_data_or_security_failure_hard_stops(self):
        for failure in ("data_quality", "future_leakage", "unknown_writer"):
            with self.subTest(failure=failure):
                decision = decide_next_action(layer="L3", event="stage_failed", failure_class=failure)
                self.assertEqual(decision.action, "hard_stop")

    def test_invalid_contract_hard_stops(self):
        decision = decide_next_action(layer="L2", event="layer_completed", contract_valid=False)
        self.assertEqual(decision.action, "hard_stop")


if __name__ == "__main__":
    unittest.main()
