import unittest

from validate_v12_research_execution_gate import validate_execution_gate


class ValidateV12ResearchExecutionGateTest(unittest.TestCase):
    def _packet(self, status="waiting_user_or_supervisor_approval"):
        return {
            "approval_status": status,
            "requested_authorization": {"experiments_count": 6},
            "boundaries": {
                "no_training_executed_by_packet_generation": True,
                "no_prediction_generated_by_packet_generation": True,
                "no_production_manifest_change": True,
                "no_signal": True,
                "no_backtest": True,
            },
        }

    def _packet_validation(self, ok=True):
        return {
            "ok": ok,
            "errors": [] if ok else ["packet mismatch"],
            "experiments_checked": 6,
        }

    def test_waiting_packet_is_not_ready_even_when_validation_passes(self):
        result = validate_execution_gate(
            packet=self._packet(),
            packet_validation=self._packet_validation(),
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "not_ready_waiting_for_user_or_supervisor_approval")
        self.assertIn("authorization packet is not approved", result["errors"])

    def test_research_area_policy_packet_is_ready_without_per_run_approval(self):
        result = validate_execution_gate(
            packet=self._packet(status="research_area_auto_approved_by_user_policy"),
            packet_validation=self._packet_validation(),
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "ready_for_research_only_training_execution")
        self.assertEqual(result["approval_status"], "research_area_auto_approved_by_user_policy")

    def test_failed_packet_validation_blocks_execution(self):
        result = validate_execution_gate(
            packet=self._packet(status="approved_by_user_or_supervisor"),
            packet_validation=self._packet_validation(ok=False),
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "not_ready")
        self.assertIn("authorization packet validation must be ok", result["errors"])

    def test_approved_and_valid_packet_is_ready(self):
        result = validate_execution_gate(
            packet=self._packet(status="approved_by_user_or_supervisor"),
            packet_validation=self._packet_validation(),
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "ready_for_research_only_training_execution")
        self.assertEqual(result["experiments_authorized"], 6)


if __name__ == "__main__":
    unittest.main()
