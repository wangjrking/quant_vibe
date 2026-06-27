import unittest

from validate_v12_authorization_packet import validate_authorization_packet


class ValidateV12AuthorizationPacketTest(unittest.TestCase):
    def _packet(self):
        return {
            "approval_status": "waiting_user_or_supervisor_approval",
            "requested_authorization": {
                "experiments_count": 2,
                "explicitly_not_allowed": [
                    "production manifest change",
                    "formal L4 or L5 promotion",
                    "trading signal generation",
                    "strategy backtest",
                    "legacy feature or prediction asset usage",
                ],
            },
            "experiments": [
                {"horizon": "3d", "output_table": "table_3d_research"},
                {"horizon": "1d", "output_table": "table_1d_research"},
            ],
            "pre_execution_validation": {
                "plan_validation_ok": True,
                "experiments_checked": 2,
                "parse_errors": 0,
            },
            "boundaries": {
                "no_training_executed_by_packet_generation": True,
                "no_prediction_generated_by_packet_generation": True,
                "no_production_manifest_change": True,
                "no_signal": True,
                "no_backtest": True,
            },
        }

    def _plan(self):
        return {
            "experiments": [
                {"horizon": "3d", "output_table": "table_3d_research"},
                {"horizon": "1d", "output_table": "table_1d_research"},
            ],
        }

    def _validation(self):
        return {
            "ok": True,
            "experiments_checked": 2,
            "errors": [],
        }

    def test_valid_authorization_packet_matches_plan_and_validation(self):
        result = validate_authorization_packet(
            packet=self._packet(),
            plan=self._plan(),
            plan_validation=self._validation(),
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["experiments_checked"], 2)

    def test_approved_authorization_packet_matches_plan_and_validation(self):
        packet = self._packet()
        packet["approval_status"] = "approved_by_user_or_supervisor"

        result = validate_authorization_packet(
            packet=packet,
            plan=self._plan(),
            plan_validation=self._validation(),
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["errors"], [])

    def test_research_area_policy_authorization_packet_matches_plan_and_validation(self):
        packet = self._packet()
        packet["approval_status"] = "research_area_auto_approved_by_user_policy"

        result = validate_authorization_packet(
            packet=packet,
            plan=self._plan(),
            plan_validation=self._validation(),
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["errors"], [])

    def test_packet_table_mismatch_is_rejected(self):
        packet = self._packet()
        packet["experiments"][0]["output_table"] = "different_research"

        result = validate_authorization_packet(
            packet=packet,
            plan=self._plan(),
            plan_validation=self._validation(),
        )

        self.assertFalse(result["ok"])
        self.assertIn("authorization packet experiments do not match experiment plan", result["errors"])

    def test_failed_plan_validation_is_rejected(self):
        validation = self._validation()
        validation["ok"] = False
        validation["errors"] = ["parse failed"]

        result = validate_authorization_packet(
            packet=self._packet(),
            plan=self._plan(),
            plan_validation=validation,
        )

        self.assertFalse(result["ok"])
        self.assertIn("experiment plan validation must be ok", result["errors"])

    def test_missing_boundary_is_rejected(self):
        packet = self._packet()
        packet["boundaries"]["no_backtest"] = False

        result = validate_authorization_packet(
            packet=packet,
            plan=self._plan(),
            plan_validation=self._validation(),
        )

        self.assertFalse(result["ok"])
        self.assertIn("boundary no_backtest must be true", result["errors"])


if __name__ == "__main__":
    unittest.main()
