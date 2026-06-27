import unittest

from validate_v12_candidate_acceptance import validate_candidate_acceptance


class ValidateV12CandidateAcceptanceTest(unittest.TestCase):
    def test_passed_objective_and_clean_artifacts_are_ready_for_model_audit(self):
        artifact_validation = {
            "ok": True,
            "errors": [],
            "prediction_table_summary": {
                "row_count": 100,
                "null_pred_prob": 0,
                "duplicate_key_groups": 0,
            },
        }
        runbook_validation = {"ok": True, "errors": []}
        objective_score = {
            "results": [
                {
                    "horizon": "10d",
                    "gate_status": "passed",
                    "failed_constraints": [],
                    "missing_metrics": [],
                    "weighted_score": 0.018,
                }
            ]
        }

        result = validate_candidate_acceptance(
            horizon="10d",
            artifact_validation=artifact_validation,
            runbook_validation=runbook_validation,
            objective_score=objective_score,
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "ready_for_model_side_audit")
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["weighted_score"], 0.018)

    def test_failed_objective_blocks_candidate_even_when_artifacts_are_clean(self):
        artifact_validation = {
            "ok": True,
            "errors": [],
            "prediction_table_summary": {
                "row_count": 100,
                "null_pred_prob": 0,
                "duplicate_key_groups": 0,
            },
        }
        runbook_validation = {"ok": True, "errors": []}
        objective_score = {
            "results": [
                {
                    "horizon": "3d",
                    "gate_status": "failed",
                    "failed_constraints": ["full_daily_rank_ic_delta_floor"],
                    "missing_metrics": [],
                    "weighted_score": 0.007,
                }
            ]
        }

        result = validate_candidate_acceptance(
            horizon="3d",
            artifact_validation=artifact_validation,
            runbook_validation=runbook_validation,
            objective_score=objective_score,
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "not_ready")
        self.assertIn("objective gate failed: full_daily_rank_ic_delta_floor", result["errors"])

    def test_missing_prediction_table_summary_blocks_candidate(self):
        artifact_validation = {"ok": True, "errors": []}
        runbook_validation = {"ok": True, "errors": []}
        objective_score = {
            "results": [
                {
                    "horizon": "10d",
                    "gate_status": "passed",
                    "failed_constraints": [],
                    "missing_metrics": [],
                    "weighted_score": 0.018,
                }
            ]
        }

        result = validate_candidate_acceptance(
            horizon="10d",
            artifact_validation=artifact_validation,
            runbook_validation=runbook_validation,
            objective_score=objective_score,
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "not_ready")
        self.assertIn("artifact validation must include prediction_table_summary", result["errors"])

    def test_bad_prediction_table_summary_blocks_candidate(self):
        artifact_validation = {
            "ok": True,
            "errors": [],
            "prediction_table_summary": {
                "row_count": 100,
                "null_pred_prob": 1,
                "duplicate_key_groups": 2,
            },
        }
        runbook_validation = {"ok": True, "errors": []}
        objective_score = {
            "results": [
                {
                    "horizon": "10d",
                    "gate_status": "passed",
                    "failed_constraints": [],
                    "missing_metrics": [],
                    "weighted_score": 0.018,
                }
            ]
        }

        result = validate_candidate_acceptance(
            horizon="10d",
            artifact_validation=artifact_validation,
            runbook_validation=runbook_validation,
            objective_score=objective_score,
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "not_ready")
        self.assertIn("prediction table null pred_prob count must be 0", result["errors"])
        self.assertIn("prediction table duplicate key groups must be 0", result["errors"])


if __name__ == "__main__":
    unittest.main()
