import unittest

from validate_v12_experiment_acceptance_matrix import validate_acceptance_matrix


class ValidateV12ExperimentAcceptanceMatrixTest(unittest.TestCase):
    def _matrix(self):
        return {
            "experiments": [
                {
                    "horizon": "3d",
                    "variant": "stability_first",
                    "primary_failed_constraint": "full_daily_rank_ic_delta_floor",
                    "primary_metric_key": "full.daily_rank_ic_mean_delta",
                    "primary_floor": -0.002,
                    "acceptance_metrics": ["full.daily_rank_ic_mean_delta", "recent63.top5_delta"],
                }
            ]
        }

    def test_candidate_passes_when_primary_gap_is_repaired(self):
        objective_score = {
            "results": [
                {
                    "horizon": "3d",
                    "weighted_terms": [
                        {"window": "full", "metric": "daily_rank_ic_mean_delta", "value": -0.001},
                        {"window": "recent63", "metric": "top5_delta", "value": 0.02},
                    ],
                }
            ]
        }

        result = validate_acceptance_matrix(matrix=self._matrix(), objective_score=objective_score)

        self.assertTrue(result["ok"])
        self.assertEqual(result["experiments"][0]["primary_passed"], True)
        self.assertAlmostEqual(result["experiments"][0]["primary_gap"], 0.001)

    def test_candidate_fails_when_primary_gap_is_not_repaired(self):
        objective_score = {
            "results": [
                {
                    "horizon": "3d",
                    "weighted_terms": [
                        {"window": "full", "metric": "daily_rank_ic_mean_delta", "value": -0.004},
                    ],
                }
            ]
        }

        result = validate_acceptance_matrix(matrix=self._matrix(), objective_score=objective_score)

        self.assertFalse(result["ok"])
        self.assertEqual(result["experiments"][0]["primary_passed"], False)
        self.assertAlmostEqual(result["experiments"][0]["primary_gap"], -0.002)
        self.assertIn("3d/stability_first primary metric full.daily_rank_ic_mean_delta below floor", result["errors"])


if __name__ == "__main__":
    unittest.main()
