import unittest

from build_v12_experiment_acceptance_matrix import build_acceptance_matrix


class BuildV12ExperimentAcceptanceMatrixTest(unittest.TestCase):
    def test_maps_training_experiments_to_objective_gaps(self):
        plan = {
            "experiments": [
                {
                    "horizon": "3d",
                    "variant": "stability_first",
                    "output_table": "table_3d_research",
                },
                {
                    "horizon": "3d",
                    "variant": "objective_repair",
                    "output_table": "table_3d_research_2",
                },
            ],
            "skipped_validation_only": ["10d"],
        }
        gap_report = {
            "horizons": {
                "3d": {
                    "gate_status": "failed",
                    "failed_constraints": ["full_daily_rank_ic_delta_floor"],
                    "constraint_gaps": [
                        {
                            "constraint": "full_daily_rank_ic_delta_floor",
                            "metric_key": "full.daily_rank_ic_mean_delta",
                            "actual": -0.007,
                            "floor": -0.002,
                            "gap": -0.005,
                            "passed": False,
                        },
                        {
                            "constraint": "recent63_top5_delta_floor",
                            "metric_key": "recent63.top5_delta",
                            "actual": 0.03,
                            "floor": 0.0,
                            "gap": 0.03,
                            "passed": True,
                        },
                    ],
                }
            }
        }

        matrix = build_acceptance_matrix(plan=plan, gap_report=gap_report)

        self.assertEqual(matrix["summary"]["experiments"], 2)
        self.assertEqual(matrix["summary"]["skipped_validation_only"], ["10d"])
        first = matrix["experiments"][0]
        self.assertEqual(first["horizon"], "3d")
        self.assertEqual(first["primary_failed_constraint"], "full_daily_rank_ic_delta_floor")
        self.assertEqual(first["primary_metric_key"], "full.daily_rank_ic_mean_delta")
        self.assertAlmostEqual(first["primary_gap"], -0.005)
        self.assertIn("full.daily_rank_ic_mean_delta", first["acceptance_metrics"])
        self.assertIn("recent63.top5_delta", first["acceptance_metrics"])


if __name__ == "__main__":
    unittest.main()
