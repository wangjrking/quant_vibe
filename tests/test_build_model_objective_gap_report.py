import unittest

from build_model_objective_gap_report import build_gap_report


class BuildModelObjectiveGapReportTest(unittest.TestCase):
    def test_builds_constraint_gaps_from_objective_score_and_config(self):
        objective_score = {
            "results": [
                {
                    "horizon": "3d",
                    "label": "executable_3d_open_return",
                    "gate_status": "failed",
                    "failed_constraints": ["full_daily_rank_ic_delta_floor"],
                    "weighted_score": 0.007,
                    "weighted_terms": [
                        {
                            "window": "full",
                            "metric": "daily_rank_ic_mean_delta",
                            "value": -0.007,
                        },
                        {
                            "window": "recent63",
                            "metric": "top5_delta",
                            "value": 0.03,
                        },
                    ],
                }
            ]
        }
        objective_config = {
            "horizon_objectives": {
                "3d": {
                    "priority": "P1",
                    "additional_constraints": {
                        "full_daily_rank_ic_delta_floor": -0.002,
                        "recent63_top5_delta_floor": 0.0,
                    },
                }
            }
        }

        report = build_gap_report(objective_score=objective_score, objective_config=objective_config)

        self.assertEqual(report["summary"]["failed_horizons"], ["3d"])
        gap = report["horizons"]["3d"]["constraint_gaps"][0]
        self.assertEqual(gap["constraint"], "full_daily_rank_ic_delta_floor")
        self.assertEqual(gap["metric_key"], "full.daily_rank_ic_mean_delta")
        self.assertAlmostEqual(gap["actual"], -0.007)
        self.assertAlmostEqual(gap["floor"], -0.002)
        self.assertAlmostEqual(gap["gap"], -0.005)
        self.assertFalse(gap["passed"])

    def test_unknown_constraint_is_reported_as_missing_metric(self):
        objective_score = {
            "results": [
                {
                    "horizon": "5d",
                    "label": "executable_5d_open_return",
                    "gate_status": "failed",
                    "failed_constraints": ["custom_unknown_floor"],
                    "weighted_score": 0.01,
                    "weighted_terms": [],
                }
            ]
        }
        objective_config = {
            "horizon_objectives": {
                "5d": {
                    "additional_constraints": {
                        "custom_unknown_floor": 0.0,
                    },
                }
            }
        }

        report = build_gap_report(objective_score=objective_score, objective_config=objective_config)

        gap = report["horizons"]["5d"]["constraint_gaps"][0]
        self.assertEqual(gap["constraint"], "custom_unknown_floor")
        self.assertIsNone(gap["actual"])
        self.assertFalse(gap["passed"])
        self.assertIn("metric mapping missing", gap["note"])


if __name__ == "__main__":
    unittest.main()
