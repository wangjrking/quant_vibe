import unittest

import pandas as pd

from score_model_research_objective import score_horizon_candidate


class ScoreModelResearchObjectiveTest(unittest.TestCase):
    def test_rankic_gate_failure_blocks_topn_gain(self):
        metrics = pd.DataFrame(
            [
                {
                    "horizon": "3d",
                    "window": "full",
                    "delta_vs_formal_rank_ic": -0.003,
                    "delta_vs_formal_top5": 0.08,
                },
                {
                    "horizon": "3d",
                    "window": "recent63",
                    "delta_vs_formal_rank_ic": 0.01,
                    "delta_vs_formal_top5": 0.02,
                },
                {
                    "horizon": "3d",
                    "window": "recent20",
                    "delta_vs_formal_rank_ic": 0.02,
                    "delta_vs_formal_top5": 0.03,
                },
            ]
        )
        config = {
            "score_weights": {
                "full.daily_rank_ic_mean_delta": 0.4,
                "full.top5_delta": 0.1,
                "recent63.daily_rank_ic_mean_delta": 0.1,
                "recent63.top5_delta": 0.1,
            },
            "additional_constraints": {
                "full_daily_rank_ic_delta_floor": -0.002,
                "recent63_daily_rank_ic_delta_floor": 0.0,
                "recent63_top5_delta_floor": 0.0,
                "topn_gain_cannot_compensate_rankic_gate_failure": True,
            },
        }

        result = score_horizon_candidate(metrics, horizon="3d", objective=config)

        self.assertEqual(result["gate_status"], "failed")
        self.assertIn("full_daily_rank_ic_delta_floor", result["failed_constraints"])
        self.assertGreater(result["weighted_score"], 0)

    def test_candidate_passes_when_weighted_metrics_and_constraints_pass(self):
        metrics = pd.DataFrame(
            [
                {
                    "horizon": "5d",
                    "window": "full",
                    "delta_vs_formal_rank_ic": 0.01,
                    "delta_vs_formal_top1": 0.03,
                    "delta_vs_formal_top5": 0.02,
                    "delta_vs_formal_top10": 0.004,
                },
                {
                    "horizon": "5d",
                    "window": "recent63",
                    "delta_vs_formal_rank_ic": 0.02,
                    "delta_vs_formal_top5": 0.01,
                    "delta_vs_formal_top10": 0.003,
                },
                {
                    "horizon": "5d",
                    "window": "recent20",
                    "delta_vs_formal_top5": 0.02,
                    "delta_vs_formal_top10": 0.01,
                },
            ]
        )
        config = {
            "score_weights": {
                "full.daily_rank_ic_mean_delta": 0.2,
                "full.top1_delta": 0.1,
                "full.top5_delta": 0.15,
                "full.top10_delta": 0.25,
                "recent63.daily_rank_ic_mean_delta": 0.1,
                "recent63.top5_delta": 0.1,
                "recent20.top5_delta": 0.05,
                "recent20.top10_delta": 0.05,
            },
            "additional_constraints": {
                "full_top10_delta_floor": 0.0,
                "full_daily_rank_ic_delta_floor": 0.0,
                "recent63_top10_delta_floor": 0.0,
            },
        }

        result = score_horizon_candidate(metrics, horizon="5d", objective=config)

        self.assertEqual(result["gate_status"], "passed")
        self.assertEqual(result["failed_constraints"], [])
        self.assertAlmostEqual(result["weighted_score"], 0.0135, places=9)

    def test_accepts_plain_delta_metric_columns(self):
        metrics = pd.DataFrame(
            [
                {
                    "horizon": "3d",
                    "window": "full",
                    "rank_ic_positive_ratio_delta": 0.12,
                },
            ]
        )
        config = {
            "score_weights": {
                "full.rank_ic_positive_ratio_delta": 1.0,
            },
            "additional_constraints": {},
        }

        result = score_horizon_candidate(metrics, horizon="3d", objective=config)

        self.assertEqual(result["gate_status"], "passed")
        self.assertEqual(result["missing_metrics"], [])
        self.assertAlmostEqual(result["weighted_score"], 0.12, places=9)


if __name__ == "__main__":
    unittest.main()
