import unittest

from build_model_research_queue import build_research_queue


class BuildModelResearchQueueTest(unittest.TestCase):
    def test_failed_horizons_are_prioritized_before_passed_validation_items(self):
        score_payload = {
            "results": [
                {
                    "horizon": "10d",
                    "gate_status": "passed",
                    "failed_constraints": [],
                    "weighted_score": 0.018,
                    "priority": "P1",
                    "optimization_style": "preserve_v11_constrained_blend_then_validate",
                },
                {
                    "horizon": "5d",
                    "gate_status": "failed",
                    "failed_constraints": ["full_top10_delta_floor"],
                    "weighted_score": 0.008,
                    "priority": "P2",
                    "optimization_style": "top10_constrained_stability_objective",
                },
                {
                    "horizon": "3d",
                    "gate_status": "failed",
                    "failed_constraints": ["full_daily_rank_ic_delta_floor"],
                    "weighted_score": 0.007,
                    "priority": "P1",
                    "optimization_style": "rankic_constrained_topn_objective",
                },
            ]
        }
        objective_config = {
            "horizon_objectives": {
                "3d": {
                    "label": "executable_3d_open_return",
                    "search_recommendation": ["retrain with RankIC gate"],
                },
                "5d": {
                    "label": "executable_5d_open_return",
                    "search_recommendation": ["repair Top10"],
                },
                "10d": {
                    "label": "executable_10d_open_return",
                    "search_recommendation": ["validate before more blends"],
                },
            }
        }

        queue = build_research_queue(score_payload, objective_config)

        self.assertEqual([item["horizon"] for item in queue], ["3d", "5d", "10d"])
        self.assertEqual(queue[0]["next_action_type"], "retrain_or_feature_search")
        self.assertEqual(queue[0]["failed_constraints"], "full_daily_rank_ic_delta_floor")
        self.assertEqual(queue[2]["next_action_type"], "audit_and_strategy_research_validation")
        self.assertEqual(queue[2]["status"], "passed_objective_gate")


if __name__ == "__main__":
    unittest.main()
