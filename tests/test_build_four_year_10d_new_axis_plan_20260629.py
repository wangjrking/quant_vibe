import unittest

from build_four_year_10d_new_axis_plan_20260629 import build_plan


class BuildFourYear10DNewAxisPlanTest(unittest.TestCase):
    def test_builds_research_only_new_axis_plan_from_frontier(self):
        frontier = {
            "current_bestset": {
                "10d": {
                    "asset": "research_10d_best",
                    "table": "stock_predict_data_research_10d_best",
                    "eval_min_trade_date": "20220606",
                    "eval_max_trade_date": "20260528",
                    "eval_trade_days": 965,
                    "full_rank_ic": 0.10,
                    "full_top5": 0.04,
                    "recent63_abs_top5": 0.05,
                    "recent20_abs_top5": 0.06,
                    "month_top5_negative": 5,
                    "month_min_top5_delta": -0.01,
                }
            },
            "recent_progress": {
                "10d_candidate_pool_scan_v1": {"decision": "no_candidate_pool_hard_pass"}
            },
            "next_focus": {
                "priority_label": "10d",
                "action": "switch_from_score_reuse_to_new_model_or_feature_axis",
            },
        }

        plan = build_plan(frontier, run_id="unit_run")

        self.assertEqual(plan["current_10d_bestset"]["asset"], "research_10d_best")
        self.assertEqual(len(plan["experiments"]), 3)
        self.assertTrue(plan["boundaries"]["research_only"])
        self.assertTrue(plan["boundaries"]["production_release_requires_user_authorization"])
        first = plan["experiments"][0]
        self.assertEqual(first["params"]["train_years"], 4)
        self.assertEqual(first["params"]["step_months"], 3)
        self.assertEqual(first["params"]["early_stopping_rounds"], 300)
        self.assertIn("--train-mode fixed", first["command"])
        self.assertIn("--feature-source production_split", first["command"])
        self.assertIn("--xgb-early-stopping-rounds 300", first["command"])
        self.assertIn("_research", first["output_table"])
        gate = plan["hard_gate_for_entering_production_candidate_discussion"]
        self.assertEqual(gate["full_rank_ic_delta_vs_current_bestset"], "> 0")


if __name__ == "__main__":
    unittest.main()
