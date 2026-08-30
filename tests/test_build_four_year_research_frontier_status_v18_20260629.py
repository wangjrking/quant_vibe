from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "quant" / "main"))

import build_four_year_research_frontier_status_v18_20260629 as target


class ResearchFrontierStatusV18Tests(unittest.TestCase):
    def test_assess_10d_candidate_pool_rejects_no_hard_pass(self):
        status = target.assess_10d_candidate_pool(
            {
                "scanned_count": 14,
                "hard_pass_count": 0,
                "best": {
                    "table": "candidate",
                    "full_rank_ic_delta_vs_base": 0.0001,
                    "full_top5_delta_vs_base": 0.0002,
                    "recent63_top5_delta_vs_base": -0.001,
                    "recent20_top5_delta_vs_base": -0.002,
                    "min_month_top5_delta_vs_base": -0.003,
                },
            }
        )
        self.assertEqual(status["decision"], "reject_existing_four_year_candidate_pool")
        self.assertEqual(status["hard_pass_count"], 0)
        self.assertEqual(status["best_table"], "candidate")

    def test_choose_next_focus_moves_to_new_axis_when_pool_fails(self):
        focus = target.choose_next_focus({"decision": "reject_existing_four_year_candidate_pool"})
        self.assertEqual(focus["priority_label"], "10d")
        self.assertEqual(focus["action"], "start_new_10d_formula_or_model_axis")


if __name__ == "__main__":
    unittest.main()
