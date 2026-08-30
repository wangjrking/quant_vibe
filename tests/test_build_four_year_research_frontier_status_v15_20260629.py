from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "quant" / "main"))

import build_four_year_research_frontier_status_v15_20260629 as target


class ResearchFrontierStatusV15Tests(unittest.TestCase):
    def test_assess_5d_status_keeps_current_when_direct_deltas_not_positive(self):
        status = target.assess_5d_status(
            {
                "current_baseline_delta": {
                    "full_rank_ic_delta": -0.00001,
                    "full_top5_delta": 0.0001,
                    "recent63_top5_delta": 0.0,
                    "recent20_top5_delta": 0.0,
                }
            },
            {
                "decision": "continue_research_no_bestset_clear_gate_v2_hard_pass",
            },
            {
                "full_rank_ic": 0.08,
                "full_top5": 0.027,
                "recent63_abs_top5": 0.023,
                "recent20_abs_top5": 0.03,
            },
        )
        self.assertEqual(status["decision"], "keep_current_bestset")
        self.assertEqual(status["next_priority"], "new_formula_search")

    def test_assess_1d_status_marks_fixed4y_retrain_not_competitive(self):
        status = target.assess_1d_status(
            {
                "best_candidate": {"objective": -0.05, "top5": -0.006, "rank_ic": -0.04, "candidate": "demo"},
                "baselines": [
                    {"candidate": "current_four_year_bestset", "objective": 0.09, "top5": 0.008, "rank_ic": 0.02}
                ],
            }
        )
        self.assertEqual(status["decision"], "keep_current_bestset")
        self.assertEqual(status["next_priority"], "pause_fixed4y_retrain_line")
        self.assertLess(status["objective_delta_vs_bestset"], 0.0)


if __name__ == "__main__":
    unittest.main()
