from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "quant" / "main"))

import build_four_year_research_frontier_status_v16_20260629 as target


class ResearchFrontierStatusV16Tests(unittest.TestCase):
    def test_assess_5d_progress_marks_bestset_refreshed(self):
        status = target.assess_5d_progress(
            {
                "asset": "research_5d_four_year_front_rank_state_gate_v4_20260629",
                "current_baseline_delta": {
                    "full_rank_ic_delta": 0.0002,
                    "full_top5_delta": 0.0003,
                    "recent63_top5_delta": 0.0,
                    "recent20_top5_delta": 0.0,
                },
            },
            {
                "best": {
                    "min_month_top5_delta_vs_base": 0.0,
                    "active_days": 3,
                }
            },
            {
                "result": {
                    "hard_constraint_passed": True,
                }
            },
        )
        self.assertEqual(status["decision"], "updated_bestset")
        self.assertEqual(status["next_priority"], "new_formula_search_on_new_bestset")

    def test_next_focus_prefers_10d_after_5d_refresh(self):
        self.assertEqual(target.DEFAULT_NEXT_PRIORITY_LABEL, "10d")


if __name__ == "__main__":
    unittest.main()
