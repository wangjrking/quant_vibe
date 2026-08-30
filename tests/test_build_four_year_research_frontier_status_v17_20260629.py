from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "quant" / "main"))

import build_four_year_research_frontier_status_v17_20260629 as target


class ResearchFrontierStatusV17Tests(unittest.TestCase):
    def test_assess_10d_source_front_gate_rejects_no_hard_pass(self):
        status = target.assess_10d_source_front_gate_v5(
            {
                "decision": "continue_research_no_source_front_gate_v5_hard_pass",
                "target_table": None,
                "pass_hard_count": 0,
                "best": {
                    "condition": "x",
                    "full_rank_ic_delta_vs_base": -0.001,
                    "recent20_top5_delta_vs_base": 0.0,
                    "recent20_rank_ic_delta_vs_base": 0.0,
                    "pass_hard": False,
                },
            }
        )
        self.assertEqual(status["decision"], "reject_axis_keep_current_bestset")
        self.assertEqual(status["pass_hard_count"], 0)
        self.assertIsNone(status["candidate_table"])

    def test_choose_next_focus_keeps_10d_after_rejected_axis(self):
        focus = target.choose_next_focus({"decision": "reject_axis_keep_current_bestset"})
        self.assertEqual(focus["priority_label"], "10d")
        self.assertEqual(focus["action"], "start_new_10d_axis_with_different_source_or_structure")


if __name__ == "__main__":
    unittest.main()
