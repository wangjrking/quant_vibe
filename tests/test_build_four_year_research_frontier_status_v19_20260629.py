from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "quant" / "main"))

import build_four_year_research_frontier_status_v19_20260629 as target


class ResearchFrontierStatusV19Tests(unittest.TestCase):
    def test_assess_sparse_triple_rejects_no_hard_pass(self):
        status = target.assess_sparse_triple_v6(
            {
                "scan_rows": 100,
                "pass_hard_count": 0,
                "best": {
                    "condition": "a",
                    "full_rank_ic_delta_vs_base": 0.0001,
                    "full_top1_delta_vs_base": -0.0001,
                    "full_top5_delta_vs_base": 0.0002,
                    "recent63_top5_delta_vs_base": 0.0,
                    "recent20_top5_delta_vs_base": 0.0,
                    "min_month_top5_delta_vs_base": 0.0,
                },
            }
        )
        self.assertEqual(status["decision"], "reject_sparse_triple_gate_v6_keep_current_bestset")
        self.assertEqual(status["pass_hard_count"], 0)

    def test_choose_next_focus_switches_axis_after_rejection(self):
        focus = target.choose_next_focus({"decision": "reject_sparse_triple_gate_v6_keep_current_bestset"})
        self.assertEqual(focus["priority_label"], "10d")
        self.assertEqual(focus["action"], "switch_from_score_reuse_to_new_model_or_feature_axis")


if __name__ == "__main__":
    unittest.main()
