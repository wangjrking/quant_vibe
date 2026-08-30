from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "quant" / "main"))

import build_four_year_research_frontier_status_v20_20260630 as target


class ResearchFrontierStatusV20Tests(unittest.TestCase):
    def test_assess_newaxis_status_accepts_gate_passed_candidate(self):
        status = target.assess_newaxis_status(
            {
                "asset": "research_10d_newaxis",
                "table": "table_a",
                "gate_result": {"hard_constraint_passed": True},
                "full_delta_vs_current_bestset": {"rank_ic": 0.001, "top5": 0.002},
                "recent63_delta_vs_current_bestset": {"top5": 0.003},
                "recent20_delta_vs_current_bestset": {"top5": 0.004},
            },
            evidence_path="candidate_summary.json",
        )
        self.assertEqual(status["decision"], "newaxis_train_candidate_promoted_into_current_bestset")
        self.assertEqual(status["asset"], "research_10d_newaxis")
        self.assertEqual(status["evidence"], "candidate_summary.json")

    def test_choose_next_focus_moves_into_newaxis_family_after_promotion(self):
        focus = target.choose_next_focus(
            {"decision": "newaxis_train_candidate_promoted_into_current_bestset"}
        )
        self.assertEqual(focus["priority_label"], "10d")
        self.assertEqual(
            focus["action"],
            "continue_newaxis_model_family_and_compare_alternative_param_axes",
        )


if __name__ == "__main__":
    unittest.main()
