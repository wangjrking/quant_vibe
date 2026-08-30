import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import build_four_year_3d_hybrid_refine_candidate_20260629 as target


class BuildFourYear3dHybridRefineCandidate20260629Tests(unittest.TestCase):
    def test_pass_hard_accepts_positive_improvement_vs_current_bestset(self):
        row = {
            "full_rank_ic_delta_vs_base": 0.0002,
            "full_top5_delta_vs_base": 0.0003,
            "recent20_rank_ic_delta_vs_base": 0.0001,
            "recent63_rank_ic_delta_vs_base": 0.0001,
            "recent20_top5_delta_vs_base": 0.0004,
            "recent63_top5_delta_vs_base": 0.0002,
            "full_top5_delta_vs_control": 0.005,
            "recent20_top5_delta_vs_control": 0.02,
            "recent63_top5_delta_vs_control": 0.01,
            "recent20_abs_top5": 0.02,
            "recent63_abs_top5": 0.015,
        }
        self.assertTrue(target.pass_hard(row))

    def test_pass_hard_rejects_recent20_rankic_regression_vs_base(self):
        row = {
            "full_rank_ic_delta_vs_base": 0.0002,
            "full_top5_delta_vs_base": 0.0003,
            "recent20_rank_ic_delta_vs_base": -0.0001,
            "recent63_rank_ic_delta_vs_base": 0.0001,
            "recent20_top5_delta_vs_base": 0.0004,
            "recent63_top5_delta_vs_base": 0.0002,
            "full_top5_delta_vs_control": 0.005,
            "recent20_top5_delta_vs_control": 0.02,
            "recent63_top5_delta_vs_control": 0.01,
            "recent20_abs_top5": 0.02,
            "recent63_abs_top5": 0.015,
        }
        self.assertFalse(target.pass_hard(row))

    def test_tables_compare_current_bestset_with_prior_hybrid(self):
        self.assertIn("clear_replacement_gate", target.BASE_TABLE)
        self.assertIn("active_clear_gate_hybrid", target.CAND_TABLE)


if __name__ == "__main__":
    unittest.main()
