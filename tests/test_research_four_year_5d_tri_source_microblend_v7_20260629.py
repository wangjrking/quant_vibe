import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_four_year_5d_tri_source_microblend_v7_20260629 import (
    candidate_weight_patterns,
    tri_blend_objective,
    tri_blend_pass_hard,
)


class TestResearchFourYear5DTriSourceMicroblendV7(unittest.TestCase):
    def test_candidate_weight_patterns_respect_total_cap(self) -> None:
        patterns = candidate_weight_patterns([0.0, 0.01, 0.02], max_total=0.03)
        self.assertIn((0.01, 0.01), patterns)
        self.assertIn((0.02, 0.0), patterns)
        self.assertNotIn((0.02, 0.02), patterns)
        self.assertTrue(all(v5 + ac <= 0.03 for v5, ac in patterns))

    def test_tri_blend_pass_hard_accepts_small_positive_edge(self) -> None:
        row = {
            "full_top5_delta_vs_base": 0.00010,
            "recent63_top5_delta_vs_base": 0.00020,
            "recent20_top5_delta_vs_base": 0.00080,
            "full_rank_ic_delta_vs_base": -0.00004,
            "recent63_rank_ic_delta_vs_base": 0.0,
            "recent20_rank_ic_delta_vs_base": 0.0,
            "nonnegative_top5_months_vs_base": 39,
            "month_count_vs_base": 49,
            "min_month_top5_delta_vs_base": -0.0018,
            "recent63_abs_top5": 0.023,
            "recent20_abs_top5": 0.031,
            "active_days": 220,
        }
        self.assertTrue(tri_blend_pass_hard(row))

    def test_tri_blend_pass_hard_rejects_deep_month_drawdown(self) -> None:
        row = {
            "full_top5_delta_vs_base": 0.00010,
            "recent63_top5_delta_vs_base": 0.00020,
            "recent20_top5_delta_vs_base": 0.00080,
            "full_rank_ic_delta_vs_base": -0.00004,
            "recent63_rank_ic_delta_vs_base": 0.0,
            "recent20_rank_ic_delta_vs_base": 0.0,
            "nonnegative_top5_months_vs_base": 34,
            "month_count_vs_base": 49,
            "min_month_top5_delta_vs_base": -0.0060,
            "recent63_abs_top5": 0.023,
            "recent20_abs_top5": 0.031,
            "active_days": 220,
        }
        self.assertFalse(tri_blend_pass_hard(row))

    def test_tri_blend_objective_penalizes_rankic_loss_and_month_tail(self) -> None:
        stable = {
            "full_top5_delta_vs_base": 0.00009,
            "recent63_top5_delta_vs_base": 0.00030,
            "recent20_top5_delta_vs_base": 0.00120,
            "full_top1_delta_vs_base": 0.00005,
            "recent63_top1_delta_vs_base": 0.00010,
            "recent20_top1_delta_vs_base": 0.00030,
            "full_top5_delta_vs_control": 0.00530,
            "full_rank_ic_delta_vs_base": -0.00003,
            "recent63_rank_ic_delta_vs_base": 0.0,
            "recent20_rank_ic_delta_vs_base": 0.0,
            "min_month_top5_delta_vs_base": -0.0015,
            "active_days": 220,
        }
        unstable = {
            "full_top5_delta_vs_base": 0.00016,
            "recent63_top5_delta_vs_base": 0.00010,
            "recent20_top5_delta_vs_base": 0.00350,
            "full_top1_delta_vs_base": 0.00005,
            "recent63_top1_delta_vs_base": 0.00010,
            "recent20_top1_delta_vs_base": 0.00030,
            "full_top5_delta_vs_control": 0.00530,
            "full_rank_ic_delta_vs_base": -0.00018,
            "recent63_rank_ic_delta_vs_base": -0.00010,
            "recent20_rank_ic_delta_vs_base": -0.00020,
            "min_month_top5_delta_vs_base": -0.0070,
            "active_days": 220,
        }
        self.assertGreater(tri_blend_objective(stable), tri_blend_objective(unstable))


if __name__ == "__main__":
    unittest.main()
