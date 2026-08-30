import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_four_year_5d_bucketed_blend_v5_20260629 import (
    bucketed_blend_objective,
    bucketed_blend_pass_hard,
    monotonic_weight_patterns,
)


class TestResearchFourYear5DBucketedBlendV5(unittest.TestCase):
    def test_monotonic_weight_patterns_high_direction(self) -> None:
        patterns = monotonic_weight_patterns([0.0, 0.02, 0.05], "high")
        self.assertIn((0.0, 0.02, 0.05), patterns)
        self.assertIn((0.0, 0.0, 0.02), patterns)
        self.assertNotIn((0.05, 0.02, 0.0), patterns)
        self.assertTrue(all(a <= b <= c for a, b, c in patterns))

    def test_bucketed_blend_pass_hard_rejects_sparse_candidate(self) -> None:
        row = {
            "full_top5_delta_vs_base": 0.001,
            "recent63_top5_delta_vs_base": 0.002,
            "recent20_top5_delta_vs_base": 0.003,
            "full_rank_ic_delta_vs_base": 0.0,
            "recent63_rank_ic_delta_vs_base": 0.0,
            "recent20_rank_ic_delta_vs_base": 0.0,
            "nonnegative_top5_months_vs_base": 46,
            "month_count_vs_base": 49,
            "min_month_top5_delta_vs_base": -0.004,
            "recent63_abs_top5": 0.02,
            "recent20_abs_top5": 0.03,
            "active_days": 12,
        }
        self.assertFalse(bucketed_blend_pass_hard(row))

    def test_bucketed_blend_objective_prefers_stable_recent_gain(self) -> None:
        stable = {
            "recent20_top5_delta_vs_base": 0.022,
            "recent63_top5_delta_vs_base": 0.010,
            "full_top5_delta_vs_base": 0.003,
            "recent20_top1_delta_vs_base": 0.030,
            "recent63_top1_delta_vs_base": 0.010,
            "full_top1_delta_vs_base": 0.004,
            "full_top5_delta_vs_control": 0.006,
            "full_rank_ic_delta_vs_base": 0.0002,
            "recent63_rank_ic_delta_vs_base": 0.0001,
            "recent20_rank_ic_delta_vs_base": 0.0,
            "min_month_top5_delta_vs_base": -0.004,
            "active_days": 220,
        }
        spiky = {
            "recent20_top5_delta_vs_base": 0.028,
            "recent63_top5_delta_vs_base": 0.006,
            "full_top5_delta_vs_base": 0.001,
            "recent20_top1_delta_vs_base": 0.040,
            "recent63_top1_delta_vs_base": 0.005,
            "full_top1_delta_vs_base": 0.002,
            "full_top5_delta_vs_control": 0.004,
            "full_rank_ic_delta_vs_base": -0.0008,
            "recent63_rank_ic_delta_vs_base": -0.0003,
            "recent20_rank_ic_delta_vs_base": -0.0012,
            "min_month_top5_delta_vs_base": -0.018,
            "active_days": 35,
        }
        self.assertGreater(bucketed_blend_objective(stable), bucketed_blend_objective(spiky))


if __name__ == "__main__":
    unittest.main()
