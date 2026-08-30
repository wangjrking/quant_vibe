import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_four_year_5d_active_clear_microblend_v6_20260629 import (
    microblend_objective,
    microblend_pass_hard,
)


class TestResearchFourYear5DActiveClearMicroblendV6(unittest.TestCase):
    def test_pass_hard_accepts_near_base_candidate_with_better_month_tail(self) -> None:
        row = {
            "full_top5_delta_vs_base": 0.00005,
            "recent63_top5_delta_vs_base": 0.00030,
            "recent20_top5_delta_vs_base": 0.00000,
            "full_rank_ic_delta_vs_base": 0.0,
            "recent63_rank_ic_delta_vs_base": 0.0,
            "recent20_rank_ic_delta_vs_base": 0.0,
            "nonnegative_top5_months_vs_base": 41,
            "month_count_vs_base": 49,
            "min_month_top5_delta_vs_base": -0.0009,
            "recent63_abs_top5": 0.023,
            "recent20_abs_top5": 0.030,
            "active_days": 320,
        }
        self.assertTrue(microblend_pass_hard(row))

    def test_pass_hard_rejects_if_month_tail_too_deep(self) -> None:
        row = {
            "full_top5_delta_vs_base": 0.00005,
            "recent63_top5_delta_vs_base": 0.00030,
            "recent20_top5_delta_vs_base": 0.00000,
            "full_rank_ic_delta_vs_base": 0.0,
            "recent63_rank_ic_delta_vs_base": 0.0,
            "recent20_rank_ic_delta_vs_base": 0.0,
            "nonnegative_top5_months_vs_base": 35,
            "month_count_vs_base": 49,
            "min_month_top5_delta_vs_base": -0.006,
            "recent63_abs_top5": 0.023,
            "recent20_abs_top5": 0.030,
            "active_days": 320,
        }
        self.assertFalse(microblend_pass_hard(row))

    def test_objective_prefers_small_positive_top5_with_shallower_tail(self) -> None:
        stable = {
            "full_top5_delta_vs_base": 0.00008,
            "recent63_top5_delta_vs_base": 0.00040,
            "recent20_top5_delta_vs_base": 0.00000,
            "full_top1_delta_vs_base": 0.00010,
            "recent63_top1_delta_vs_base": 0.00020,
            "recent20_top1_delta_vs_base": 0.00000,
            "full_top5_delta_vs_control": 0.0053,
            "full_rank_ic_delta_vs_base": 0.0,
            "recent63_rank_ic_delta_vs_base": 0.0,
            "recent20_rank_ic_delta_vs_base": 0.0,
            "min_month_top5_delta_vs_base": -0.0008,
            "active_days": 320,
        }
        unstable = {
            "full_top5_delta_vs_base": -0.00003,
            "recent63_top5_delta_vs_base": 0.00065,
            "recent20_top5_delta_vs_base": 0.00000,
            "full_top1_delta_vs_base": 0.00010,
            "recent63_top1_delta_vs_base": 0.00020,
            "recent20_top1_delta_vs_base": 0.00000,
            "full_top5_delta_vs_control": 0.00525,
            "full_rank_ic_delta_vs_base": 0.0,
            "recent63_rank_ic_delta_vs_base": 0.0,
            "recent20_rank_ic_delta_vs_base": 0.0,
            "min_month_top5_delta_vs_base": -0.0032,
            "active_days": 486,
        }
        self.assertGreater(microblend_objective(stable), microblend_objective(unstable))


if __name__ == "__main__":
    unittest.main()
