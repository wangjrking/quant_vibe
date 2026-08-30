import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_four_year_10d_monthsafe_blend_gate_v4_20260629 import (
    monthsafe_objective,
    monthsafe_pass_hard,
)


class TestResearchFourYear10DMonthsafeBlendGateV4(unittest.TestCase):
    def test_monthsafe_pass_hard_accepts_all_nonnegative_months(self) -> None:
        row = {
            "full_rank_ic_delta_vs_base": 0.00002,
            "full_top5_delta_vs_base": 0.00016,
            "recent63_rank_ic_delta_vs_base": 0.00010,
            "recent63_top5_delta_vs_base": 0.00081,
            "recent20_rank_ic_delta_vs_base": 0.00030,
            "recent20_top5_delta_vs_base": 0.00255,
            "full_top5_delta_vs_control": 0.0082,
            "nonnegative_top5_months_vs_base": 48,
            "month_count_vs_base": 48,
            "min_month_top5_delta_vs_base": 0.0,
            "recent63_abs_top5": 0.057,
            "recent20_abs_top5": 0.072,
            "active_days": 10,
        }
        self.assertTrue(monthsafe_pass_hard(row))

    def test_monthsafe_pass_hard_rejects_negative_month_tail(self) -> None:
        row = {
            "full_rank_ic_delta_vs_base": 0.00014,
            "full_top5_delta_vs_base": 0.00056,
            "recent63_rank_ic_delta_vs_base": 0.00009,
            "recent63_top5_delta_vs_base": 0.00081,
            "recent20_rank_ic_delta_vs_base": 0.00030,
            "recent20_top5_delta_vs_base": 0.00255,
            "full_top5_delta_vs_control": 0.0086,
            "nonnegative_top5_months_vs_base": 47,
            "month_count_vs_base": 48,
            "min_month_top5_delta_vs_base": -0.00070,
            "recent63_abs_top5": 0.057,
            "recent20_abs_top5": 0.072,
            "active_days": 43,
        }
        self.assertFalse(monthsafe_pass_hard(row))

    def test_monthsafe_objective_prefers_stable_small_mask_over_broader_tail_loss(self) -> None:
        stable = {
            "recent20_top5_delta_vs_base": 0.00255,
            "recent63_top5_delta_vs_base": 0.00081,
            "full_top5_delta_vs_base": 0.00016,
            "recent20_rank_ic_delta_vs_base": 0.00030,
            "recent63_rank_ic_delta_vs_base": 0.00010,
            "full_rank_ic_delta_vs_base": 0.00002,
            "full_top5_delta_vs_control": 0.00822,
            "min_month_top5_delta_vs_base": 0.0,
            "active_days": 10,
        }
        unstable = {
            "recent20_top5_delta_vs_base": 0.00255,
            "recent63_top5_delta_vs_base": 0.00081,
            "full_top5_delta_vs_base": 0.00056,
            "recent20_rank_ic_delta_vs_base": 0.00030,
            "recent63_rank_ic_delta_vs_base": 0.00010,
            "full_rank_ic_delta_vs_base": 0.00014,
            "full_top5_delta_vs_control": 0.00863,
            "min_month_top5_delta_vs_base": -0.00070,
            "active_days": 43,
        }
        self.assertGreater(monthsafe_objective(stable), monthsafe_objective(unstable))


if __name__ == "__main__":
    unittest.main()
