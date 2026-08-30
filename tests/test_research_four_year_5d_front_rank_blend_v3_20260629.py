import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_four_year_5d_front_rank_blend_v3_20260629 import (
    front_rank_objective,
    front_rank_pass_hard,
)


class ResearchFourYear5DFrontRankBlendV3Test(unittest.TestCase):
    def test_pass_hard_allows_small_rankic_tradeoff_when_topn_improves(self) -> None:
        row = {
            "full_top5_delta_vs_base": 0.0012,
            "recent63_top5_delta_vs_base": 0.0085,
            "recent20_top5_delta_vs_base": 0.0110,
            "full_rank_ic_delta_vs_base": -0.00018,
            "recent63_rank_ic_delta_vs_base": -0.00010,
            "recent20_rank_ic_delta_vs_base": -0.00090,
            "nonnegative_top5_months_vs_base": 22,
            "month_count_vs_base": 23,
            "min_month_top5_delta_vs_base": -0.0015,
            "recent63_abs_top5": 0.020,
            "recent20_abs_top5": 0.028,
        }

        self.assertTrue(front_rank_pass_hard(row))

    def test_pass_hard_rejects_if_recent_top5_turns_negative(self) -> None:
        row = {
            "full_top5_delta_vs_base": 0.0012,
            "recent63_top5_delta_vs_base": 0.0085,
            "recent20_top5_delta_vs_base": -0.0001,
            "full_rank_ic_delta_vs_base": -0.00018,
            "recent63_rank_ic_delta_vs_base": -0.00010,
            "recent20_rank_ic_delta_vs_base": -0.00090,
            "nonnegative_top5_months_vs_base": 22,
            "month_count_vs_base": 23,
            "min_month_top5_delta_vs_base": -0.0015,
            "recent63_abs_top5": 0.020,
            "recent20_abs_top5": 0.028,
        }

        self.assertFalse(front_rank_pass_hard(row))

    def test_objective_prefers_stronger_front_rank_candidate(self) -> None:
        safer = {
            "full_top5_delta_vs_base": 0.0004,
            "recent63_top5_delta_vs_base": 0.0020,
            "recent20_top5_delta_vs_base": 0.0030,
            "full_top1_delta_vs_base": 0.0003,
            "recent63_top1_delta_vs_base": 0.0010,
            "recent20_top1_delta_vs_base": 0.0012,
            "full_rank_ic_delta_vs_base": 0.0002,
            "recent63_rank_ic_delta_vs_base": 0.0001,
            "recent20_rank_ic_delta_vs_base": 0.0000,
            "full_top5_delta_vs_control": 0.0010,
            "min_month_top5_delta_vs_base": -0.0003,
        }
        stronger = {
            "full_top5_delta_vs_base": 0.0011,
            "recent63_top5_delta_vs_base": 0.0080,
            "recent20_top5_delta_vs_base": 0.0105,
            "full_top1_delta_vs_base": 0.0007,
            "recent63_top1_delta_vs_base": 0.0030,
            "recent20_top1_delta_vs_base": 0.0040,
            "full_rank_ic_delta_vs_base": -0.00015,
            "recent63_rank_ic_delta_vs_base": -0.00005,
            "recent20_rank_ic_delta_vs_base": -0.0006,
            "full_top5_delta_vs_control": 0.0025,
            "min_month_top5_delta_vs_base": -0.0010,
        }

        self.assertGreater(front_rank_objective(stronger), front_rank_objective(safer))


if __name__ == "__main__":
    unittest.main()
