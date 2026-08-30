import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_four_year_5d_front_rank_state_gate_v4_20260629 import (
    front_rank_state_gate_objective,
    front_rank_state_gate_pass_hard,
)


class ResearchFourYear5DFrontRankStateGateV4Test(unittest.TestCase):
    def test_pass_hard_accepts_front_rank_gain_with_repaired_monthly_tail(self) -> None:
        row = {
            "full_top5_delta_vs_base": 0.0004,
            "recent63_top5_delta_vs_base": 0.0065,
            "recent20_top5_delta_vs_base": 0.0180,
            "full_rank_ic_delta_vs_base": -0.00028,
            "recent63_rank_ic_delta_vs_base": 0.0008,
            "recent20_rank_ic_delta_vs_base": -0.0004,
            "nonnegative_top5_months_vs_base": 45,
            "month_count_vs_base": 49,
            "min_month_top5_delta_vs_base": -0.0085,
            "recent63_abs_top5": 0.026,
            "recent20_abs_top5": 0.041,
        }

        self.assertTrue(front_rank_state_gate_pass_hard(row))

    def test_pass_hard_rejects_when_monthly_tail_still_too_deep(self) -> None:
        row = {
            "full_top5_delta_vs_base": 0.0004,
            "recent63_top5_delta_vs_base": 0.0065,
            "recent20_top5_delta_vs_base": 0.0180,
            "full_rank_ic_delta_vs_base": -0.00028,
            "recent63_rank_ic_delta_vs_base": 0.0008,
            "recent20_rank_ic_delta_vs_base": -0.0004,
            "nonnegative_top5_months_vs_base": 45,
            "month_count_vs_base": 49,
            "min_month_top5_delta_vs_base": -0.0185,
            "recent63_abs_top5": 0.026,
            "recent20_abs_top5": 0.041,
        }

        self.assertFalse(front_rank_state_gate_pass_hard(row))

    def test_objective_prefers_repaired_monthly_tail_over_raw_front_spike(self) -> None:
        raw_spike = {
            "full_top5_delta_vs_base": 0.0002,
            "recent63_top5_delta_vs_base": 0.0080,
            "recent20_top5_delta_vs_base": 0.0280,
            "full_top1_delta_vs_base": 0.0012,
            "recent63_top1_delta_vs_base": 0.0300,
            "recent20_top1_delta_vs_base": 0.0850,
            "full_rank_ic_delta_vs_base": -0.0005,
            "recent63_rank_ic_delta_vs_base": 0.0020,
            "recent20_rank_ic_delta_vs_base": -0.0006,
            "full_top5_delta_vs_control": 0.0050,
            "min_month_top5_delta_vs_base": -0.0270,
        }
        repaired = {
            "full_top5_delta_vs_base": 0.0005,
            "recent63_top5_delta_vs_base": 0.0055,
            "recent20_top5_delta_vs_base": 0.0150,
            "full_top1_delta_vs_base": 0.0007,
            "recent63_top1_delta_vs_base": 0.0180,
            "recent20_top1_delta_vs_base": 0.0400,
            "full_rank_ic_delta_vs_base": -0.0002,
            "recent63_rank_ic_delta_vs_base": 0.0010,
            "recent20_rank_ic_delta_vs_base": -0.0002,
            "full_top5_delta_vs_control": 0.0040,
            "min_month_top5_delta_vs_base": -0.0060,
        }

        self.assertGreater(front_rank_state_gate_objective(repaired), front_rank_state_gate_objective(raw_spike))


if __name__ == "__main__":
    unittest.main()
