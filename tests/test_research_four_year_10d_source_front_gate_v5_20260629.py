import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_four_year_10d_source_front_gate_v5_20260629 import (
    source_front_gate_objective,
    source_front_gate_pass_hard,
)


class TestResearchFourYear10DSourceFrontGateV5(unittest.TestCase):
    def test_source_front_gate_pass_hard_accepts_nonnegative_recent_and_month_tail(self) -> None:
        row = {
            "full_rank_ic_delta_vs_base": 0.00010,
            "full_top5_delta_vs_base": 0.00020,
            "recent63_top5_delta_vs_base": 0.00005,
            "recent20_top5_delta_vs_base": 0.00120,
            "recent20_rank_ic_delta_vs_base": 0.00030,
            "full_top5_delta_vs_control": 0.0080,
            "nonnegative_top5_months_vs_base": 24,
            "month_count_vs_base": 24,
            "min_month_top5_delta_vs_base": 0.0,
            "recent20_abs_top5": 0.070,
            "recent63_abs_top5": 0.030,
            "active_days": 6,
        }
        self.assertTrue(source_front_gate_pass_hard(row))

    def test_source_front_gate_pass_hard_rejects_negative_recent20_top5(self) -> None:
        row = {
            "full_rank_ic_delta_vs_base": 0.00010,
            "full_top5_delta_vs_base": 0.00020,
            "recent63_top5_delta_vs_base": 0.00005,
            "recent20_top5_delta_vs_base": -0.00001,
            "recent20_rank_ic_delta_vs_base": 0.00030,
            "full_top5_delta_vs_control": 0.0080,
            "nonnegative_top5_months_vs_base": 24,
            "month_count_vs_base": 24,
            "min_month_top5_delta_vs_base": 0.0,
            "recent20_abs_top5": 0.070,
            "recent63_abs_top5": 0.030,
            "active_days": 6,
        }
        self.assertFalse(source_front_gate_pass_hard(row))

    def test_source_front_gate_objective_penalizes_negative_month_tail(self) -> None:
        stable = {
            "recent20_top5_delta_vs_base": 0.002,
            "recent63_top5_delta_vs_base": 0.0005,
            "full_top5_delta_vs_base": 0.0002,
            "recent20_rank_ic_delta_vs_base": 0.0004,
            "full_rank_ic_delta_vs_base": 0.0001,
            "full_top5_delta_vs_control": 0.008,
            "min_month_top5_delta_vs_base": 0.0,
            "active_days": 8,
        }
        unstable = {
            "recent20_top5_delta_vs_base": 0.002,
            "recent63_top5_delta_vs_base": 0.0005,
            "full_top5_delta_vs_base": 0.0002,
            "recent20_rank_ic_delta_vs_base": 0.0004,
            "full_rank_ic_delta_vs_base": 0.0001,
            "full_top5_delta_vs_control": 0.008,
            "min_month_top5_delta_vs_base": -0.002,
            "active_days": 8,
        }
        self.assertGreater(source_front_gate_objective(stable), source_front_gate_objective(unstable))


if __name__ == "__main__":
    unittest.main()
