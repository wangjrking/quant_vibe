from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "quant" / "main"))

import scan_four_year_10d_candidate_pool_v1_20260629 as target


class ScanFourYear10DCandidatePoolV1Tests(unittest.TestCase):
    def test_hard_pass_requires_positive_full_and_nonnegative_recent_month_tail(self):
        row = {
            "full_rank_ic_delta_vs_base": 0.0001,
            "full_top5_delta_vs_base": 0.0002,
            "recent63_top5_delta_vs_base": 0.0,
            "recent20_top5_delta_vs_base": 0.0,
            "nonnegative_top5_months_vs_base": 24,
            "month_count_vs_base": 24,
            "min_month_top5_delta_vs_base": 0.0,
        }
        self.assertTrue(target.hard_pass(row))
        row["min_month_top5_delta_vs_base"] = -0.0001
        self.assertFalse(target.hard_pass(row))

    def test_objective_penalizes_negative_month_tail(self):
        stable = {
            "full_top5_delta_vs_base": 0.001,
            "full_rank_ic_delta_vs_base": 0.001,
            "recent63_top5_delta_vs_base": 0.001,
            "recent20_top5_delta_vs_base": 0.001,
            "min_month_top5_delta_vs_base": 0.0,
        }
        unstable = dict(stable)
        unstable["min_month_top5_delta_vs_base"] = -0.002
        self.assertGreater(target.objective(stable), target.objective(unstable))


if __name__ == "__main__":
    unittest.main()
