import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import refresh_four_year_latest_bestset_status_v16_20260630 as target


class RefreshFourYearLatestBestsetStatusV1620260630Tests(unittest.TestCase):
    def test_should_replace_10d_bestset_requires_gate_and_nonnegative_recent_direct_deltas(self):
        self.assertTrue(
            target.should_replace_10d_bestset(
                {
                    "full_rank_ic_delta": 0.0001,
                    "full_top5_delta": 0.0002,
                    "recent63_rank_ic_delta": 0.0,
                    "recent63_top5_delta": 0.0003,
                    "recent20_rank_ic_delta": 0.0,
                    "recent20_top5_delta": 0.0004,
                },
                gate_passed=True,
            )
        )
        self.assertFalse(
            target.should_replace_10d_bestset(
                {
                    "full_rank_ic_delta": 0.0001,
                    "full_top5_delta": 0.0002,
                    "recent63_rank_ic_delta": -0.0001,
                    "recent63_top5_delta": 0.0003,
                    "recent20_rank_ic_delta": 0.0,
                    "recent20_top5_delta": 0.0004,
                },
                gate_passed=True,
            )
        )
        self.assertFalse(
            target.should_replace_10d_bestset(
                {
                    "full_rank_ic_delta": 0.0001,
                    "full_top5_delta": 0.0002,
                    "recent63_rank_ic_delta": 0.0,
                    "recent63_top5_delta": 0.0003,
                    "recent20_rank_ic_delta": 0.0,
                    "recent20_top5_delta": 0.0004,
                },
                gate_passed=False,
            )
        )

    def test_monthly_delta_stats_counts_positive_and_negative_months(self):
        stats = target.monthly_delta_stats(
            [
                {"rank_ic": "0.01", "top5": "0.02"},
                {"rank_ic": "-0.03", "top5": "0.00"},
                {"rank_ic": "0.02", "top5": "-0.01"},
            ]
        )
        self.assertEqual(stats["month_count"], 3)
        self.assertEqual(stats["month_top5_positive"], 1)
        self.assertEqual(stats["month_top5_nonnegative"], 2)
        self.assertEqual(stats["month_top5_negative"], 1)
        self.assertAlmostEqual(stats["month_min_rank_ic_delta"], -0.03)
        self.assertAlmostEqual(stats["month_min_top5_delta"], -0.01)


if __name__ == "__main__":
    unittest.main()
