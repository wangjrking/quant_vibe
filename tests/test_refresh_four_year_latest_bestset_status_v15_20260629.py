import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import refresh_four_year_latest_bestset_status_v15_20260629 as target


class RefreshFourYearLatestBestsetStatusV1520260629Tests(unittest.TestCase):
    def test_should_replace_5d_bestset_requires_positive_direct_deltas_and_nonnegative_monthly_tail(self):
        self.assertTrue(
            target.should_replace_5d_bestset(
                {
                    "full_rank_ic_delta": 0.0001,
                    "full_top5_delta": 0.0002,
                    "recent63_top5_delta": 0.0,
                    "recent20_top5_delta": 0.0,
                },
                min_month_top5_delta_vs_base=0.0,
                gate_passed=True,
            )
        )
        self.assertFalse(
            target.should_replace_5d_bestset(
                {
                    "full_rank_ic_delta": 0.0001,
                    "full_top5_delta": 0.0002,
                    "recent63_top5_delta": 0.0,
                    "recent20_top5_delta": 0.0,
                },
                min_month_top5_delta_vs_base=-0.0001,
                gate_passed=True,
            )
        )
        self.assertFalse(
            target.should_replace_5d_bestset(
                {
                    "full_rank_ic_delta": -0.0001,
                    "full_top5_delta": 0.0002,
                    "recent63_top5_delta": 0.0,
                    "recent20_top5_delta": 0.0,
                },
                min_month_top5_delta_vs_base=0.0,
                gate_passed=True,
            )
        )

    def test_target_asset_name_is_front_rank_state_gate_v4(self):
        self.assertIn("front_rank_state_gate_v4", target.NEW_5D_ASSET)


if __name__ == "__main__":
    unittest.main()
