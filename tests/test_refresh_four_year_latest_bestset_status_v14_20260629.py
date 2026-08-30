import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import refresh_four_year_latest_bestset_status_v14_20260629 as target


class RefreshFourYearLatestBestsetStatusV1420260629Tests(unittest.TestCase):
    def test_should_replace_10d_bestset_requires_positive_full_and_recent_deltas(self):
        self.assertTrue(
            target.should_replace_10d_bestset(
                {
                    "full_rank_ic_delta": 0.0001,
                    "full_top5_delta": 0.0002,
                    "recent63_rank_ic_delta": 0.0003,
                    "recent63_top5_delta": 0.0004,
                    "recent20_rank_ic_delta": 0.0005,
                    "recent20_top5_delta": 0.0006,
                }
            )
        )
        self.assertFalse(
            target.should_replace_10d_bestset(
                {
                    "full_rank_ic_delta": 0.0001,
                    "full_top5_delta": 0.0002,
                    "recent63_rank_ic_delta": -0.0003,
                    "recent63_top5_delta": 0.0004,
                    "recent20_rank_ic_delta": 0.0005,
                    "recent20_top5_delta": 0.0006,
                }
            )
        )

    def test_target_asset_name_is_new_10d_blend_gate_v3(self):
        self.assertIn("active_recent_blend_gate_v3", target.NEW_10D_ASSET)


if __name__ == "__main__":
    unittest.main()
