import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import refresh_four_year_latest_bestset_status_v13_20260629 as target


class RefreshFourYearLatestBestsetStatusV1320260629Tests(unittest.TestCase):
    def test_should_replace_3d_bestset_requires_positive_rankic_and_top5(self):
        self.assertTrue(
            target.should_replace_3d_bestset(
                {
                    "full_rank_ic": 0.0001,
                    "full_top5": 0.0002,
                    "recent63_rank_ic": 0.0003,
                    "recent20_rank_ic": 0.0004,
                    "recent63_top5": 0.0005,
                    "recent20_top5": 0.0006,
                }
            )
        )
        self.assertFalse(
            target.should_replace_3d_bestset(
                {
                    "full_rank_ic": -0.0001,
                    "full_top5": 0.0002,
                    "recent63_rank_ic": 0.0003,
                    "recent20_rank_ic": 0.0004,
                    "recent63_top5": 0.0005,
                    "recent20_top5": 0.0006,
                }
            )
        )

    def test_target_asset_name_is_new_refined_3d_candidate(self):
        self.assertIn("hybrid_refine_candidate", target.NEW_3D_ASSET)


if __name__ == "__main__":
    unittest.main()
