from pathlib import Path
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "quant" / "main"))

import select_and_try_promote_four_year_10d_newaxis_candidates_20260630 as target


class SelectAndTryPromoteFourYear10DNewaxisCandidates20260630Tests(unittest.TestCase):
    def test_selected_fs40_uses_fs40_asset_name(self):
        selected_dir = target.DEFAULT_EXPERIMENTS[0][1]

        with patch.object(
            target.package_candidate,
            "check_experiment_complete",
            side_effect=[
                {"expected_folds": [1], "missing_folds": [], "bad_status_folds": [], "missing_prediction_files": [], "is_complete": True},
                {"expected_folds": [1, 2], "missing_folds": [2], "bad_status_folds": [], "missing_prediction_files": ["x"], "is_complete": False},
            ],
        ):
            with patch.object(target.try_pipeline, "main") as try_main:
                with patch.object(target.Path, "write_text"):
                    rc = target.main(["--report-dir", "D:/tmp/report"])

        self.assertEqual(rc, 0)
        try_main.assert_called_once_with(
            [
                "--experiment-dir",
                str(selected_dir),
                "--asset",
                "research_10d_four_year_newaxis_fs40_d3_l8_alpha01_topn_balance_candidate_20260630",
            ]
        )


if __name__ == "__main__":
    unittest.main()
