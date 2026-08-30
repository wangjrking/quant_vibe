from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "quant" / "main"))

import try_promote_four_year_10d_newaxis_research_pipeline_20260630 as target


class TryPromoteFourYear10DNewaxisResearchPipeline20260630Tests(unittest.TestCase):
    def test_decide_action_returns_wait_when_folds_missing(self):
        status = {
            "is_complete": False,
            "missing_folds": [12, 13],
            "bad_status_folds": [],
            "missing_prediction_files": [],
        }
        decision = target.decide_action(status)
        self.assertEqual(decision["action"], "wait_for_more_folds")
        self.assertEqual(decision["reason"], "experiment_incomplete")

    def test_decide_action_returns_promote_when_complete(self):
        status = {
            "is_complete": True,
            "missing_folds": [],
            "bad_status_folds": [],
            "missing_prediction_files": [],
        }
        decision = target.decide_action(status)
        self.assertEqual(decision["action"], "run_promotion_pipeline")
        self.assertEqual(decision["reason"], "experiment_complete")


if __name__ == "__main__":
    unittest.main()
