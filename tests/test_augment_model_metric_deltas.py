import unittest

import pandas as pd

from augment_model_metric_deltas import rank_ic_positive_ratio_by_window


class AugmentModelMetricDeltasTest(unittest.TestCase):
    def test_rank_ic_positive_ratio_by_window_uses_inclusive_dates(self):
        daily = pd.DataFrame(
            [
                {"trade_date": "20240102", "candidate_rank_ic": 0.1, "formal_rank_ic": -0.1},
                {"trade_date": "20240103", "candidate_rank_ic": -0.2, "formal_rank_ic": -0.1},
                {"trade_date": "20240104", "candidate_rank_ic": 0.3, "formal_rank_ic": 0.2},
            ]
        )
        windows = pd.DataFrame(
            [
                {"window": "full", "eval_date_min": "20240102", "eval_date_max": "20240104"},
                {"window": "recent2", "eval_date_min": "20240103", "eval_date_max": "20240104"},
            ]
        )

        result = rank_ic_positive_ratio_by_window(daily, windows)

        by_window = {item["window"]: item for item in result}
        self.assertAlmostEqual(by_window["full"]["candidate_rank_ic_positive_ratio"], 2 / 3)
        self.assertAlmostEqual(by_window["full"]["formal_rank_ic_positive_ratio"], 1 / 3)
        self.assertAlmostEqual(by_window["full"]["rank_ic_positive_ratio_delta"], 1 / 3)
        self.assertAlmostEqual(by_window["recent2"]["candidate_rank_ic_positive_ratio"], 1 / 2)
        self.assertAlmostEqual(by_window["recent2"]["formal_rank_ic_positive_ratio"], 1 / 2)
        self.assertAlmostEqual(by_window["recent2"]["rank_ic_positive_ratio_delta"], 0.0)


if __name__ == "__main__":
    unittest.main()
