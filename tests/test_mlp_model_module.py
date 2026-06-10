import unittest

import numpy as np
import pandas as pd

from mlp_model_module import build_prediction_frame


class MlpModelModuleTests(unittest.TestCase):
    def test_build_prediction_frame_matches_existing_prediction_schema(self):
        test_data = pd.DataFrame(
            {
                "trade_date": ["20260101"],
                "stock_code": ["000001.SZ"],
                "name": ["Ping An"],
                "close": [10.0],
                "atr_qfq": [0.2],
                "10d_yield_rate": [0.03],
            }
        )
        test_y = pd.Series([0.03], name="10d_yield_rate")

        result = build_prediction_frame(test_data, test_y, np.array([0.04]))

        self.assertEqual(result.loc[0, "pred_prob"], 0.04)
        self.assertEqual(result.loc[0, "stock_code"], "000001.SZ")


if __name__ == "__main__":
    unittest.main()
