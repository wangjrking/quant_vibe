import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from ai_module import get_factor_data


class AiModuleTests(unittest.TestCase):
    def test_get_factor_data_can_delegate_to_light_factor_data(self):
        fake_result = (
            pd.DataFrame({"f1": [1.0]}),
            pd.Series([0.1], name="executable_5d_open_return"),
            pd.DataFrame({"f1": [2.0]}),
            pd.Series([0.2], name="executable_5d_open_return"),
            pd.DataFrame({"stock_code": ["000001.SZ"], "trade_date": ["20240101"]}),
            pd.DataFrame({"stock_code": ["000001.SZ"], "trade_date": ["20250101"]}),
        )
        captured = {}

        def fake_get_light_factor_data(**kwargs):
            captured.update(kwargs)
            return fake_result

        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            with patch("ai_module.pd.read_parquet", side_effect=AssertionError("parquet path should be skipped")):
                with patch("ai_module.get_light_factor_data", side_effect=fake_get_light_factor_data, create=True):
                    result = get_factor_data(
                        "20100101",
                        "20250101",
                        "executable_5d_open_return",
                        str(data_dir),
                        stock_pool_path="pool.csv",
                        selected_features=["close_rate", "pb"],
                        use_light_factor_data=True,
                    )

        self.assertEqual(result, fake_result)
        self.assertEqual(captured["data_dir"], str(data_dir))
        self.assertEqual(captured["train_start"], "20100101")
        self.assertEqual(captured["test_start"], "20250101")
        self.assertEqual(captured["label"], "executable_5d_open_return")
        self.assertEqual(captured["stock_pool_path"], "pool.csv")


if __name__ == "__main__":
    unittest.main()
