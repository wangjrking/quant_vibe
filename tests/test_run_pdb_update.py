import unittest
from unittest.mock import patch

import run_pdb_update


class RunPdbUpdateTests(unittest.TestCase):
    def test_parse_args_accepts_stock_pool_and_model_backend(self):
        args = run_pdb_update.parse_args(
            [
                "--stock-pool",
                "data_file/stock_pool_hs300_zz500.csv",
                "--model-backend",
                "mlp",
            ]
        )

        self.assertEqual(args.stock_pool, "data_file/stock_pool_hs300_zz500.csv")
        self.assertEqual(args.model_backend, "mlp")

    @patch("run_pdb_update.model_assess")
    @patch("run_pdb_update.get_factor_data")
    def test_main_passes_stock_pool_to_factor_loader(self, fake_get_factor_data, fake_model_assess):
        class FakeMatrix:
            shape = (1, 1)

        class FakeFrame:
            shape = (0, 0)

            def __getitem__(self, key):
                return self

            def astype(self, value):
                return self

            def max(self):
                return "20260604"

            def to_sql(self, *args, **kwargs):
                return None

        fake_get_factor_data.return_value = (FakeMatrix(), "train_y", FakeMatrix(), "test_y", FakeFrame(), FakeFrame())
        fake_model_assess.return_value = FakeFrame()

        with patch("run_pdb_update.sqlite3.connect"):
            run_pdb_update.main(
                [
                    "--stock-pool",
                    "data_file/stock_pool_hs300_zz500.csv",
                    "--output-table",
                    "dummy",
                ]
            )

        self.assertEqual(fake_get_factor_data.call_args.args[-1], "data_file/stock_pool_hs300_zz500.csv")


if __name__ == "__main__":
    unittest.main()
