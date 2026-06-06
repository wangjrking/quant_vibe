import unittest
import sys
import tempfile
import types

from rolling_train_module import (
    RollingWindow,
    build_rolling_windows,
    parse_args,
    run_rolling_training,
)


class RollingTrainModuleTests(unittest.TestCase):
    def test_build_monthly_fixed_windows_with_label_embargo(self):
        windows = build_rolling_windows(
            data_start="20100101",
            first_test="20260101",
            final_test="20260331",
            train_years=5,
            test_months=1,
            step_months=1,
            embargo_days=10,
        )

        self.assertEqual(
            windows,
            [
                RollingWindow(1, "20210101", "20251221", "20260101", "20260131"),
                RollingWindow(2, "20210201", "20260121", "20260201", "20260228"),
                RollingWindow(3, "20210301", "20260218", "20260301", "20260331"),
            ],
        )

    def test_expanding_windows_keep_training_start_at_data_start(self):
        windows = build_rolling_windows(
            data_start="20200101",
            first_test="20260201",
            final_test="20260228",
            train_years=5,
            test_months=1,
            step_months=1,
            embargo_days=0,
            train_mode="expanding",
        )

        self.assertEqual(windows[0].train_start, "20200101")
        self.assertEqual(windows[0].train_end, "20260131")

    def test_run_rolling_training_calls_train_function_and_merges_metrics(self):
        windows = [
            RollingWindow(1, "20210101", "20251221", "20260101", "20260131"),
            RollingWindow(2, "20210201", "20260121", "20260201", "20260228"),
        ]
        called = []

        def train_fn(window):
            called.append(window.fold)
            return {"prediction_rows": window.fold * 100, "mean_ic": window.fold / 100}

        summary = run_rolling_training(windows, train_fn)

        self.assertEqual(called, [1, 2])
        self.assertEqual(summary[0]["fold"], 1)
        self.assertEqual(summary[1]["prediction_rows"], 200)
        self.assertAlmostEqual(summary[1]["mean_ic"], 0.02)

    def test_cli_defaults_are_dry_run_and_embargoed(self):
        args = parse_args(
            [
                "--data-start",
                "20100101",
                "--first-test",
                "20260101",
                "--final-test",
                "20260331",
            ]
        )

        self.assertFalse(args.execute)
        self.assertEqual(args.train_years, 5)
        self.assertEqual(args.embargo_days, 10)
        self.assertEqual(args.label, "10d_yield_rate")

    def test_train_one_fold_disables_shap_for_backtest_runs(self):
        from rolling_train_module import train_one_fold_with_ai

        calls = []
        fake_ai = types.ModuleType("ai_module")

        class FakePredictions:
            def __len__(self):
                return 3

            def to_sql(self, table_name, con, if_exists, index):
                calls.append(("to_sql", table_name, if_exists, index))

        fake_ai.get_factor_data = lambda *args: ([1, 2], [1, 2], [1, 2, 3], [1, 2, 3], [1, 2], [1, 2, 3])

        def fake_model_assess(*args, **kwargs):
            calls.append(("model_assess", kwargs.get("save_shap")))
            return FakePredictions()

        fake_ai.model_assess = fake_model_assess
        previous_ai = sys.modules.get("ai_module")
        sys.modules["ai_module"] = fake_ai
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                metrics = train_one_fold_with_ai(
                    RollingWindow(1, "20200101", "20250524", "20250604", "20260603"),
                    data_file_url=temp_dir,
                    output_table="pred_1y",
                )
        finally:
            if previous_ai is None:
                sys.modules.pop("ai_module", None)
            else:
                sys.modules["ai_module"] = previous_ai

        self.assertIn(("model_assess", False), calls)
        self.assertIn(("to_sql", "pred_1y", "append", False), calls)
        self.assertEqual(metrics["prediction_rows"], 3)


if __name__ == "__main__":
    unittest.main()
