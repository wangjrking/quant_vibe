import unittest
import sys
import tempfile
import types

from rolling_train_module import (
    FoldFeatureSelectionConfig,
    RollingWindow,
    build_validation_window,
    build_rolling_windows,
    build_fold_feature_selection_fn,
    parse_args,
    run_rolling_training,
    train_predict_slice,
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

    def test_execute_rolling_training_can_resume_from_later_fold_without_replace(self):
        from rolling_train_module import execute_rolling_training

        windows = [
            RollingWindow(1, "20210101", "20251221", "20260101", "20260131"),
            RollingWindow(2, "20210201", "20260121", "20260201", "20260228"),
            RollingWindow(3, "20210301", "20260218", "20260301", "20260331"),
        ]
        calls = []

        def fake_train(window, **kwargs):
            calls.append((window.fold, kwargs.get("if_exists")))
            return {"prediction_rows": window.fold}

        import rolling_train_module as module

        original = module.train_one_fold_with_ai
        module.train_one_fold_with_ai = fake_train
        try:
            summary = execute_rolling_training(
                windows,
                data_file_url="unused",
                output_table="pred_resume",
                start_fold=2,
                resume_existing_table=True,
            )
        finally:
            module.train_one_fold_with_ai = original

        self.assertEqual(calls, [(2, "append"), (3, "append")])
        self.assertEqual([row["fold"] for row in summary], [2, 3])

    def test_execute_rolling_training_can_stop_at_end_fold(self):
        from rolling_train_module import execute_rolling_training

        windows = [
            RollingWindow(1, "20210101", "20251221", "20260101", "20260131"),
            RollingWindow(2, "20210201", "20260121", "20260201", "20260228"),
            RollingWindow(3, "20210301", "20260218", "20260301", "20260331"),
        ]
        calls = []

        def fake_train(window, **kwargs):
            calls.append((window.fold, kwargs.get("if_exists")))
            return {"prediction_rows": window.fold}

        import rolling_train_module as module

        original = module.train_one_fold_with_ai
        module.train_one_fold_with_ai = fake_train
        try:
            summary = execute_rolling_training(
                windows,
                data_file_url="unused",
                output_table="pred_resume",
                start_fold=2,
                end_fold=2,
                resume_existing_table=True,
            )
        finally:
            module.train_one_fold_with_ai = original

        self.assertEqual(calls, [(2, "append")])
        self.assertEqual([row["fold"] for row in summary], [2])

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
        self.assertEqual(args.start_fold, 1)
        self.assertIsNone(args.end_fold)
        self.assertFalse(args.resume_existing_table)

    def test_train_one_fold_disables_shap_for_backtest_runs(self):
        from rolling_train_module import train_one_fold_with_ai

        calls = []
        fake_ai = types.ModuleType("ai_module")

        class FakePredictions:
            def __len__(self):
                return 3

            def to_sql(self, table_name, con, if_exists, index):
                calls.append(("to_sql", table_name, if_exists, index))

        fake_ai.get_factor_data = lambda *args, **kwargs: ([1, 2], [1, 2], [1, 2, 3], [1, 2, 3], [1, 2], [1, 2, 3])

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

    def test_train_one_fold_passes_selected_features_when_provided(self):
        from rolling_train_module import train_one_fold_with_ai

        calls = []
        fake_ai = types.ModuleType("ai_module")

        class FakePredictions:
            def __len__(self):
                return 2

            def to_sql(self, table_name, con, if_exists, index):
                calls.append(("to_sql", table_name, if_exists, index))

        def fake_get_factor_data(*args, **kwargs):
            calls.append(("get_factor_data", kwargs.get("selected_features")))
            return ([1], [1], [1, 2], [1, 2], [1], [1, 2])

        fake_ai.get_factor_data = fake_get_factor_data
        fake_ai.model_assess = lambda *args, **kwargs: FakePredictions()
        previous_ai = sys.modules.get("ai_module")
        sys.modules["ai_module"] = fake_ai
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                train_one_fold_with_ai(
                    RollingWindow(1, "20200101", "20250524", "20250604", "20260603"),
                    data_file_url=temp_dir,
                    output_table="pred_selected",
                    selected_features=["feature_a", "feature_b"],
                )
        finally:
            if previous_ai is None:
                sys.modules.pop("ai_module", None)
            else:
                sys.modules["ai_module"] = previous_ai

        self.assertIn(("get_factor_data", ["feature_a", "feature_b"]), calls)

    def test_train_one_fold_passes_stock_pool_path_when_provided(self):
        from rolling_train_module import train_one_fold_with_ai

        calls = []
        fake_ai = types.ModuleType("ai_module")

        class FakePredictions:
            def __len__(self):
                return 2

            def to_sql(self, table_name, con, if_exists, index):
                return None

        def fake_get_factor_data(*args, **kwargs):
            calls.append(("get_factor_data", kwargs.get("stock_pool_path")))
            return ([1], [1], [1, 2], [1, 2], [1], [1, 2])

        fake_ai.get_factor_data = fake_get_factor_data
        fake_ai.model_assess = lambda *args, **kwargs: FakePredictions()
        previous_ai = sys.modules.get("ai_module")
        sys.modules["ai_module"] = fake_ai
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                train_one_fold_with_ai(
                    RollingWindow(1, "20200101", "20250524", "20250604", "20260603"),
                    data_file_url=temp_dir,
                    output_table="pred_pool",
                    stock_pool_path="pool.csv",
                )
        finally:
            if previous_ai is None:
                sys.modules.pop("ai_module", None)
            else:
                sys.modules["ai_module"] = previous_ai

        self.assertIn(("get_factor_data", "pool.csv"), calls)

    def test_train_one_fold_passes_use_light_factor_data_when_enabled(self):
        from rolling_train_module import train_one_fold_with_ai

        calls = []
        fake_ai = types.ModuleType("ai_module")

        class FakePredictions:
            def __len__(self):
                return 2

            def to_sql(self, table_name, con, if_exists, index):
                return None

        def fake_get_factor_data(*args, **kwargs):
            calls.append(("get_factor_data", kwargs.get("use_light_factor_data")))
            return ([1], [1], [1, 2], [1, 2], [1], [1, 2])

        fake_ai.get_factor_data = fake_get_factor_data
        fake_ai.model_assess = lambda *args, **kwargs: FakePredictions()
        previous_ai = sys.modules.get("ai_module")
        sys.modules["ai_module"] = fake_ai
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                train_one_fold_with_ai(
                    RollingWindow(1, "20200101", "20250524", "20250604", "20260603"),
                    data_file_url=temp_dir,
                    output_table="pred_light",
                    use_light_factor_data=True,
                )
        finally:
            if previous_ai is None:
                sys.modules.pop("ai_module", None)
            else:
                sys.modules["ai_module"] = previous_ai

        self.assertIn(("get_factor_data", True), calls)

    def test_build_fold_feature_selection_fn_filters_to_train_window(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            rows = []
            for trade_date in ("20240101", "20240201"):
                for idx in range(120):
                    rows.append(
                        {
                            "trade_date": trade_date,
                            "stock_code": f"S{trade_date}_{idx:03d}",
                            "feature_keep": idx,
                            "feature_drop": 119 - idx,
                            "10d_yield_rate": idx,
                        }
                    )
            for idx in range(120):
                rows.append(
                    {
                        "trade_date": "20240501",
                        "stock_code": f"FUT_{idx:03d}",
                        "feature_keep": 1000 - idx,
                        "feature_drop": idx,
                        "10d_yield_rate": idx,
                    }
                )
            frame = __import__("pandas").DataFrame(rows)
            data_dir = tempfile.mkdtemp(dir=temp_dir)
            __import__("pathlib").Path(data_dir, "stock_factor_data.parquet")
            frame.to_parquet(__import__("pathlib").Path(data_dir) / "stock_factor_data.parquet")

            selector = build_fold_feature_selection_fn(
                data_file_url=data_dir,
                config=FoldFeatureSelectionConfig(label="10d_yield_rate", top_n=1, min_abs_ic=0.5),
            )
            selected = selector(RollingWindow(1, "20240101", "20240228", "20240301", "20240331"))
            self.assertEqual(selected, ["feature_keep"])

    def test_build_fold_feature_selection_fn_can_use_light_factor_data(self):
        captured = {}
        fake_fast = types.ModuleType("fast_feature_selection")
        fake_fast.score_features_fast = lambda frame, **kwargs: (
            [{"feature": "close_rate", "abs_mean_ic": 0.1}],
            ["close_rate"],
        )
        fake_fast.write_score_csv = lambda rows, path: None

        fake_light = types.ModuleType("light_factor_module")

        def fake_read_raw_frame(db_path, start, end, needed_columns, stock_pool_path=None):
            captured["read_raw_frame"] = {
                "db_path": str(db_path),
                "start": start,
                "end": end,
                "stock_pool_path": stock_pool_path,
            }
            return __import__("pandas").DataFrame(
                [
                    {
                        "trade_date": "20240102",
                        "stock_code": "000001.SZ",
                        "name": "PingAn",
                        "industry": "Bank",
                        "st_type": None,
                        "limit_times": None,
                        "open": 10,
                        "high": 11,
                        "low": 9,
                        "close": 10,
                        "pre_close": 9.5,
                        "atr_qfq": 0.5,
                    }
                ]
            )

        def fake_build_light_factor_frame(raw, features, label):
            captured["build_light_factor_frame"] = {
                "features": list(features),
                "label": label,
            }
            return __import__("pandas").DataFrame(
                [
                    {
                        "trade_date": "20240102",
                        "stock_code": "000001.SZ",
                        "name": "PingAn",
                        "close_rate": 1.0,
                        "executable_5d_open_return": 0.02,
                    }
                ]
            )

        fake_light.read_raw_frame = fake_read_raw_frame
        fake_light.build_light_factor_frame = fake_build_light_factor_frame
        fake_light.DERIVED_FEATURES = {"close_rate"}
        fake_light.META_COLUMNS = ["trade_date", "stock_code", "name"]

        previous_fast = sys.modules.get("fast_feature_selection")
        previous_light = sys.modules.get("light_factor_module")
        sys.modules["fast_feature_selection"] = fake_fast
        sys.modules["light_factor_module"] = fake_light
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                selector = build_fold_feature_selection_fn(
                    data_file_url=temp_dir,
                    config=FoldFeatureSelectionConfig(label="executable_5d_open_return", top_n=1, min_abs_ic=0.005),
                    stock_pool_path="all_a.csv",
                    use_light_factor_data=True,
                )
                selected = selector(RollingWindow(1, "20240101", "20240529", "20240604", "20240903"))
        finally:
            if previous_fast is None:
                sys.modules.pop("fast_feature_selection", None)
            else:
                sys.modules["fast_feature_selection"] = previous_fast
            if previous_light is None:
                sys.modules.pop("light_factor_module", None)
            else:
                sys.modules["light_factor_module"] = previous_light

        self.assertEqual(selected, ["close_rate"])
        self.assertEqual(captured["read_raw_frame"]["stock_pool_path"], "all_a.csv")
        self.assertEqual(captured["build_light_factor_frame"]["label"], "executable_5d_open_return")

    def test_build_validation_window_carves_validation_slice_before_test(self):
        validation = build_validation_window(
            RollingWindow(4, "20100101", "20250524", "20250604", "20250903"),
            validation_months=12,
            embargo_days=10,
        )
        self.assertEqual(validation.fold, 4)
        self.assertEqual(validation.validation_start, "20240604")
        self.assertEqual(validation.validation_end, "20250524")
        self.assertEqual(validation.train_end, "20240524")
        self.assertEqual(validation.test_start, "20250604")

    def test_train_predict_slice_passes_selected_features(self):
        calls = []
        fake_ai = types.ModuleType("ai_module")

        class FakePredictions:
            def __len__(self):
                return 2

        def fake_get_factor_data(*args, **kwargs):
            calls.append(("get_factor_data", kwargs.get("selected_features")))
            return ([1], [1], [1, 2], [1, 2], [1], [1, 2])

        def fake_model_assess(*args, **kwargs):
            calls.append(("model_assess", kwargs.get("save_shap")))
            return FakePredictions()

        fake_ai.get_factor_data = fake_get_factor_data
        fake_ai.model_assess = fake_model_assess
        previous_ai = sys.modules.get("ai_module")
        sys.modules["ai_module"] = fake_ai
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                result = train_predict_slice(
                    train_start="20200101",
                    train_end="20250524",
                    predict_start="20250604",
                    predict_end="20260603",
                    data_file_url=temp_dir,
                    label="10d_yield_rate",
                    selected_features=["feature_x"],
                )
        finally:
            if previous_ai is None:
                sys.modules.pop("ai_module", None)
            else:
                sys.modules["ai_module"] = previous_ai

        self.assertIsNotNone(result)
        self.assertIn(("get_factor_data", ["feature_x"]), calls)
        self.assertIn(("model_assess", False), calls)


if __name__ == "__main__":
    unittest.main()
