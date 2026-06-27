import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from ai_module import (
    _build_xgb_sample_weight,
    _predict_model_outputs,
    _save_model_artifacts,
    _split_train_validation_by_tail_trade_days,
    get_factor_data,
    get_model,
    incre_fit,
    top_return_loss,
)
from model_asset_route import MODEL_FEATURE_MODE_LEGACY


class AiModuleTests(unittest.TestCase):
    def test_get_model_reads_early_stopping_rounds_from_env(self):
        captured = {}

        class FakeRegressor:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        with patch.dict(os.environ, {"XGB_EARLY_STOPPING_ROUNDS": "250"}, clear=False):
            with patch("ai_module.xgb.XGBRegressor", FakeRegressor):
                with patch("ai_module.xgb.XGBClassifier"):
                    model = get_model("reg")

        self.assertIsInstance(model, FakeRegressor)
        self.assertEqual(captured["early_stopping_rounds"], 250)

    def test_top_return_loss_is_negative_mean_of_top_predicted_labels(self):
        import ai_module

        ai_module.group_dates = np.array(["20260102", "20260102", "20260102", "20260105", "20260105"])
        y_true = np.array([0.01, 0.05, -0.02, 0.03, 0.07])
        y_pred = np.array([0.2, 0.9, 0.1, 0.4, 0.3])

        with patch.dict(os.environ, {"XGB_TOP_RETURN_EVAL_K": "1"}, clear=False):
            loss = top_return_loss(y_true, y_pred)

        self.assertAlmostEqual(loss, -0.04)

    def test_get_model_can_use_top_return_loss_eval_metric_from_env(self):
        captured = {}

        class FakeRegressor:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        with patch.dict(os.environ, {"XGB_REG_EVAL_METRIC": "top_return_loss"}, clear=False):
            with patch("ai_module.xgb.XGBRegressor", FakeRegressor):
                with patch("ai_module.xgb.XGBClassifier"):
                    model = get_model("reg")

        self.assertIsInstance(model, FakeRegressor)
        self.assertIs(captured["eval_metric"], top_return_loss)

    def test_build_xgb_sample_weight_emphasizes_daily_top_labels(self):
        index = pd.MultiIndex.from_tuples(
            [
                ("000001.SZ", "20260102"),
                ("000002.SZ", "20260102"),
                ("000003.SZ", "20260102"),
                ("000001.SZ", "20260105"),
                ("000002.SZ", "20260105"),
                ("000003.SZ", "20260105"),
            ],
            names=["stock_code", "trade_date"],
        )
        y = pd.Series([0.01, 0.05, -0.02, -0.03, 0.02, 0.08], index=index)

        with patch.dict(
            os.environ,
            {
                "XGB_SAMPLE_WEIGHT_MODE": "daily_top_quantile",
                "XGB_SAMPLE_WEIGHT_TOP_PCT": "0.34",
                "XGB_SAMPLE_WEIGHT_TOP_MULTIPLIER": "5",
            },
            clear=False,
        ):
            weights = _build_xgb_sample_weight(y)

        np.testing.assert_allclose(weights, np.array([1.0, 5.0, 1.0, 1.0, 1.0, 5.0]))

    def test_incre_fit_passes_optional_sample_weight_to_xgboost_fit(self):
        calls = []

        class FakeModel:
            def fit(self, x, y, **kwargs):
                calls.append(kwargs)

            def predict(self, x):
                return np.array([0.1, 0.2])

        train_index = pd.MultiIndex.from_tuples(
            [
                ("000001.SZ", "20260102"),
                ("000002.SZ", "20260102"),
                ("000003.SZ", "20260102"),
            ],
            names=["stock_code", "trade_date"],
        )
        test_index = pd.MultiIndex.from_tuples(
            [("000004.SZ", "20260105"), ("000005.SZ", "20260105")],
            names=["stock_code", "trade_date"],
        )
        train_x = pd.DataFrame({"f": [1.0, 2.0, 3.0]}, index=train_index)
        train_y = pd.Series([0.01, 0.05, -0.02], index=train_index)
        test_x = pd.DataFrame({"f": [4.0, 5.0]}, index=test_index)
        test_y = pd.Series([0.03, 0.04], index=test_index)

        with patch.dict(
            os.environ,
            {
                "XGB_SAMPLE_WEIGHT_MODE": "daily_top_quantile",
                "XGB_SAMPLE_WEIGHT_TOP_PCT": "0.34",
                "XGB_SAMPLE_WEIGHT_TOP_MULTIPLIER": "4",
            },
            clear=False,
        ):
            incre_fit(FakeModel(), train_x, train_y, test_x, test_y, "data_file", save_shap=False)

        self.assertEqual(len(calls), 1)
        np.testing.assert_allclose(calls[0]["sample_weight"], np.array([1.0, 4.0, 1.0]))

    def test_split_train_validation_uses_last_trade_days_only(self):
        index = pd.MultiIndex.from_tuples(
            [
                ("000001.SZ", "20260102"),
                ("000002.SZ", "20260102"),
                ("000001.SZ", "20260105"),
                ("000002.SZ", "20260105"),
                ("000001.SZ", "20260106"),
                ("000002.SZ", "20260106"),
            ],
            names=["stock_code", "trade_date"],
        )
        x = pd.DataFrame({"f": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}, index=index)
        y = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], index=index)

        fit_x, fit_y, eval_data = _split_train_validation_by_tail_trade_days(x, y, tail_days=1)

        self.assertEqual(len(fit_x), 4)
        self.assertEqual(len(fit_y), 4)
        self.assertEqual(len(eval_data[0]), 2)
        self.assertEqual(set(eval_data[0].index.get_level_values("trade_date")), {"20260106"})
        self.assertEqual(set(fit_x.index.get_level_values("trade_date")), {"20260102", "20260105"})

    def test_incre_fit_can_use_train_tail_validation_instead_of_test_eval(self):
        calls = []

        class FakeModel:
            def fit(self, x, y, **kwargs):
                calls.append((x.copy(), y.copy(), kwargs))

            def predict(self, x):
                return np.array([0.1, 0.2])

        train_index = pd.MultiIndex.from_tuples(
            [
                ("000001.SZ", "20260102"),
                ("000002.SZ", "20260102"),
                ("000001.SZ", "20260105"),
                ("000002.SZ", "20260105"),
                ("000001.SZ", "20260106"),
                ("000002.SZ", "20260106"),
            ],
            names=["stock_code", "trade_date"],
        )
        test_index = pd.MultiIndex.from_tuples(
            [("000004.SZ", "20260109"), ("000005.SZ", "20260109")],
            names=["stock_code", "trade_date"],
        )
        train_x = pd.DataFrame({"f": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}, index=train_index)
        train_y = pd.Series([0.01, 0.02, 0.03, 0.04, 0.50, 0.60], index=train_index)
        test_x = pd.DataFrame({"f": [7.0, 8.0]}, index=test_index)
        test_y = pd.Series([0.70, 0.80], index=test_index)

        with patch.dict(
            os.environ,
            {
                "XGB_VALIDATION_MODE": "train_tail_days",
                "XGB_VALIDATION_TAIL_DAYS": "1",
            },
            clear=False,
        ):
            incre_fit(FakeModel(), train_x, train_y, test_x, test_y, "data_file", save_shap=False)

        self.assertEqual(len(calls), 1)
        fit_x, fit_y, kwargs = calls[0]
        self.assertEqual(set(fit_x.index.get_level_values("trade_date")), {"20260102", "20260105"})
        self.assertEqual(set(kwargs["eval_set"][0][0].index.get_level_values("trade_date")), {"20260106"})
        self.assertNotIn("20260109", set(kwargs["eval_set"][0][0].index.get_level_values("trade_date")))

    def test_save_model_artifacts_persists_best_iteration_metadata(self):
        class FakeModel:
            best_iteration = 123
            best_score = 0.456

            def save_model(self, path):
                Path(path).write_text("{}", encoding="utf-8")

            def get_params(self):
                return {"device": "cuda", "early_stopping_rounds": 250}

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.json"
            metadata_path = Path(temp_dir) / "model_metadata.json"
            with patch.dict(
                os.environ,
                {
                    "XGB_MODEL_SAVE_PATH": str(model_path),
                    "XGB_MODEL_METADATA_PATH": str(metadata_path),
                },
                clear=False,
            ):
                _save_model_artifacts(FakeModel(), ["f1", "f2"])

            payload = json.loads(metadata_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["best_iteration"], 123)
        self.assertEqual(payload["best_score"], 0.456)
        self.assertEqual(payload["feature_count"], 2)

    def test_predict_model_outputs_prefers_inplace_predict_for_gpu_xgboost(self):
        class FakeBooster:
            def __init__(self):
                self.calls = []

            def inplace_predict(self, data):
                self.calls.append(data.copy())
                return np.array([0.11, 0.22], dtype=float)

        class FakeGpuModel:
            def __init__(self):
                self.booster = FakeBooster()
                self.predict_called = False

            def get_booster(self):
                return self.booster

            def get_params(self):
                return {"device": "cuda"}

            def predict(self, data):
                self.predict_called = True
                raise AssertionError("predict() should not be used on the GPU fast path")

        test_x = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})
        model = FakeGpuModel()

        pred = _predict_model_outputs(model, test_x)

        np.testing.assert_allclose(pred, np.array([0.11, 0.22], dtype=float))
        self.assertFalse(model.predict_called)
        self.assertEqual(len(model.booster.calls), 1)

    def test_predict_model_outputs_coerces_nullable_dataframe_for_gpu_xgboost(self):
        test_case = self

        class FakeBooster:
            def __init__(self):
                self.calls = []

            def inplace_predict(self, data):
                self.calls.append(data.copy())
                test_case.assertEqual(data.dtype, np.dtype("float32"))
                test_case.assertFalse(any(value is pd.NA for value in data.ravel()))
                test_case.assertTrue(np.isnan(data[1, 0]))
                test_case.assertTrue(np.isnan(data[1, 1]))
                return np.array([0.31, 0.42], dtype=float)

        class FakeGpuModel:
            def __init__(self):
                self.booster = FakeBooster()

            def get_booster(self):
                return self.booster

            def get_params(self):
                return {"device": "cuda"}

            def predict(self, data):
                raise AssertionError("predict() should not be used on the GPU fast path")

        test_x = pd.DataFrame(
            {
                "f1": pd.Series([1.0, pd.NA], dtype="Float64"),
                "f2": pd.Series(["3.5", "bad"], dtype="string"),
            }
        )
        model = FakeGpuModel()

        pred = _predict_model_outputs(model, test_x)

        np.testing.assert_allclose(pred, np.array([0.31, 0.42], dtype=float))
        self.assertEqual(len(model.booster.calls), 1)

    def test_predict_model_outputs_sets_cpu_when_cupy_unavailable_for_cuda_booster(self):
        class FakeBooster:
            def __init__(self):
                self.params = []

            def set_param(self, params):
                self.params.append(dict(params))

            def inplace_predict(self, data):
                self.seen = data.copy()
                return np.array([0.51, 0.62], dtype=float)

        class FakeGpuModel:
            def __init__(self):
                self.booster = FakeBooster()

            def get_booster(self):
                return self.booster

            def get_params(self):
                return {"device": "cuda"}

            def predict(self, data):
                raise AssertionError("predict() should not be used on the GPU fast path")

        with patch.dict("sys.modules", {"cupy": None}):
            model = FakeGpuModel()
            pred = _predict_model_outputs(model, pd.DataFrame({"f1": [1.0, 2.0]}))

        np.testing.assert_allclose(pred, np.array([0.51, 0.62], dtype=float))
        self.assertEqual(model.booster.params, [{"device": "cpu"}])
        self.assertEqual(model.booster.seen.dtype, np.dtype("float32"))

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

    def test_get_factor_data_reads_split_feature_and_label_parts_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            feature_dir = data_dir / "production_factor_parts"
            label_dir = data_dir / "prediction_label_parts"
            feature_dir.mkdir()
            label_dir.mkdir()

            pd.DataFrame(
                [
                    {"stock_code": "000001.SZ", "trade_date": "20240102", "close_rate": 1.0, "pb": 2.0},
                    {"stock_code": "000001.SZ", "trade_date": "20250102", "close_rate": 1.5, "pb": 2.5},
                ]
            ).to_parquet(feature_dir / "part-000.parquet", index=False)
            pd.DataFrame(
                [
                    {
                        "stock_code": "000001.SZ",
                        "trade_date": "20240102",
                        "executable_5d_open_return": 0.02,
                        "5d_yield_rate": 0.03,
                        "open6_yield_rate": 0.04,
                        "10d_yield_rate": 0.05,
                    },
                    {
                        "stock_code": "000001.SZ",
                        "trade_date": "20250102",
                        "executable_5d_open_return": 0.06,
                        "5d_yield_rate": 0.07,
                        "open6_yield_rate": 0.08,
                        "10d_yield_rate": 0.09,
                    },
                ]
            ).to_parquet(label_dir / "part-000.parquet", index=False)

            train_x, train_y, test_x, test_y, train_data, test_data = get_factor_data(
                "20240101",
                "20250101",
                "executable_5d_open_return",
                str(data_dir),
                selected_features=["close_rate", "pb"],
            )

        self.assertEqual(list(train_x.columns), ["close_rate", "pb"])
        self.assertEqual(list(test_x.columns), ["close_rate", "pb"])
        self.assertEqual(train_y.name, "executable_5d_open_return")
        self.assertEqual(test_y.name, "executable_5d_open_return")
        self.assertEqual(len(train_data), 1)
        self.assertEqual(len(test_data), 1)

    def test_get_factor_data_rejects_legacy_feature_chain_without_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(RuntimeError, "archived and disabled by default"):
                get_factor_data(
                    "20240101",
                    "20250101",
                    "executable_5d_open_return",
                    temp_dir,
                    feature_source=MODEL_FEATURE_MODE_LEGACY,
                )

    def test_get_factor_data_split_chain_does_not_repeat_data_start_filter(self):
        frame = pd.DataFrame(
            [
                {
                    "stock_code": "000001.SZ",
                    "trade_date": "20240102",
                    "name": "PingAn",
                    "st_type": None,
                    "limit_times": None,
                    "close_rate": 1.0,
                    "pb": 2.0,
                    "executable_5d_open_return": 0.02,
                    "5d_yield_rate": 0.03,
                    "open6_yield_rate": 0.04,
                    "10d_yield_rate": 0.05,
                },
                {
                    "stock_code": "000001.SZ",
                    "trade_date": "20250102",
                    "name": "PingAn",
                    "st_type": None,
                    "limit_times": None,
                    "close_rate": 1.5,
                    "pb": 2.5,
                    "executable_5d_open_return": 0.06,
                    "5d_yield_rate": 0.07,
                    "open6_yield_rate": 0.08,
                    "10d_yield_rate": 0.09,
                },
            ]
        )

        original_ge = pd.Series.__ge__

        def guarded_ge(series, other):
            if getattr(series, "name", None) == "trade_date" and other == "20240101":
                raise AssertionError("redundant data_start filter on split chain")
            return original_ge(series, other)

        with patch("ai_module._read_split_factor_data", return_value=frame.copy()):
            with patch.object(pd.Series, "__ge__", new=guarded_ge):
                train_x, train_y, test_x, test_y, train_data, test_data = get_factor_data(
                    "20240101",
                    "20250101",
                    "executable_5d_open_return",
                    "unused-data-dir",
                    selected_features=["close_rate", "pb"],
                )

        self.assertEqual(list(train_x.columns), ["close_rate", "pb"])
        self.assertEqual(list(test_x.columns), ["close_rate", "pb"])
        self.assertEqual(train_y.name, "executable_5d_open_return")
        self.assertEqual(test_y.name, "executable_5d_open_return")
        self.assertEqual(len(train_data), 1)
        self.assertEqual(len(test_data), 1)


if __name__ == "__main__":
    unittest.main()
