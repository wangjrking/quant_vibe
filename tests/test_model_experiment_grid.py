import os
import unittest

from model_experiment_grid import (
    ExperimentSpec,
    build_year_slices,
    expand_experiments,
    temporary_xgb_env,
)


class ModelExperimentGridTests(unittest.TestCase):
    def test_expand_experiments_crosses_core_model_dimensions(self):
        spec = ExperimentSpec(
            labels=["executable_1d_open_return", "executable_5d_open_return"],
            train_windows=["2y", "expanding"],
            feature_top_ns=[80, 160],
            xgb_param_sets=[
                {"max_depth": 2, "reg_lambda": 3.0},
                {"max_depth": 3, "subsample": 0.8},
            ],
        )

        experiments = expand_experiments(spec)

        self.assertEqual(len(experiments), 16)
        self.assertEqual(experiments[0].label, "executable_1d_open_return")
        self.assertEqual(experiments[0].train_mode, "fixed")
        self.assertEqual(experiments[0].train_years, 2)
        self.assertEqual(experiments[-1].train_mode, "expanding")
        self.assertEqual(experiments[-1].feature_top_n, 160)

    def test_build_year_slices_clips_to_requested_range(self):
        self.assertEqual(
            build_year_slices("20220606", "20260613"),
            [
                ("2022", "20220606", "20221231"),
                ("2023", "20230101", "20231231"),
                ("2024", "20240101", "20241231"),
                ("2025", "20250101", "20251231"),
                ("2026", "20260101", "20260613"),
            ],
        )

    def test_temporary_xgb_env_restores_previous_values(self):
        os.environ["XGB_MAX_DEPTH"] = "9"
        with temporary_xgb_env({"max_depth": 2, "reg_lambda": 5.0}):
            self.assertEqual(os.environ["XGB_MAX_DEPTH"], "2")
            self.assertEqual(os.environ["XGB_REG_LAMBDA"], "5.0")
        self.assertEqual(os.environ["XGB_MAX_DEPTH"], "9")
        self.assertNotIn("XGB_REG_LAMBDA", os.environ)
        os.environ.pop("XGB_MAX_DEPTH", None)


if __name__ == "__main__":
    unittest.main()
