import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import run_parallel_expanding2010_folds as target


class RunParallelExpanding2010FoldsTest(unittest.TestCase):
    def test_merge_only_with_partial_fold_range_merges_all_completed_prediction_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "out"
            fold_prediction_dir = output_dir / "fold_predictions"
            fold_prediction_dir.mkdir(parents=True, exist_ok=True)
            fold01 = fold_prediction_dir / "fold01.parquet"
            fold02 = fold_prediction_dir / "fold02.parquet"
            fold01.write_bytes(b"fold01")
            fold02.write_bytes(b"fold02")
            fallback_db = Path(tmp) / "model_predictions" / "MODEL_PREDICTIONS.db"

            windows = [
                SimpleNamespace(fold=1, to_dict=lambda: {"fold": 1}),
                SimpleNamespace(fold=2, to_dict=lambda: {"fold": 2}),
            ]
            completed_rows = [
                {"fold": "1", "prediction_path": str(fold01), "status": "ok", "exists": "True"},
                {"fold": "2", "prediction_path": str(fold02), "status": "ok", "exists": "True"},
            ]

            with patch.object(target, "build_rolling_windows", return_value=windows):
                with patch.object(target, "_read_rows", return_value=completed_rows):
                    with patch.object(
                        target,
                        "resolve_model_prediction_db_path",
                        side_effect=RuntimeError("L4 prediction mainline is registered as DuckDB; use prediction manifests."),
                    ):
                        with patch.object(
                            target,
                            "_resolve_independent_research_prediction_db_path",
                            return_value=fallback_db,
                        ):
                            with patch.object(target, "_merge_prediction_files", return_value={"rows": 2}) as merge_files:
                                rc = target.main(
                                    [
                                        "--data-file-url",
                                        tmp,
                                        "--first-test",
                                        "20220606",
                                        "--final-test",
                                        "20220630",
                                        "--label",
                                        "executable_10d_open_return",
                                        "--model-type",
                                        "reg",
                                        "--output-table",
                                        "stock_predict_data_research",
                                        "--output-dir",
                                        str(output_dir),
                                        "--experiment-name",
                                        "unit",
                                        "--start-fold",
                                        "2",
                                        "--end-fold",
                                        "2",
                                        "--merge-only",
                                    ]
                                )

            self.assertEqual(rc, 0)
            merge_files.assert_called_once_with(
                fallback_db,
                "stock_predict_data_research",
                [fold01, fold02],
            )

    def test_no_merge_does_not_resolve_prediction_db(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "out"
            with patch.object(target, "_run_fold") as run_fold:
                run_fold.return_value = {
                    "fold": 1,
                    "status": "ok",
                    "prediction_path": str(output_dir / "fold_predictions" / "fold01.parquet"),
                    "exists": False,
                    "rows": 0,
                }
                with patch.object(target, "resolve_model_prediction_db_path") as resolve_db:
                    rc = target.main(
                        [
                            "--data-file-url",
                            tmp,
                            "--first-test",
                            "20220606",
                            "--final-test",
                            "20220630",
                            "--label",
                            "executable_10d_open_return",
                            "--model-type",
                            "reg",
                            "--output-table",
                            "stock_predict_data_research",
                            "--output-dir",
                            str(output_dir),
                            "--experiment-name",
                            "unit",
                            "--start-fold",
                            "1",
                            "--end-fold",
                            "1",
                            "--no-merge",
                        ]
                    )

            self.assertEqual(rc, 0)
            resolve_db.assert_not_called()
            self.assertTrue((output_dir / "merge_meta.json").exists())

    def test_main_falls_back_to_independent_sqlite_when_l4_mainline_is_duckdb(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "out"
            output_dir.mkdir(parents=True, exist_ok=True)
            fold_prediction = output_dir / "fold_predictions" / "fold01.parquet"
            fold_prediction.parent.mkdir(parents=True, exist_ok=True)
            fold_prediction.write_bytes(b"stub")
            fallback_db = Path(tmp) / "model_predictions" / "MODEL_PREDICTIONS.db"

            with patch.object(target, "_run_fold") as run_fold:
                run_fold.return_value = {
                    "fold": 1,
                    "status": "ok",
                    "prediction_path": str(fold_prediction),
                    "exists": True,
                    "rows": 1,
                    "max_trade_date": "20220630",
                }
                with patch.object(
                    target,
                    "resolve_model_prediction_db_path",
                    side_effect=RuntimeError("L4 prediction mainline is registered as DuckDB; use prediction manifests."),
                ):
                    with patch.object(
                        target,
                        "_resolve_independent_research_prediction_db_path",
                        return_value=fallback_db,
                    ) as resolve_fallback:
                        with patch.object(target, "_merge_prediction_files", return_value={"rows": 1}) as merge_files:
                            rc = target.main(
                                [
                                    "--data-file-url",
                                    tmp,
                                    "--first-test",
                                    "20220606",
                                    "--final-test",
                                    "20220630",
                                    "--label",
                                    "executable_10d_open_return",
                                    "--model-type",
                                    "reg",
                                    "--output-table",
                                    "stock_predict_data_research",
                                    "--output-dir",
                                    str(output_dir),
                                    "--experiment-name",
                                    "unit",
                                    "--start-fold",
                                    "1",
                                    "--end-fold",
                                    "1",
                                ]
                            )

            self.assertEqual(rc, 0)
            resolve_fallback.assert_called_once()
            merge_files.assert_called_once_with(
                fallback_db,
                "stock_predict_data_research",
                [fold_prediction],
            )


if __name__ == "__main__":
    unittest.main()
