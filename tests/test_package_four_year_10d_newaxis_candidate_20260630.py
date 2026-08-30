import csv
import tempfile
import unittest
from pathlib import Path

from package_four_year_10d_newaxis_candidate_20260630 import (
    check_experiment_complete,
    derive_merged_table_name,
)


class PackageFourYear10DNewaxisCandidateTest(unittest.TestCase):
    def test_check_experiment_complete_marks_missing_folds_incomplete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            experiment_dir = Path(tmpdir)
            (experiment_dir / "fold_predictions").mkdir(parents=True, exist_ok=True)

            with (experiment_dir / "fold_plan.csv").open("w", newline="", encoding="utf-8-sig") as file:
                writer = csv.DictWriter(file, fieldnames=["fold"])
                writer.writeheader()
                writer.writerow({"fold": 1})
                writer.writerow({"fold": 2})

            with (experiment_dir / "fold_results.csv").open("w", newline="", encoding="utf-8-sig") as file:
                writer = csv.DictWriter(file, fieldnames=["fold", "fold_table", "status"])
                writer.writeheader()
                writer.writerow(
                    {
                        "fold": 1,
                        "fold_table": "stock_predict_data_model_agent_demo__fold01",
                        "status": "ok",
                    }
                )

            (experiment_dir / "fold_predictions" / "fold01.parquet").write_bytes(b"ok")

            status = check_experiment_complete(experiment_dir)

            self.assertFalse(status["is_complete"])
            self.assertEqual(status["missing_folds"], [2])
            self.assertEqual(status["bad_status_folds"], [])
            self.assertEqual(len(status["missing_prediction_files"]), 1)
            self.assertTrue(status["missing_prediction_files"][0].endswith("fold02.parquet"))

    def test_check_experiment_complete_accepts_skipped_existing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            experiment_dir = Path(tmpdir)
            (experiment_dir / "fold_predictions").mkdir(parents=True, exist_ok=True)

            with (experiment_dir / "fold_plan.csv").open("w", newline="", encoding="utf-8-sig") as file:
                writer = csv.DictWriter(file, fieldnames=["fold"])
                writer.writeheader()
                writer.writerow({"fold": 1})
                writer.writerow({"fold": 2})

            with (experiment_dir / "fold_results.csv").open("w", newline="", encoding="utf-8-sig") as file:
                writer = csv.DictWriter(file, fieldnames=["fold", "fold_table", "status"])
                writer.writeheader()
                writer.writerow(
                    {
                        "fold": 1,
                        "fold_table": "stock_predict_data_model_agent_demo__fold01",
                        "status": "ok",
                    }
                )
                writer.writerow(
                    {
                        "fold": 2,
                        "fold_table": "stock_predict_data_model_agent_demo__fold02",
                        "status": "skipped_existing",
                    }
                )

            (experiment_dir / "fold_predictions" / "fold01.parquet").write_bytes(b"ok")
            (experiment_dir / "fold_predictions" / "fold02.parquet").write_bytes(b"ok")

            status = check_experiment_complete(experiment_dir)

            self.assertTrue(status["is_complete"])
            self.assertEqual(status["missing_folds"], [])
            self.assertEqual(status["bad_status_folds"], [])
            self.assertEqual(status["missing_prediction_files"], [])

    def test_derive_merged_table_name_strips_fold_suffix(self):
        rows = [
            {
                "fold": "1",
                "fold_table": "stock_predict_data_model_agent_four_year_demo_executable_10d_open_return_research__fold01",
            }
        ]
        table = derive_merged_table_name(rows)
        self.assertEqual(
            table,
            "stock_predict_data_model_agent_four_year_demo_executable_10d_open_return_research",
        )


if __name__ == "__main__":
    unittest.main()
