import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from validate_research_run_artifacts import validate_research_run


class ValidateResearchRunArtifactsTest(unittest.TestCase):
    def _write_valid_run(self, root: Path, *, folds=(1, 2)) -> None:
        (root / "fold_predictions").mkdir(parents=True)
        (root / "fold_logs").mkdir()
        (root / "fold_summaries").mkdir()
        (root / "models").mkdir()
        feature_dir = root / "feature_scores" / "exp"
        feature_dir.mkdir(parents=True)
        for fold in folds:
            suffix = f"{fold:02d}"
            (root / "fold_predictions" / f"fold{suffix}.parquet").write_bytes(b"parquet")
            (root / "fold_logs" / f"fold{suffix}.log").write_text("ok\n", encoding="utf-8")
            (root / "fold_summaries" / f"fold{suffix}.csv").write_text("fold,status\n", encoding="utf-8")
            (root / "models" / f"model_fold{suffix}.json").write_text("{}", encoding="utf-8")
            (root / "models" / f"model_fold{suffix}_metadata.json").write_text(
                json.dumps({"fold": fold}),
                encoding="utf-8",
            )
            (feature_dir / f"selected_features_executable_3d_open_return_rolling_fold{fold}.json").write_text(
                json.dumps({"features": ["alpha001"]}),
                encoding="utf-8",
            )
            (feature_dir / f"feature_ic_scores_executable_3d_open_return_rolling_fold{fold}.csv").write_text(
                "feature,ic\nalpha001,0.01\n",
                encoding="utf-8",
            )
        (root / "merge_meta.json").write_text(json.dumps({"table": "pred_research"}), encoding="utf-8")
        (root / "prediction_manifest.json").write_text(
            json.dumps(
                {
                    "approval_status": "research_only_not_approved_for_l4_or_l5",
                    "asset_role": "l4_research_prediction_asset",
                    "prediction_table": "stock_predict_data_v12_3d_research",
                }
            ),
            encoding="utf-8",
        )
        (root / "fold_results.csv").write_text("fold,status\n1,ok\n2,ok\n", encoding="utf-8")
        (root / "evaluation_summary.json").write_text("{}", encoding="utf-8")
        (root / "evaluation_daily.csv").write_text("trade_date,rank_ic\n", encoding="utf-8")

    def test_complete_research_run_passes_required_artifact_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self._write_valid_run(run_dir)

            result = validate_research_run(
                run_dir,
                label="executable_3d_open_return",
                expected_folds=[1, 2],
            )

            self.assertTrue(result["ok"])
            self.assertEqual(result["errors"], [])
            self.assertEqual(result["folds_checked"], [1, 2])

    def test_missing_model_metadata_and_formal_manifest_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self._write_valid_run(run_dir, folds=(1,))
            (run_dir / "models" / "model_fold01_metadata.json").unlink()
            (run_dir / "prediction_manifest.json").write_text(
                json.dumps(
                    {
                        "approval_status": "approved_for_l5",
                        "prediction_table": "stock_predict_data_formal",
                    }
                ),
                encoding="utf-8",
            )

            result = validate_research_run(
                run_dir,
                label="executable_3d_open_return",
                expected_folds=[1],
            )

            self.assertFalse(result["ok"])
            self.assertIn("missing models/model_fold01_metadata.json", result["errors"])
            self.assertIn("prediction_manifest approval_status must be research-only", result["errors"])
            self.assertIn("prediction table must contain _research", result["errors"])

    def test_failed_fold_result_status_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self._write_valid_run(run_dir, folds=(1, 2))
            (run_dir / "fold_results.csv").write_text(
                "fold,status\n1,ok\n2,failed\n",
                encoding="utf-8",
            )

            result = validate_research_run(
                run_dir,
                label="executable_3d_open_return",
                expected_folds=[1, 2],
            )

            self.assertFalse(result["ok"])
            self.assertIn("fold_results contains failed fold 2", result["errors"])

    def test_merged_prediction_table_rejects_null_scores_and_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            self._write_valid_run(run_dir, folds=(1,))
            db_path = Path(tmp) / "MODEL_PREDICTIONS.db"
            table = "stock_predict_data_v12_3d_research"
            conn = sqlite3.connect(db_path)
            try:
                conn.execute(
                    f"create table {table} (trade_date text, stock_code text, pred_prob real)"
                )
                conn.executemany(
                    f"insert into {table} values (?, ?, ?)",
                    [
                        ("20240604", "000001.SZ", 0.1),
                        ("20240604", "000001.SZ", 0.2),
                        ("20240605", "000002.SZ", None),
                    ],
                )
                conn.commit()
            finally:
                conn.close()

            result = validate_research_run(
                run_dir,
                label="executable_3d_open_return",
                expected_folds=[1],
                prediction_db_path=db_path,
                prediction_table=table,
            )

            self.assertFalse(result["ok"])
            self.assertIn("prediction table null pred_prob count must be 0", result["errors"])
            self.assertIn("prediction table duplicate key groups must be 0", result["errors"])


if __name__ == "__main__":
    unittest.main()
