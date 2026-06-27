import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from validate_formula_score_asset import validate_formula_score_asset


class ValidateFormulaScoreAssetTest(unittest.TestCase):
    def _write_prediction_db(self, db_path: Path, table: str, rows=None) -> None:
        rows = rows or [
            ("20240604", "000001.SZ", 0.1),
            ("20240604", "000002.SZ", 0.2),
        ]
        conn = sqlite3.connect(db_path)
        try:
            conn.execute(f"create table {table} (trade_date text, stock_code text, pred_prob real)")
            conn.executemany(f"insert into {table} values (?, ?, ?)", rows)
            conn.commit()
        finally:
            conn.close()

    def _valid_report(self, table: str) -> dict:
        return {
            "asset_role": "research_candidate_not_formal",
            "approval_status": "research_only_not_approved_for_l4_or_l5",
            "label": "executable_10d_open_return",
            "target_table": table,
            "sources": {
                "formal": "stock_predict_data_10d_formal",
                "base_stable": "stock_predict_data_10d_base_research",
                "topn_source": "stock_predict_data_10d_topn_research",
            },
            "best_params": {
                "start_date": "20260401",
                "alpha": 1.0,
                "objective": 0.69,
            },
            "target_delta_vs_formal": {
                "full": {
                    "delta_rank_ic": 0.00001,
                    "delta_top5": 0.004,
                    "delta_top10": 0.001,
                },
                "recent63": {
                    "delta_top5": 0.03,
                    "delta_top10": 0.009,
                },
            },
            "governance": {
                "no_training": True,
                "no_production_manifest_change": True,
                "no_signal": True,
                "no_backtest": True,
            },
        }

    def test_valid_formula_score_asset_passes_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            table = "stock_predict_data_10d_blend_research"
            db_path = root / "MODEL_PREDICTIONS.db"
            report_path = root / "report.json"
            self._write_prediction_db(db_path, table)
            report_path.write_text(json.dumps(self._valid_report(table)), encoding="utf-8")

            result = validate_formula_score_asset(report_path, prediction_db_path=db_path)

            self.assertTrue(result["ok"])
            self.assertEqual(result["errors"], [])
            self.assertEqual(result["prediction_table_summary"]["row_count"], 2)

    def test_formal_or_nonresearch_formula_asset_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            table = "stock_predict_data_10d_blend"
            db_path = root / "MODEL_PREDICTIONS.db"
            report_path = root / "report.json"
            self._write_prediction_db(db_path, table)
            report = self._valid_report(table)
            report["approval_status"] = "approved_for_l5"
            report["asset_role"] = "l4_formal_prediction_asset"
            report_path.write_text(json.dumps(report), encoding="utf-8")

            result = validate_formula_score_asset(report_path, prediction_db_path=db_path)

            self.assertFalse(result["ok"])
            self.assertIn("approval_status must be research-only", result["errors"])
            self.assertIn("target_table must contain _research", result["errors"])
            self.assertIn("asset_role must not be formal", result["errors"])

    def test_dirty_prediction_table_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            table = "stock_predict_data_10d_blend_research"
            db_path = root / "MODEL_PREDICTIONS.db"
            report_path = root / "report.json"
            self._write_prediction_db(
                db_path,
                table,
                rows=[
                    ("20240604", "000001.SZ", 0.1),
                    ("20240604", "000001.SZ", 0.2),
                    ("20240605", "000002.SZ", None),
                ],
            )
            report_path.write_text(json.dumps(self._valid_report(table)), encoding="utf-8")

            result = validate_formula_score_asset(report_path, prediction_db_path=db_path)

            self.assertFalse(result["ok"])
            self.assertIn("prediction table null pred_prob count must be 0", result["errors"])
            self.assertIn("prediction table duplicate key groups must be 0", result["errors"])

    def test_metric_floor_failure_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            table = "stock_predict_data_10d_blend_research"
            db_path = root / "MODEL_PREDICTIONS.db"
            report_path = root / "report.json"
            self._write_prediction_db(db_path, table)
            report = self._valid_report(table)
            report["target_delta_vs_formal"]["full"]["delta_top5"] = -0.001
            report_path.write_text(json.dumps(report), encoding="utf-8")

            result = validate_formula_score_asset(report_path, prediction_db_path=db_path)

            self.assertFalse(result["ok"])
            self.assertIn("full delta_top5 below floor 0.0", result["errors"])


if __name__ == "__main__":
    unittest.main()
