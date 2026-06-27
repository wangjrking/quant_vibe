import json
import tempfile
import unittest
from pathlib import Path

from build_research_fusion_library import resolve_prediction_inputs


class BuildResearchFusionLibraryTests(unittest.TestCase):
    def test_resolve_prediction_inputs_prefers_latest_manifest_per_label(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_dir = Path(tmpdir)
            self._write_manifest(
                manifest_dir / "executable_3d_open_return_l4_formal_20260617.json",
                label="executable_3d_open_return",
                approval_status="approved_for_l5",
                table="stock_predict_data_model_agent_toprank_latestfactor_20260618_executable_3d_open_return_score_20240604_20260618",
                generated_at="2026-06-19T19:18:03+08:00",
                max_trade_date="20260618",
            )
            self._write_manifest(
                manifest_dir / "prod_liq_prime_one_v20260620_l4_formal.json",
                label="executable_5d_open_return",
                approval_status="approved_for_l5",
                table="stock_predict_data_model_agent_standard_chain_tune_20260620_executable_5d_open_return_v3",
                generated_at="2026-06-20T09:30:00+08:00",
                max_trade_date="20260618",
            )
            self._write_manifest(
                manifest_dir / "executable_10d_open_return_l4_formal_20260617.json",
                label="executable_10d_open_return",
                approval_status="approved_for_l4_only",
                table="stock_predict_data_model_agent_standard_chain_tune_20260617_executable_10d_open_return",
                generated_at="2026-06-18T00:15:11+08:00",
                max_trade_date="20260618",
            )

            resolved = resolve_prediction_inputs(manifest_dir=manifest_dir, min_trade_date="20260618")

        self.assertEqual(
            resolved,
            {
                "model_db": manifest_dir / "MODEL_PREDICTIONS.db",
                "table_3d": "stock_predict_data_model_agent_toprank_latestfactor_20260618_executable_3d_open_return_score_20240604_20260618",
                "table_5d": "stock_predict_data_model_agent_standard_chain_tune_20260620_executable_5d_open_return_v3",
                "table_10d": "stock_predict_data_model_agent_standard_chain_tune_20260617_executable_10d_open_return",
                "manifest_3d": manifest_dir / "executable_3d_open_return_l4_formal_20260617.json",
                "manifest_5d": manifest_dir / "prod_liq_prime_one_v20260620_l4_formal.json",
                "manifest_10d": manifest_dir / "executable_10d_open_return_l4_formal_20260617.json",
                "approval_status_3d": "approved_for_l5",
                "approval_status_5d": "approved_for_l5",
                "approval_status_10d": "approved_for_l4_only",
                "max_trade_date_3d": "20260618",
                "max_trade_date_5d": "20260618",
                "max_trade_date_10d": "20260618",
            },
        )

    def test_resolve_prediction_inputs_rejects_stale_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_dir = Path(tmpdir)
            self._write_manifest(
                manifest_dir / "executable_3d_open_return_l4_formal_20260617.json",
                label="executable_3d_open_return",
                approval_status="approved_for_l5",
                table="stock_predict_data_model_agent_toprank_latestfactor_20260618_executable_3d_open_return_score_20240604_20260618",
                generated_at="2026-06-19T19:18:03+08:00",
                max_trade_date="20260618",
            )
            self._write_manifest(
                manifest_dir / "prod_liq_prime_one_v20260620_l4_formal.json",
                label="executable_5d_open_return",
                approval_status="approved_for_l5",
                table="stock_predict_data_model_agent_standard_chain_tune_20260620_executable_5d_open_return_v3",
                generated_at="2026-06-20T09:30:00+08:00",
                max_trade_date="20260618",
            )
            self._write_manifest(
                manifest_dir / "executable_10d_open_return_l4_formal_20260617.json",
                label="executable_10d_open_return",
                approval_status="approved_for_l4_only",
                table="stock_predict_data_model_agent_standard_chain_tune_20260617_executable_10d_open_return",
                generated_at="2026-06-18T00:15:11+08:00",
                max_trade_date="20260612",
            )

            with self.assertRaisesRegex(RuntimeError, "stale prediction manifest"):
                resolve_prediction_inputs(manifest_dir=manifest_dir, min_trade_date="20260618")

    @staticmethod
    def _write_manifest(
        path: Path,
        *,
        label: str,
        approval_status: str,
        table: str,
        generated_at: str,
        max_trade_date: str,
    ) -> None:
        path.write_text(
            json.dumps(
                {
                    "label": label,
                    "approval_status": approval_status,
                    "source_type": "sqlite_table",
                    "db_path": "MODEL_PREDICTIONS.db",
                    "table": table,
                    "generated_at": generated_at,
                    "max_trade_date": max_trade_date,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
