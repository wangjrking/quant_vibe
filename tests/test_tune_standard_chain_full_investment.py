import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tune_standard_chain_full_investment import main, resolve_fusion_assets, resolve_standard_chain_assets


class TuneStandardChainFullInvestmentTests(unittest.TestCase):
    def test_resolve_fusion_assets_prefers_latest_fusion_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reports_dir = Path(tmpdir)
            old_dir = reports_dir / "strategy_agent_model_application_20260618" / "published_asset_fusions"
            new_dir = reports_dir / "strategy_agent_model_application_20260619" / "published_asset_fusions_0618_refresh"
            old_dir.mkdir(parents=True)
            new_dir.mkdir(parents=True)
            self._write_fusion_manifest(
                old_dir / "fusion_manifest.json",
                output_db=old_dir / "fusion_combos.db",
                max_trade_date="20260612",
            )
            self._write_fusion_manifest(
                new_dir / "fusion_manifest.json",
                output_db=new_dir / "fusion_combos.db",
                max_trade_date="20260616",
            )

            assets = resolve_fusion_assets(reports_dir=reports_dir)

        self.assertEqual(
            assets,
            [
                {
                    "asset": "fusion_mincons",
                    "db_path": new_dir / "fusion_combos.db",
                    "table": "combo_rank_min_consensus",
                    "max_trade_date": "20260616",
                    "manifest_path": new_dir / "fusion_manifest.json",
                },
                {
                    "asset": "fusion_10d60_5d30_3d10",
                    "db_path": new_dir / "fusion_combos.db",
                    "table": "combo_rank_10d60_5d30_3d10",
                    "max_trade_date": "20260616",
                    "manifest_path": new_dir / "fusion_manifest.json",
                },
                {
                    "asset": "fusion_10d80_5d20",
                    "db_path": new_dir / "fusion_combos.db",
                    "table": "combo_rank_10d80_5d20",
                    "max_trade_date": "20260616",
                    "manifest_path": new_dir / "fusion_manifest.json",
                },
                {
                    "asset": "fusion_max_any",
                    "db_path": new_dir / "fusion_combos.db",
                    "table": "combo_rank_max_any",
                    "max_trade_date": "20260616",
                    "manifest_path": new_dir / "fusion_manifest.json",
                },
            ],
        )

    def test_resolve_fusion_assets_rejects_stale_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reports_dir = Path(tmpdir)
            new_dir = reports_dir / "strategy_agent_model_application_20260619" / "published_asset_fusions_0618_refresh"
            new_dir.mkdir(parents=True)
            self._write_fusion_manifest(
                new_dir / "fusion_manifest.json",
                output_db=new_dir / "fusion_combos.db",
                max_trade_date="20260616",
            )

            with self.assertRaisesRegex(RuntimeError, "stale fusion manifest"):
                resolve_fusion_assets(reports_dir=reports_dir, min_trade_date="20260618")

    def test_resolve_fusion_assets_can_find_nested_research_run_fusion_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reports_dir = Path(tmpdir)
            old_dir = reports_dir / "strategy_agent_model_application_20260619" / "published_asset_fusions_0618_refresh"
            nested_dir = reports_dir / "strategy_agent_model_application_20260620" / "latest_standard_chain_research" / "fusion"
            old_dir.mkdir(parents=True)
            nested_dir.mkdir(parents=True)
            self._write_fusion_manifest(
                old_dir / "fusion_manifest.json",
                output_db=old_dir / "fusion_combos.db",
                max_trade_date="20260616",
            )
            self._write_fusion_manifest(
                nested_dir / "fusion_manifest.json",
                output_db=nested_dir / "fusion_combos.db",
                max_trade_date="20260618",
            )

            assets = resolve_fusion_assets(reports_dir=reports_dir, min_trade_date="20260618")

        self.assertTrue(all(row["db_path"] == nested_dir / "fusion_combos.db" for row in assets))
        self.assertTrue(all(row["max_trade_date"] == "20260618" for row in assets))

    def test_resolve_standard_chain_assets_prefers_latest_standard_chain_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_dir = Path(tmpdir)

            self._write_manifest(
                manifest_dir / "prod_liq_prime_one_v20260612_l4_formal.json",
                label="executable_5d_open_return",
                approval_status="approved_for_l5",
                table="stock_predict_data_model_agent_standard_chain_tune_20260617_executable_5d_open_return_v2",
                generated_at="2026-06-18T00:15:11+08:00",
                max_trade_date="20260612",
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
                manifest_dir / "prod_liq_prime_one_v20260612_l4_legacy_archive.json",
                label="executable_5d_open_return",
                approval_status="rollback_archive_only",
                table="stock_predict_data_executable_5d_open_return_prod_aligned_202206_202606",
                generated_at="2026-06-18T00:15:11+08:00",
                max_trade_date="20260612",
            )
            self._write_manifest(
                manifest_dir / "executable_10d_open_return_l4_formal_20260617.json",
                label="executable_10d_open_return",
                approval_status="approved_for_l4_only",
                table="stock_predict_data_model_agent_standard_chain_tune_20260617_executable_10d_open_return",
                generated_at="2026-06-18T00:15:11+08:00",
                max_trade_date="20260612",
            )

            assets = resolve_standard_chain_assets(manifest_dir=manifest_dir)

        self.assertEqual(
            assets[-2:],
            [
                {
                    "asset": "std_10d",
                    "db_path": manifest_dir / "MODEL_PREDICTIONS.db",
                    "table": "stock_predict_data_model_agent_standard_chain_tune_20260617_executable_10d_open_return",
                    "approval_status": "approved_for_l4_only",
                    "max_trade_date": "20260612",
                    "manifest_path": manifest_dir / "executable_10d_open_return_l4_formal_20260617.json",
                },
                {
                    "asset": "std_5d",
                    "db_path": manifest_dir / "MODEL_PREDICTIONS.db",
                    "table": "stock_predict_data_model_agent_standard_chain_tune_20260620_executable_5d_open_return_v3",
                    "approval_status": "approved_for_l5",
                    "max_trade_date": "20260618",
                    "manifest_path": manifest_dir / "prod_liq_prime_one_v20260620_l4_formal.json",
                },
            ],
        )

    def test_resolve_standard_chain_assets_accepts_latest_formal_tables_without_standard_chain_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_dir = Path(tmpdir)
            self._write_manifest(
                manifest_dir / "prod_liq_prime_one_v20260620_l4_formal.json",
                label="executable_5d_open_return",
                approval_status="approved_for_l5",
                table="stock_predict_data_model_agent_toprank_latestfactor_20260618_executable_5d_open_return_score_20240604_20260618",
                generated_at="2026-06-20T09:30:00+08:00",
                max_trade_date="20260618",
            )
            self._write_manifest(
                manifest_dir / "executable_10d_open_return_l4_formal_20260617.json",
                label="executable_10d_open_return",
                approval_status="approved_for_l5",
                table="stock_predict_data_model_agent_toprank_latestfactor_20260618_executable_10d_open_return_score_20240604_20260618",
                generated_at="2026-06-20T09:35:00+08:00",
                max_trade_date="20260618",
            )

            assets = resolve_standard_chain_assets(manifest_dir=manifest_dir, min_trade_date="20260618")

        self.assertEqual(
            [row["table"] for row in assets],
            [
                "stock_predict_data_model_agent_toprank_latestfactor_20260618_executable_10d_open_return_score_20240604_20260618",
                "stock_predict_data_model_agent_toprank_latestfactor_20260618_executable_5d_open_return_score_20240604_20260618",
            ],
        )

    def test_resolve_standard_chain_assets_requires_5d_and_10d_manifests(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_dir = Path(tmpdir)
            self._write_manifest(
                manifest_dir / "prod_liq_prime_one_v20260620_l4_formal.json",
                label="executable_5d_open_return",
                approval_status="approved_for_l5",
                table="stock_predict_data_model_agent_standard_chain_tune_20260620_executable_5d_open_return_v3",
                generated_at="2026-06-20T09:30:00+08:00",
                max_trade_date="20260618",
            )

            with self.assertRaisesRegex(RuntimeError, "missing standard-chain manifest"):
                resolve_standard_chain_assets(manifest_dir=manifest_dir)

    def test_resolve_standard_chain_assets_rejects_stale_assets_when_min_trade_date_required(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_dir = Path(tmpdir)
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

            with self.assertRaisesRegex(RuntimeError, "stale standard-chain manifest"):
                resolve_standard_chain_assets(manifest_dir=manifest_dir, min_trade_date="20260618")

    def test_main_rejects_stale_standard_chain_assets_via_cli_guard(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_dir = Path(tmpdir) / "manifests"
            reports_dir = Path(tmpdir) / "reports"
            fusion_dir = reports_dir / "strategy_agent_model_application_20260620" / "published_asset_fusions_0618_refresh"
            report_dir = Path(tmpdir) / "report"
            manifest_dir.mkdir(parents=True)
            fusion_dir.mkdir(parents=True)
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
            self._write_fusion_manifest(
                fusion_dir / "fusion_manifest.json",
                output_db=fusion_dir / "fusion_combos.db",
                max_trade_date="20260618",
            )

            with patch("tune_standard_chain_full_investment.MANIFEST_DIR", manifest_dir), patch(
                "tune_standard_chain_full_investment.REPORTS_DIR", reports_dir
            ):
                with self.assertRaisesRegex(RuntimeError, "stale standard-chain manifest"):
                    main(
                        [
                            "--limit",
                            "0",
                            "--report-dir",
                            str(report_dir),
                            "--min-standard-trade-date",
                            "20260618",
                        ]
                    )

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

    @staticmethod
    def _write_fusion_manifest(path: Path, *, output_db: Path, max_trade_date: str) -> None:
        path.write_text(
            json.dumps(
                {
                    "output_db": str(output_db),
                    "combos": [
                        {
                            "combo_name": "combo_rank_min_consensus",
                            "table": "combo_rank_min_consensus",
                            "max_trade_date": max_trade_date,
                        },
                        {
                            "combo_name": "combo_rank_10d60_5d30_3d10",
                            "table": "combo_rank_10d60_5d30_3d10",
                            "max_trade_date": max_trade_date,
                        },
                        {
                            "combo_name": "combo_rank_10d80_5d20",
                            "table": "combo_rank_10d80_5d20",
                            "max_trade_date": max_trade_date,
                        },
                        {
                            "combo_name": "combo_rank_max_any",
                            "table": "combo_rank_max_any",
                            "max_trade_date": max_trade_date,
                        },
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
