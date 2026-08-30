import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from build_standard_chain_score_asset import _manifest_payload, read_registered_factor_dates


class BuildStandardChainScoreAssetTests(unittest.TestCase):
    def test_manifest_payload_uses_actual_duckdb_output_path_instead_of_bundle_path(self):
        frame = pd.DataFrame(
            [
                {"trade_date": "20240624", "stock_code": "000001.SZ"},
                {"trade_date": "20240625", "stock_code": "000002.SZ"},
            ]
        )
        model_metadata = {
            "model_path": "D:/models/model.json",
            "metadata_path": "D:/models/model_meta.json",
            "feature_columns": ["feat_a", "feat_b"],
        }
        manifest_dir = Path("D:/workspace/runs/candidate_x")
        output_db_path = Path("D:/workspace/data_file/production_assets/duckdb/l4_candidate.duckdb")
        market_db_path = Path("D:/workspace/data_file/production_assets/duckdb/l2_stock_daily_data.duckdb")

        with patch(
            "build_standard_chain_score_asset.resolve_stock_daily_duckdb_path",
            return_value=market_db_path,
        ):
            payload = _manifest_payload(
                label="executable_5d_open_return",
                candidate_id="candidate_x",
                output_table="score_table",
                base_table="base_table",
                frame=frame,
                model_metadata=model_metadata,
                incremental_dates=["20240625"],
                generated_at="2026-07-01T12:00:00+08:00",
                output_db_path=output_db_path,
                manifest_dir=manifest_dir,
            )

        self.assertEqual(payload["source_type"], "duckdb_table")
        self.assertEqual(payload["db_path"], "../../data_file/production_assets/duckdb/l4_candidate.duckdb")
        self.assertEqual(payload["market_db_path"], "../../data_file/production_assets/duckdb/l2_stock_daily_data.duckdb")
        self.assertNotIn("quant_production.duckdb", json.dumps(payload, ensure_ascii=False))

    def test_read_registered_factor_dates_rejects_missing_duckdb_route(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data_file"
            with self.assertRaisesRegex(RuntimeError, "DuckDB feature table is required"):
                read_registered_factor_dates(
                    data_dir,
                    ["20240625"],
                    ["trade_date", "stock_code", "feat_a"],
                )

    @unittest.skipIf(importlib.util.find_spec("duckdb") is None, "duckdb is not installed")
    def test_read_registered_factor_dates_uses_duckdb_when_registry_points_to_duckdb(self):
        import duckdb

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data_file"
            registry_dir = data_dir / "asset_registry"
            duckdb_dir = data_dir / "production_assets" / "duckdb"
            registry_dir.mkdir(parents=True)
            duckdb_dir.mkdir(parents=True)
            duckdb_path = duckdb_dir / "l3_feature_current.duckdb"
            with duckdb.connect(str(duckdb_path)) as conn:
                conn.execute(
                    """
                    CREATE TABLE prod_l3_feature_table (
                        trade_date VARCHAR,
                        stock_code VARCHAR,
                        feat_a DOUBLE
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO prod_l3_feature_table VALUES
                    ('20240624', '000001.SZ', 1.0),
                    ('20240625', '000001.SZ', 2.0)
                    """
                )
            (registry_dir / "production_assets.json").write_text(
                json.dumps(
                    {
                        "assets": [
                            {
                                "asset_id": "prod_l3_feature_duckdb",
                                "layer": "L3_features",
                                "asset_type": "duckdb_table",
                                "status": "production_active",
                                "allowed_for_main_workflow": True,
                                "asset_path": f"{duckdb_path.as_posix()}::prod_l3_feature_table",
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with patch.dict("os.environ", {"QUANT_DATA_DIR": str(data_dir)}):
                frame = read_registered_factor_dates(
                    data_dir,
                    ["20240625"],
                    ["trade_date", "stock_code", "feat_a"],
                )

        self.assertEqual(frame.shape[0], 1)
        self.assertEqual(frame.iloc[0]["trade_date"], "20240625")
        self.assertEqual(frame.iloc[0]["feat_a"], 2.0)


if __name__ == "__main__":
    unittest.main()
