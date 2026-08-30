import unittest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch

import duckdb
import pandas as pd

from fast_feature_selection import _read_parquet_date_range, score_features_fast_split
from ai_module import prepare_training_label
from feature_selection_module import (
    FeatureSelectionConfig,
    load_selection_frame,
    prepare_selection_label,
    select_features,
)


class FeatureSelectionModuleTests(unittest.TestCase):
    def test_read_parquet_date_range_ignores_transient_tmp_parts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            parts_dir = Path(temp_dir) / "production_factor_parts"
            parts_dir.mkdir()
            pd.DataFrame(
                [
                    {"trade_date": "20240102", "stock_code": "000001.SZ", "factor_a": 1.0},
                    {"trade_date": "20240103", "stock_code": "000002.SZ", "factor_a": 2.0},
                ]
            ).to_parquet(parts_dir / "production_factor_part_0000.parquet", index=False)
            (parts_dir / "production_factor_part_0001.parquet.tmp_20260624").write_bytes(b"partial")

            result = _read_parquet_date_range(
                parts_dir,
                ["trade_date", "stock_code", "factor_a"],
                "20240103",
                "20240103",
            )

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["stock_code"], "000002.SZ")

    def test_select_features_prefers_predictive_non_leaky_factor(self):
        frame = pd.DataFrame(
            {
                "trade_date": ["20260105"] * 4 + ["20260106"] * 4,
                "stock_code": list("ABCDEFGH"),
                "good_factor": [1, 2, 3, 4, 1, 2, 3, 4],
                "bad_factor": [4, 1, 4, 1, 4, 1, 4, 1],
                "post_open": [10, 10, 10, 10, 10, 10, 10, 10],
                "10d_yield_rate": [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4],
            }
        )

        selected = select_features(
            frame,
            FeatureSelectionConfig(label="10d_yield_rate", top_n=2, min_abs_ic=0.5),
        )

        self.assertIn("good_factor", selected)
        self.assertNotIn("post_open", selected)

    def test_prepare_training_label_builds_risk_adjusted_return(self):
        frame = pd.DataFrame(
            {
                "10d_yield_rate": [0.10, 0.05],
                "atr_qfq": [1.0, 0.5],
                "close_qfq": [20.0, 10.0],
            }
        )

        result = prepare_training_label(frame, "risk_adjusted_10d_yield_rate")

        self.assertAlmostEqual(result.loc[0, "risk_adjusted_10d_yield_rate"], 2.0)
        self.assertAlmostEqual(result.loc[1, "risk_adjusted_10d_yield_rate"], 1.0)

    def test_prepare_training_label_prefers_close_qfq_when_present(self):
        frame = pd.DataFrame(
            {
                "10d_yield_rate": [0.10],
                "atr_qfq": [1.0],
                "close": [20.0],
                "close_qfq": [10.0],
            }
        )

        result = prepare_selection_label(frame, "risk_adjusted_10d_yield_rate")

        self.assertAlmostEqual(result.loc[0, "risk_adjusted_10d_yield_rate"], 1.0)

    def test_prepare_selection_label_requires_explicit_close_qfq(self):
        frame = pd.DataFrame(
            {
                "10d_yield_rate": [0.10],
                "atr_qfq": [1.0],
                "close": [20.0],
            }
        )

        with self.assertRaisesRegex(KeyError, "requires explicit front-adjusted column: close_qfq"):
            prepare_selection_label(frame, "risk_adjusted_10d_yield_rate")

    def test_prepare_training_label_builds_executable_open_return(self):
        frame = pd.DataFrame({"post_open": [10.0], "post12_open": [11.0]})

        result = prepare_training_label(frame, "executable_10d_open_return")

        expected = (11.0 * (1 - 0.0003 - 0.0005 - 0.001)) / (10.0 * (1 + 0.0003 + 0.001)) - 1
        self.assertAlmostEqual(result.loc[0, "executable_10d_open_return"], expected)

    def test_prepare_training_label_builds_executable_5d_open_return(self):
        frame = pd.DataFrame({"post_open": [10.0], "post6_open": [10.8]})

        result = prepare_training_label(frame, "executable_5d_open_return")

        expected = (10.8 * (1 - 0.0003 - 0.0005 - 0.001)) / (10.0 * (1 + 0.0003 + 0.001)) - 1
        self.assertAlmostEqual(result.loc[0, "executable_5d_open_return"], expected)

    def test_prepare_training_label_builds_daily_top_quantile_label(self):
        frame = pd.DataFrame(
            {
                "trade_date": ["20260105"] * 10 + ["20260106"] * 10,
                "post_open": [10.0] * 20,
                "post6_open": list(range(10, 20)) + list(range(20, 10, -1)),
            }
        )

        result = prepare_training_label(frame, "executable_5d_open_top10")

        self.assertEqual(int(result["executable_5d_open_top10"].sum()), 2)
        self.assertEqual(int(result.loc[9, "executable_5d_open_top10"]), 1)
        self.assertEqual(int(result.loc[10, "executable_5d_open_top10"]), 1)

    def test_prepare_selection_label_builds_daily_top_quantile_label(self):
        frame = pd.DataFrame(
            {
                "trade_date": ["20260105"] * 10,
                "post_open": [10.0] * 10,
                "post6_open": list(range(10, 20)),
            }
        )

        result = prepare_selection_label(frame, "executable_5d_open_top10")

        self.assertEqual(int(result["executable_5d_open_top10"].sum()), 1)

    def test_prepare_selection_label_builds_executable_5d_open_return(self):
        frame = pd.DataFrame({"post_open": [10.0], "post6_open": [10.8]})

        result = prepare_selection_label(frame, "executable_5d_open_return")

        expected = (10.8 * (1 - 0.0003 - 0.0005 - 0.001)) / (10.0 * (1 + 0.0003 + 0.001)) - 1
        self.assertAlmostEqual(result.loc[0, "executable_5d_open_return"], expected)

    def test_prepare_training_label_builds_executable_3d_open_return(self):
        frame = pd.DataFrame({"post_open": [10.0], "post4_open": [10.6]})

        result = prepare_training_label(frame, "executable_3d_open_return")

        expected = (10.6 * (1 - 0.0003 - 0.0005 - 0.001)) / (10.0 * (1 + 0.0003 + 0.001)) - 1
        self.assertAlmostEqual(result.loc[0, "executable_3d_open_return"], expected)

    def test_prepare_selection_label_builds_executable_3d_open_return(self):
        frame = pd.DataFrame({"post_open": [10.0], "post4_open": [10.6]})

        result = prepare_selection_label(frame, "executable_3d_open_return")

        expected = (10.6 * (1 - 0.0003 - 0.0005 - 0.001)) / (10.0 * (1 + 0.0003 + 0.001)) - 1
        self.assertAlmostEqual(result.loc[0, "executable_3d_open_return"], expected)

    def test_prepare_training_label_builds_executable_1d_open_return(self):
        frame = pd.DataFrame({"post_open": [10.0], "post2_open": [10.3]})

        result = prepare_training_label(frame, "executable_1d_open_return")

        expected = (10.3 * (1 - 0.0003 - 0.0005 - 0.001)) / (10.0 * (1 + 0.0003 + 0.001)) - 1
        self.assertAlmostEqual(result.loc[0, "executable_1d_open_return"], expected)

    def test_prepare_selection_label_builds_executable_1d_open_return(self):
        frame = pd.DataFrame({"post_open": [10.0], "post2_open": [10.3]})

        result = prepare_selection_label(frame, "executable_1d_open_return")

        expected = (10.3 * (1 - 0.0003 - 0.0005 - 0.001)) / (10.0 * (1 + 0.0003 + 0.001)) - 1
        self.assertAlmostEqual(result.loc[0, "executable_1d_open_return"], expected)

    def test_load_selection_frame_reads_duckdb_mainline_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data_file"
            registry_dir = data_dir / "asset_registry"
            duckdb_dir = data_dir / "production_assets" / "duckdb"
            registry_dir.mkdir(parents=True)
            duckdb_dir.mkdir(parents=True)
            duckdb_path = duckdb_dir / "quant_production.duckdb"
            with duckdb.connect(str(duckdb_path)) as conn:
                conn.execute(
                    """
                    CREATE TABLE prod_l3_feature_table(
                        trade_date TEXT,
                        stock_code TEXT,
                        name TEXT,
                        industry TEXT,
                        act_ent_type TEXT,
                        factor_a DOUBLE
                    )
                    """
                )
                conn.execute(
                    "INSERT INTO prod_l3_feature_table VALUES ('20240102', '000001.SZ', 'A', 'I1', 'SOE', 1.5)"
                )
                conn.execute(
                    """
                    CREATE TABLE prod_l3_label_table(
                        trade_date TEXT,
                        stock_code TEXT,
                        "5d_yield_rate" DOUBLE,
                        "open6_yield_rate" DOUBLE,
                        "10d_yield_rate" DOUBLE
                    )
                    """
                )
                conn.execute(
                    "INSERT INTO prod_l3_label_table VALUES ('20240102', '000001.SZ', 0.1, 0.2, 0.3)"
                )
            (registry_dir / "production_assets.json").write_text(
                f"""
                {{
                  "assets": [
                    {{
                      "asset_id": "prod_l3_feature_duckdb",
                      "layer": "L3_features",
                      "asset_type": "duckdb_table",
                      "status": "production_active",
                      "allowed_for_main_workflow": true,
                      "asset_path": "{duckdb_path.as_posix()}::prod_l3_feature_table"
                    }},
                    {{
                      "asset_id": "prod_l3_label_duckdb",
                      "layer": "L3_labels",
                      "asset_type": "duckdb_table",
                      "status": "production_active",
                      "allowed_for_main_workflow": true,
                      "asset_path": "{duckdb_path.as_posix()}::prod_l3_label_table"
                    }}
                  ]
                }}
                """,
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                {
                    "QUANT_MODEL_FEATURE_DUCKDB": str(duckdb_path),
                    "QUANT_MODEL_LABEL_DUCKDB": str(duckdb_path),
                    "QUANT_MODEL_FEATURE_DUCKDB_TABLE": "prod_l3_feature_table",
                    "QUANT_MODEL_LABEL_DUCKDB_TABLE": "prod_l3_label_table",
                },
                clear=False,
            ):
                frame = load_selection_frame(
                    "",
                    label_path="",
                    label="10d_yield_rate",
                    feature_source="production_split",
                )

        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0]["stock_code"], "000001.SZ")
        self.assertIn("10d_yield_rate", frame.columns)

    def test_score_features_fast_split_reads_duckdb_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            duckdb_path = root / "quant_production.duckdb"
            feature_rows = []
            label_rows = []
            for day in ("20240102", "20240103"):
                for idx in range(120):
                    stock_code = f"{idx:06d}.SZ"
                    feature_rows.append(
                        f"('{day}', '{stock_code}', {idx + 1}, {120 - idx})"
                    )
                    label_rows.append(
                        f"('{day}', '{stock_code}', {(idx + 1) / 1000.0})"
                    )
            with duckdb.connect(str(duckdb_path)) as conn:
                conn.execute(
                    """
                    CREATE TABLE feature_table(
                        trade_date TEXT,
                        stock_code TEXT,
                        factor_a DOUBLE,
                        factor_b DOUBLE
                    )
                    """
                )
                conn.execute(
                    "INSERT INTO feature_table VALUES " + ", ".join(feature_rows)
                )
                conn.execute(
                    """
                    CREATE TABLE label_table(
                        trade_date TEXT,
                        stock_code TEXT,
                        "10d_yield_rate" DOUBLE
                    )
                    """
                )
                conn.execute(
                    "INSERT INTO label_table VALUES " + ", ".join(label_rows)
                )

            rows, selected = score_features_fast_split(
                duckdb_path,
                label_path=duckdb_path,
                feature_table="feature_table",
                label_table="label_table",
                label="10d_yield_rate",
                start="20240102",
                end="20240103",
                top_n=2,
                min_abs_ic=0.5,
                max_missing_ratio=0.35,
                folds=2,
            )

        self.assertTrue(rows)
        self.assertIn("factor_a", selected)


if __name__ == "__main__":
    unittest.main()
