import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import refresh_l3_active_duckdb_full_delivery as refresh_l3


class RefreshL3ActiveDuckdbFullDeliveryTests(unittest.TestCase):
    def test_build_target_feature_frame_uses_current_raw_target_qfq_contract(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_parts_dir = root / "raw_factor_by_stock_parts"
            raw_parts_dir.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(
                [
                    {
                        "trade_date": "20260701",
                        "stock_code": "000001.SZ",
                        "industry": "Bank",
                        "act_ent_type": "A",
                        "open": 10.0,
                        "high": 10.5,
                        "low": 9.8,
                        "close": 10.2,
                        "pre_close": 9.9,
                        "amount": 1000.0,
                        "vol": 200.0,
                    }
                ]
            ).to_parquet(raw_parts_dir / "raw_part_0000.parquet", index=False)

            raw_target = pd.DataFrame(
                [
                    {
                        "trade_date": "20260701",
                        "stock_code": "000001.SZ",
                        "industry": "Bank",
                        "act_ent_type": "A",
                        "open": 10.0,
                        "high": 10.5,
                        "low": 9.8,
                        "close": 10.2,
                        "pre_close": 9.9,
                        "open_qfq": 5.0,
                        "high_qfq": 5.2,
                        "low_qfq": 4.9,
                        "close_qfq": 5.1,
                        "pre_close_qfq": 4.95,
                        "atr": 0.8,
                        "atr_qfq": 0.4,
                        "macd": 1.2,
                        "macd_qfq": 0.6,
                        "amount": 1000.0,
                        "vol": 200.0,
                        "turnover_rate": 0.1,
                    }
                ]
            )
            gtja_target = pd.DataFrame(
                [
                    {
                        "trade_date": "20260701",
                        "stock_code": "000001.SZ",
                        "gtja_alpha001_qfq": 0.123,
                    }
                ]
            )

            mapping_path = root / "industry_encode_mapping.json"
            with mock.patch.object(
                refresh_l3,
                "default_industry_encode_mapping_path",
                return_value=mapping_path,
            ):
                result = refresh_l3._build_target_feature_frame(
                    raw_parts_dir,
                    raw_target,
                    gtja_target,
                )

            self.assertIn("open_qfq", result.columns)
            self.assertIn("high_qfq", result.columns)
            self.assertIn("low_qfq", result.columns)
            self.assertIn("close_qfq", result.columns)
            self.assertIn("pre_close_qfq", result.columns)
            self.assertIn("atr_qfq", result.columns)
            self.assertIn("macd_qfq", result.columns)
            self.assertIn("gtja_alpha001_qfq", result.columns)
            self.assertNotIn("open", result.columns)
            self.assertNotIn("high", result.columns)
            self.assertNotIn("low", result.columns)
            self.assertNotIn("close", result.columns)
            self.assertNotIn("pre_close", result.columns)
            self.assertNotIn("atr", result.columns)
            self.assertNotIn("macd", result.columns)

    def test_build_gtja_target_frame_includes_recent_bridge_rows_beyond_raw_parts_max_date(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_parts_dir = root / "raw_factor_by_stock_parts"
            raw_parts_dir.mkdir(parents=True, exist_ok=True)

            needed = refresh_l3.required_gtja_raw_columns()

            def make_row(trade_date: str, stock_code: str, base: float) -> dict:
                row = {column: base for column in needed if column not in {"trade_date", "stock_code"}}
                row["trade_date"] = trade_date
                row["stock_code"] = stock_code
                return row

            pd.DataFrame(
                [
                    make_row("20260701", "000001.SZ", 1.0),
                ]
            ).to_parquet(raw_parts_dir / "raw_part_0000.parquet", index=False)

            raw_recent = pd.DataFrame(
                [
                    make_row("20260703", "000001.SZ", 2.0),
                    make_row("20260706", "000001.SZ", 3.0),
                ]
            )

            captured = {}

            def fake_gtja(frame, **kwargs):
                captured["trade_dates"] = sorted(frame["trade_date"].astype(str).unique().tolist())
                return pd.DataFrame(
                    [
                        {
                            "trade_date": "20260706",
                            "stock_code": "000001.SZ",
                            refresh_l3.GTJA_ALPHA_COLUMNS[0]: 0.123,
                        }
                    ]
                )

            with mock.patch.object(refresh_l3, "compute_gtja_alpha_from_raw_factor", side_effect=fake_gtja):
                result = refresh_l3._build_gtja_target_frame(
                    raw_parts_dir,
                    raw_recent,
                    read_start="20250101",
                    target_date="20260706",
                )

            self.assertEqual(captured["trade_dates"], ["20260701", "20260703", "20260706"])
            self.assertEqual(result["trade_date"].astype(str).tolist(), ["20260706"])
            self.assertEqual(result["stock_code"].tolist(), ["000001.SZ"])


if __name__ == "__main__":
    unittest.main()
