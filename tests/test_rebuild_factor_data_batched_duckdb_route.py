import json
import sys
import tempfile
import unittest
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rebuild_factor_data_batched import _load_batch, _stock_codes


class RebuildFactorDataBatchedDuckdbRouteTests(unittest.TestCase):
    def test_helpers_follow_active_l2_duckdb_route_without_sqlite_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            asset_registry_dir = data_dir / "asset_registry"
            asset_registry_dir.mkdir(parents=True)
            duckdb_path = data_dir / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
            duckdb_path.parent.mkdir(parents=True)

            with duckdb.connect(str(duckdb_path)) as conn:
                conn.execute(
                    """
                    CREATE TABLE STOCK_DAILY_DATA(
                        stock_code TEXT,
                        trade_date TEXT,
                        close DOUBLE
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO STOCK_DAILY_DATA VALUES
                    ('000001.SZ', '20260701', 10.0),
                    ('000002.SZ', '20260701', 20.0)
                    """
                )

            registry = {
                "assets": [
                    {
                        "asset_id": "prod_l2_duckdb_stock_daily_data_split_20260701",
                        "layer": "l2_stock_daily_base",
                        "asset_type": "duckdb_table",
                        "status": "production_active",
                        "allowed_for_main_workflow": True,
                        "asset_path": str(duckdb_path) + "::STOCK_DAILY_DATA",
                    }
                ]
            }
            (asset_registry_dir / "production_assets.json").write_text(
                json.dumps(registry, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            codes = _stock_codes(data_dir=data_dir)
            frame = _load_batch(data_dir=data_dir, codes=["000001.SZ"])

            self.assertEqual(codes, ["000001.SZ", "000002.SZ"])
            self.assertEqual(frame["stock_code"].tolist(), ["000001.SZ"])
            self.assertEqual(frame["trade_date"].tolist(), ["20260701"])
            self.assertEqual(frame["close"].tolist(), [10.0])


if __name__ == "__main__":
    unittest.main()
