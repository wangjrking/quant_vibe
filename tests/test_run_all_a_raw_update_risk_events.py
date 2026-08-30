from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import duckdb
import pandas as pd


MAIN_DIR = Path(__file__).resolve().parents[1]
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

import run_all_a_raw_update as raw_update
from tushare_stock_risk_event_contract import fetch_complete_range


class CappedPro:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def query(self, api: str, *, start_date: str, end_date: str):
        self.calls.append((start_date, end_date))
        if start_date == "20260801" and end_date == "20260804":
            return pd.DataFrame({"row": range(1000)})
        return pd.DataFrame({"row": [f"{start_date}:{end_date}"]})


class RiskEventRawUpdateTests(unittest.TestCase):
    def test_source_cap_is_split_by_date(self) -> None:
        pro = CappedPro()

        result = fetch_complete_range(pro, "stk_shock", "20260801", "20260804")

        self.assertEqual(len(result), 2)
        self.assertEqual(pro.calls[0], ("20260801", "20260804"))
        self.assertGreater(len(pro.calls), 1)

    def test_parquet_sync_creates_one_table_duckdb(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary)
            parquet = data_dir / "stk_shock.parquet"
            pd.DataFrame(
                {
                    "ts_code": ["000001.SZ"],
                    "trade_date": ["20260821"],
                    "name": ["A"],
                    "trade_market": ["深交所"],
                    "reason": ["reason"],
                    "period": ["period"],
                    "source_api": ["stk_shock"],
                    "source_row_sha256": ["digest"],
                    "fetched_at": ["2026-08-22T00:00:00+08:00"],
                }
            ).to_parquet(parquet, index=False)

            path = raw_update._sync_risk_event_duckdb(
                data_dir,
                "stk_shock",
                parquet,
            )

            connection = duckdb.connect(str(path), read_only=True)
            try:
                tables = connection.execute("SHOW TABLES").fetchall()
                rows = connection.execute("SELECT COUNT(*) FROM stk_shock").fetchone()[0]
            finally:
                connection.close()
            self.assertEqual(tables, [("stk_shock",)])
            self.assertEqual(rows, 1)


if __name__ == "__main__":
    unittest.main()
