from __future__ import annotations

import sys
import unittest
from pathlib import Path

import duckdb


MAIN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAIN))

import update_fw_soft_deepdrop_l5_signal as module  # noqa: E402


class UpdateFwSoftDeepdropL5SignalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.con = duckdb.connect()
        for schema in ("l4_1d", "l4_3d", "l4_5d", "l4_10d", "market"):
            self.con.execute(f"CREATE SCHEMA {schema}")
        for schema in ("l4_1d", "l4_3d", "l4_5d", "l4_10d"):
            self.con.execute(
                f"CREATE TABLE {schema}.predictions "
                "(trade_date VARCHAR, stock_code VARCHAR, pred_prob DOUBLE)"
            )
            self.con.execute(
                f"INSERT INTO {schema}.predictions VALUES "
                "('20260717', '000001.SZ', 0.90), "
                "('20260717', '000002.SZ', 0.70), "
                "('20260717', '000003.SZ', 0.50), "
                "('20260717', '000004.SZ', 0.30)"
            )
        self.con.execute(
            """
            CREATE TABLE market.STOCK_DAILY_DATA (
                trade_date VARCHAR,
                stock_code VARCHAR,
                name VARCHAR,
                market VARCHAR,
                amount DOUBLE,
                turnover_rate DOUBLE,
                total_mv DOUBLE,
                atr_qfq DOUBLE,
                pct_chg DOUBLE,
                open DOUBLE,
                pre_close DOUBLE,
                ST_TYPE VARCHAR,
                ST_TYPE_name VARCHAR
            )
            """
        )
        for code, name in (
            ("000001.SZ", "正常股票"),
            ("000002.SZ", "高开股票"),
            ("000003.SZ", "风险股票"),
            ("000004.SZ", "涨停股票"),
        ):
            self.con.execute(
                "INSERT INTO market.STOCK_DAILY_DATA VALUES "
                "('20260717', ?, ?, '主板', 200000, 5, 300000, 5, -3, 10, 10.5, '', '')",
                [code, name],
            )
        self.con.execute(
            "INSERT INTO market.STOCK_DAILY_DATA VALUES "
            "('20260720', '000001.SZ', '正常股票', '主板', 200000, 5, 300000, 5, 0, 10, 10, '', '')"
        )
        self.con.execute(
            "INSERT INTO market.STOCK_DAILY_DATA VALUES "
            "('20260720', '000002.SZ', '高开股票', '主板', 200000, 5, 300000, 5, 0, 10.2, 10, '', '')"
        )
        self.con.execute(
            "INSERT INTO market.STOCK_DAILY_DATA VALUES "
            "('20260720', '000003.SZ', 'ST风险股票', '主板', 200000, 5, 300000, 5, 0, 10, 10, '1', 'ST')"
        )
        self.con.execute(
            "INSERT INTO market.STOCK_DAILY_DATA VALUES "
            "('20260720', '000004.SZ', '涨停股票', '主板', 200000, 5, 300000, 5, 0, 11, 10, '', '')"
        )

    def tearDown(self) -> None:
        self.con.close()

    def test_build_latest_compiles_and_applies_available_buy_day_gate(self) -> None:
        sources = {
            label: {"table": "predictions"}
            for label in ("1d", "3d", "5d", "10d")
        }

        latest, audit = module.build_latest(
            self.con,
            sources,
            signal_date="20260717",
            buy_date="20260720",
            buy_day_market_available=True,
        )

        self.assertEqual(latest["stock_code"].tolist(), ["000001.SZ"])
        self.assertTrue(bool(latest.iloc[0]["buy_day_hard_gate_complete"]))
        self.assertEqual(audit["actual_rows"], 1)
        self.assertEqual(audit["shortage"], 2)


if __name__ == "__main__":
    unittest.main()
