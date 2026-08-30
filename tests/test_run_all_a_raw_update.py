import unittest

import pandas as pd

from run_all_a_raw_update import DEFAULT_INDEX_CODES, build_source_preflight_report


class FakeTushare:
    def __init__(self, *, daily_rows=None, per_code_stk_factor=None):
        self.daily_rows = daily_rows if daily_rows is not None else ["000001.SZ", "920000.BJ"]
        self.per_code_stk_factor = per_code_stk_factor or {}

    @staticmethod
    def _frame(codes, trade_date="20260629"):
        return pd.DataFrame({"ts_code": codes, "trade_date": [trade_date] * len(codes)})

    def query(self, table, **kwargs):
        trade_date = kwargs.get("trade_date") or kwargs.get("start_date") or "20260629"
        if table == "daily":
            if "ts_code" in kwargs:
                code = kwargs["ts_code"]
                return self._frame([code], trade_date) if code in self.daily_rows else pd.DataFrame()
            return self._frame(self.daily_rows, trade_date)
        if table == "daily_basic":
            return self._frame([*self.daily_rows, "200011.SZ"], trade_date)
        if table == "index_daily":
            code = kwargs["ts_code"]
            return self._frame([code], trade_date)
        raise AssertionError(f"unexpected query table {table}")

    def moneyflow(self, **kwargs):
        trade_date = kwargs.get("trade_date") or kwargs.get("start_date") or "20260629"
        return self._frame(["000001.SZ"], trade_date)

    def stk_factor_pro(self, **kwargs):
        trade_date = kwargs.get("trade_date") or kwargs.get("start_date") or "20260629"
        if "ts_code" in kwargs:
            code = kwargs["ts_code"]
            return self._frame([code], trade_date) if self.per_code_stk_factor.get(code) else pd.DataFrame()
        return self._frame(["000001.SZ"], trade_date)

    def limit_list_d(self, **kwargs):
        trade_date = kwargs.get("trade_date") or kwargs.get("start_date") or "20260629"
        return self._frame(["000001.SZ"], trade_date)

    def cyq_perf(self, **kwargs):
        trade_date = kwargs.get("trade_date") or kwargs.get("start_date") or "20260629"
        if "ts_code" in kwargs:
            code = kwargs["ts_code"]
            return self._frame([code], trade_date) if code in self.daily_rows else pd.DataFrame()
        return self._frame(self.daily_rows, trade_date)

    def adj_factor(self, **kwargs):
        trade_date = kwargs.get("trade_date") or "20260629"
        return self._frame([*self.daily_rows, "000524.SZ"], trade_date)

    def stock_st(self, **kwargs):
        return pd.DataFrame(columns=["ts_code", "trade_date"])


class RunAllARawUpdatePreflightTests(unittest.TestCase):
    def test_source_preflight_filters_bj_from_driver_universe(self):
        report = build_source_preflight_report(
            FakeTushare(),
            ["daily", "daily_basic", "stk_factor", "moneyflow", "limit_list", "cyq_perf", "adj_factor", "stock_st", "index_daily"],
            "20260629",
            ["000001.SZ", "920000.BJ"],
            DEFAULT_INDEX_CODES,
        )

        self.assertTrue(report["gate_pass"])
        self.assertEqual(report["status"], "source_ready")
        self.assertEqual(report["universe_rule"], "no_bj")
        self.assertEqual(report["driver_source_rows"], 1)
        self.assertEqual(report["driver_source_codes"], 1)
        self.assertEqual(report["tables"]["stk_factor"]["missing_vs_daily_count"], 0)
        self.assertNotIn("gap_nature", report["tables"]["stk_factor"])
        self.assertEqual(report["tables"]["adj_factor"]["calendar_governance_note"], "daily_data remains the driver")

    def test_source_preflight_ignores_bj_only_per_code_discrepancy_after_no_bj_filter(self):
        report = build_source_preflight_report(
            FakeTushare(per_code_stk_factor={"920000.BJ": True}),
            ["daily", "stk_factor", "cyq_perf", "adj_factor", "index_daily"],
            "20260629",
            ["000001.SZ", "920000.BJ"],
            DEFAULT_INDEX_CODES,
        )

        self.assertTrue(report["gate_pass"])
        self.assertNotIn("stk_factor direct trade_date source missed codes that per-code source returned", report["blockers"])
        self.assertEqual(report["tables"]["stk_factor"]["missing_vs_daily_count"], 0)

    def test_source_preflight_blocks_zero_daily_driver(self):
        report = build_source_preflight_report(
            FakeTushare(daily_rows=[]),
            ["daily", "index_daily"],
            "20260629",
            ["000001.SZ"],
            DEFAULT_INDEX_CODES,
        )

        self.assertFalse(report["gate_pass"])
        self.assertIn("daily source returned zero rows", report["blockers"])

    def test_source_preflight_blocks_multi_code_driver_coverage_gap(self):
        report = build_source_preflight_report(
            FakeTushare(daily_rows=["000001.SZ", "000002.SZ", "000003.SZ"]),
            ["daily", "stk_factor", "cyq_perf", "adj_factor", "index_daily"],
            "20260629",
            ["000001.SZ", "000002.SZ", "000003.SZ"],
            DEFAULT_INDEX_CODES,
        )

        self.assertFalse(report["gate_pass"])
        self.assertEqual(report["status"], "source_not_ready")
        self.assertIn(
            "stk_factor source coverage gap vs daily_data exceeds single-code source-limited review scope: 2 codes",
            report["blockers"],
        )
        self.assertEqual(
            report["tables"]["stk_factor"]["gap_nature"],
            "source_not_ready_multi_code_coverage_gap",
        )


if __name__ == "__main__":
    unittest.main()
