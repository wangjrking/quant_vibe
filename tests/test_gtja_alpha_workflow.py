import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from data_process_module import get_factor_data
from gtja_alpha_workflow import audit_gtja_rank_scope, compute_gtja_alpha_from_raw_factor, write_gtja_rank_audit


class GtjaAlphaWorkflowTests(unittest.TestCase):
    @staticmethod
    def _trade_dates(count: int = 25) -> list[str]:
        return pd.date_range("2026-01-02", periods=count, freq="B").strftime("%Y%m%d").tolist()

    @classmethod
    def _build_raw_panel(cls) -> pd.DataFrame:
        records = []
        dates = cls._trade_dates()
        stocks = [
            ("A", -1.0),
            ("B", 0.0),
            ("C", 1.0),
            ("D", 2.0),
        ]
        last_close_by_code: dict[str, float] = {}

        for day_idx, trade_date in enumerate(dates):
            index_open = 3000.0 + day_idx * 2.0
            index_close = index_open + (1.5 if day_idx % 2 == 0 else -1.0)
            for stock_idx, (stock_code, slope) in enumerate(stocks):
                trend = day_idx * slope * 0.2
                open_price = 10.0 + stock_idx * 1.2 + trend
                close = open_price + 0.35 + stock_idx * 0.07 + day_idx * 0.03
                high = close + 0.8 + stock_idx * 0.02
                low = open_price - 0.6 - stock_idx * 0.03
                pre_close = last_close_by_code.get(stock_code, close - 0.5)
                vol = 1000.0 + stock_idx * 180.0 + day_idx * 22.0 + slope * day_idx * 6.0
                intended_vwap = (open_price + close) / 2.0
                amount = vol * intended_vwap / 10.0
                last_close_by_code[stock_code] = close
                records.append(
                    {
                        "trade_date": trade_date,
                        "stock_code": stock_code,
                        "name": stock_code,
                        "industry": "I1" if stock_idx < 2 else "I2",
                        "act_ent_type": "T",
                        "open": open_price,
                        "close": close,
                        "high": high,
                        "low": low,
                        "pre_close": pre_close,
                        "vol": vol,
                        "amount": amount,
                        "total_mv": 1000000.0 + stock_idx * 120000.0 + day_idx * 5000.0,
                        "pb": 1.2 + stock_idx * 0.35 + day_idx * 0.01,
                        "index_2000_open": index_open,
                        "index_2000_close": index_close,
                        "cost_5pct": close * 0.95,
                        "cost_15pct": close * 0.97,
                        "cost_50pct": close,
                        "cost_85pct": close * 1.03,
                        "cost_95pct": close * 1.05,
                        "weight_avg": intended_vwap,
                        "fd_amount": amount * 0.4,
                        "first_time": "09:30:00",
                        "last_time": "15:00:00",
                        "up_stat": "0/0",
                        "his_high": high + 0.5,
                        "his_low": low - 0.5,
                    }
                )
        return pd.DataFrame.from_records(records)

    @staticmethod
    def _alpha191_expected(frame: pd.DataFrame) -> pd.Series:
        ordered = frame.sort_values(["stock_code", "trade_date"]).copy()
        ordered["mean_vol_20"] = ordered.groupby("stock_code")["vol"].transform(
            lambda s: s.rolling(20, min_periods=1).mean()
        )
        corr_parts = []
        for _, group in ordered.groupby("stock_code", sort=False):
            corr_parts.append(group["mean_vol_20"].rolling(5, min_periods=1).corr(group["low"]))
        ordered["corr_mean_vol_low_5"] = pd.concat(corr_parts).sort_index()
        ordered["expected_alpha191"] = (
            ordered["corr_mean_vol_low_5"] + ((ordered["high"] + ordered["low"]) / 2.0) - ordered["close"]
        )
        last_date = ordered["trade_date"].max()
        expected = ordered.loc[ordered["trade_date"].eq(last_date), ["stock_code", "expected_alpha191"]]
        return expected.set_index("stock_code")["expected_alpha191"]

    @staticmethod
    def _alpha006_identity_expected(frame: pd.DataFrame) -> pd.Series:
        ordered = frame.sort_values(["stock_code", "trade_date"]).copy()
        sign_parts = []
        for _, group in ordered.groupby("stock_code", sort=False):
            sign_parts.append(np.sign(((group["open"] * 0.85) + (group["high"] * 0.15)).diff(4)))
        ordered["sign_delta4"] = pd.concat(sign_parts).sort_index()
        last_date = ordered["trade_date"].max()
        expected = ordered.loc[ordered["trade_date"].eq(last_date), ["stock_code", "sign_delta4"]]
        return (-1.0 * expected.set_index("stock_code")["sign_delta4"]).astype(float)

    def test_audit_gtja_rank_scope_flags_batch_rank_artifact(self):
        frame = pd.DataFrame(
            {
                "trade_date": ["20260102"] * 4,
                "stock_code": ["A", "B", "C", "D"],
                "gtja_alpha101": [0.5, 1.0, 0.5, 1.0],
            }
        )

        audit = audit_gtja_rank_scope(frame, ["gtja_alpha101"], min_stocks_per_day=4)

        self.assertFalse(audit["passed"])
        self.assertEqual(audit["issues"][0]["issue"], "rank_unique_count_too_low")

    def test_audit_gtja_rank_scope_accepts_full_market_rank(self):
        frame = pd.DataFrame(
            {
                "trade_date": ["20260102"] * 4,
                "stock_code": ["A", "B", "C", "D"],
                "gtja_alpha101": [0.25, 0.5, 0.75, 1.0],
            }
        )

        audit = audit_gtja_rank_scope(frame, ["gtja_alpha101"], min_stocks_per_day=4)

        self.assertTrue(audit["passed"])
        self.assertEqual(audit["issues"], [])

    def test_write_gtja_rank_audit_persists_report(self):
        frame = pd.DataFrame(
            {
                "trade_date": ["20260102"] * 4,
                "stock_code": ["A", "B", "C", "D"],
                "gtja_alpha101": [0.25, 0.5, 0.75, 1.0],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "audit.json"
            audit = write_gtja_rank_audit(frame, report_path, ["gtja_alpha101"], min_stocks_per_day=4)

            self.assertTrue(audit["passed"])
            self.assertTrue(report_path.exists())

    def test_compute_gtja_alpha_from_raw_factor_uses_official_alpha191_formula(self):
        raw = self._build_raw_panel()

        result = compute_gtja_alpha_from_raw_factor(raw, encode=False, drop_ts=True)
        last_date = raw["trade_date"].max()
        actual = result.loc[result["trade_date"].eq(last_date), ["stock_code", "gtja_alpha191"]].set_index("stock_code")[
            "gtja_alpha191"
        ]
        expected = self._alpha191_expected(raw)

        pd.testing.assert_series_equal(actual.sort_index(), expected.sort_index(), check_names=False, atol=1e-10, rtol=1e-10)

    def test_compute_gtja_alpha_from_raw_factor_identity_mode_keeps_raw_rank_inputs(self):
        raw = self._build_raw_panel()

        ranked = compute_gtja_alpha_from_raw_factor(raw, encode=False, drop_ts=True, cross_sectional_rank_mode="rank")
        identity = compute_gtja_alpha_from_raw_factor(raw, encode=False, drop_ts=True, cross_sectional_rank_mode="identity")
        last_date = raw["trade_date"].max()

        ranked_last = ranked.loc[ranked["trade_date"].eq(last_date), ["stock_code", "gtja_alpha006"]].set_index("stock_code")[
            "gtja_alpha006"
        ]
        identity_last = identity.loc[
            identity["trade_date"].eq(last_date), ["stock_code", "gtja_alpha006"]
        ].set_index("stock_code")["gtja_alpha006"]

        expected_identity = self._alpha006_identity_expected(raw)

        pd.testing.assert_series_equal(
            identity_last.sort_index(),
            expected_identity.sort_index(),
            check_names=False,
            atol=1e-10,
            rtol=1e-10,
        )
        self.assertFalse(np.allclose(ranked_last.sort_index().to_numpy(), identity_last.sort_index().to_numpy(), equal_nan=True))

    def test_compute_gtja_alpha_from_raw_factor_derives_ff3_inputs_for_alpha030(self):
        raw = self._build_raw_panel()

        result = compute_gtja_alpha_from_raw_factor(raw, encode=False, drop_ts=True)
        last_date = raw["trade_date"].max()
        alpha030_last = result.loc[result["trade_date"].eq(last_date), "gtja_alpha030"]

        self.assertGreater(int(alpha030_last.notna().sum()), 0)

    def test_governance_doc_marks_alpha030_ff3_inputs_as_project_level_exception(self):
        agents_doc = Path(__file__).resolve().parents[1] / "AGENTS.md"
        text = agents_doc.read_text(encoding="utf-8")

        self.assertIn("alpha030", text)
        self.assertTrue("项目内近似" in text or "治理例外" in text)

    def test_get_factor_data_uses_official_alpha191_formula(self):
        integ = self._build_raw_panel()

        result = get_factor_data(integ)
        last_date = integ["trade_date"].max()
        actual = result.loc[result["trade_date"].astype(str).eq(last_date), ["stock_code", "gtja_alpha191"]].set_index("stock_code")[
            "gtja_alpha191"
        ]
        expected = self._alpha191_expected(integ)

        pd.testing.assert_series_equal(actual.sort_index(), expected.sort_index(), check_names=False, atol=1e-10, rtol=1e-10)

    def test_data_process_module_does_not_keep_unreachable_legacy_gtja_block(self):
        source = (Path(__file__).resolve().parents[1] / "data_process_module.py").read_text(encoding="utf-8")

        return_pos = source.index("return factor_data")
        tail = source[return_pos:]
        self.assertNotIn("def cross_sectional_rank(series):", tail)
        self.assertNotIn("factor_data['gtja_alpha191']", tail)


if __name__ == "__main__":
    unittest.main()
