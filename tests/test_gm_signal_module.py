import tempfile
import unittest
from pathlib import Path

from gm_signal_module import build_gm_signal_rows, to_gm_symbol, write_gm_signals_csv
from selection_module import SelectionConfig


class GmSignalModuleTests(unittest.TestCase):
    def test_to_gm_symbol_converts_tushare_codes(self):
        self.assertEqual(to_gm_symbol("600000.SH"), "SHSE.600000")
        self.assertEqual(to_gm_symbol("000001.SZ"), "SZSE.000001")
        self.assertEqual(to_gm_symbol("430047.BJ"), "BJSE.430047")

    def test_build_gm_signal_rows_uses_next_trade_date(self):
        rows = [
            {
                "trade_date": "20260102",
                "stock_code": "600000.SH",
                "name": "A",
                "pred_prob": 0.30,
                "close": 10.0,
                "atr_qfq": 0.2,
            },
            {
                "trade_date": "20260103",
                "stock_code": "000001.SZ",
                "name": "B",
                "pred_prob": 0.20,
                "close": 10.0,
                "atr_qfq": 0.2,
            },
        ]

        signals = build_gm_signal_rows(rows, SelectionConfig(top_k=1, min_pred_prob=0.01))

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0]["signal_date"], "20260102")
        self.assertEqual(signals[0]["buy_date"], "20260103")
        self.assertEqual(signals[0]["symbol"], "SHSE.600000")

    def test_write_gm_signals_csv_rejects_empty_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                write_gm_signals_csv([], Path(tmpdir) / "signals.csv")

    def test_build_gm_signal_rows_can_attach_target_pct(self):
        rows = [
            {
                "trade_date": "20260102",
                "stock_code": "600000.SH",
                "name": "A",
                "pred_prob": 0.30,
                "close": 10.0,
                "atr_qfq": 0.2,
            },
            {
                "trade_date": "20260102",
                "stock_code": "000001.SZ",
                "name": "B",
                "pred_prob": 0.10,
                "close": 10.0,
                "atr_qfq": 0.2,
            },
            {
                "trade_date": "20260103",
                "stock_code": "000002.SZ",
                "name": "C",
                "pred_prob": 0.20,
                "close": 10.0,
                "atr_qfq": 0.2,
            },
        ]

        signals = build_gm_signal_rows(
            rows,
            SelectionConfig(top_k=2, min_pred_prob=0.01),
            weight_mode="score",
            target_total_pct=0.98,
        )

        self.assertEqual(len(signals), 2)
        self.assertAlmostEqual(sum(signal["target_pct"] for signal in signals), 0.98, places=6)
        self.assertGreater(signals[0]["target_pct"], signals[1]["target_pct"])

    def test_build_gm_signal_rows_can_downweight_low_liquidity_targets(self):
        rows = [
            {
                "trade_date": "20260102",
                "stock_code": "600000.SH",
                "name": "A",
                "pred_prob": 0.30,
                "close": 10.0,
                "atr_qfq": 0.2,
                "amount": 40000,
                "turnover_rate": 0.4,
            },
            {
                "trade_date": "20260102",
                "stock_code": "000001.SZ",
                "name": "B",
                "pred_prob": 0.29,
                "close": 10.0,
                "atr_qfq": 0.2,
                "amount": 300000,
                "turnover_rate": 2.0,
            },
            {
                "trade_date": "20260103",
                "stock_code": "000002.SZ",
                "name": "C",
                "pred_prob": 0.20,
                "close": 10.0,
                "atr_qfq": 0.2,
            },
        ]

        signals = build_gm_signal_rows(
            rows,
            SelectionConfig(top_k=2, min_pred_prob=0.01),
            weight_mode="equal",
            target_total_pct=0.98,
            liquidity_target_pct_enabled=True,
            liquidity_min_amount=100000,
            liquidity_min_turnover_rate=1.0,
            liquidity_mid_scale=0.8,
            liquidity_low_scale=0.6,
        )

        self.assertEqual(len(signals), 2)
        self.assertAlmostEqual(sum(signal["target_pct"] for signal in signals), 0.98, places=6)
        by_code = {signal["stock_code"]: signal for signal in signals}
        self.assertLess(by_code["600000.SH"]["target_pct"], by_code["000001.SZ"]["target_pct"])


if __name__ == "__main__":
    unittest.main()
