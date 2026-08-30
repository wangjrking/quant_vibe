import unittest

from selection_module import SelectionConfig, parse_args, select_candidates


class SelectionModuleTests(unittest.TestCase):
    def test_default_config_uses_optimized_risk_parameters(self):
        config = SelectionConfig()

        self.assertEqual(config.top_k, 3)
        self.assertEqual(config.min_pred_prob, 0.01)
        self.assertEqual(config.max_atr_ratio, 0.10)

    def test_cli_defaults_use_optimized_risk_parameters(self):
        args = parse_args([])

        self.assertEqual(args.top_k, 3)
        self.assertEqual(args.min_pred, 0.01)
        self.assertEqual(args.max_atr_ratio, 0.10)

    def test_selects_latest_date_and_applies_risk_filters(self):
        rows = [
            {"trade_date": "20260105", "stock_code": "OLD", "name": "Old", "close_qfq": 10.0, "pred_prob": 0.90, "atr_qfq": 0.1, "industry_encode": 1},
            {"trade_date": "20260106", "stock_code": "ST1", "name": "ST Bad", "close_qfq": 10.0, "pred_prob": 0.80, "atr_qfq": 0.1, "industry_encode": 1},
            {"trade_date": "20260106", "stock_code": "VOL", "name": "Volatile", "close_qfq": 10.0, "pred_prob": 0.70, "atr_qfq": 2.0, "industry_encode": 1},
            {"trade_date": "20260106", "stock_code": "OK1", "name": "Good One", "close_qfq": 10.0, "pred_prob": 0.60, "atr_qfq": 0.3, "industry_encode": 1},
            {"trade_date": "20260106", "stock_code": "OK2", "name": "Good Two", "close_qfq": 10.0, "pred_prob": 0.50, "atr_qfq": 0.2, "industry_encode": 1},
            {"trade_date": "20260106", "stock_code": "OK3", "name": "Good Three", "close_qfq": 10.0, "pred_prob": 0.40, "atr_qfq": 0.1, "industry_encode": 2},
        ]

        selected = select_candidates(rows, SelectionConfig(top_k=3, max_atr_ratio=0.10, max_per_industry=1))

        self.assertEqual([row["stock_code"] for row in selected], ["OK1", "OK3"])

    def test_filters_limit_rows_and_low_predictions(self):
        rows = [
            {"trade_date": "20260106", "stock_code": "LIMIT", "name": "Limit", "close_qfq": 10.0, "pred_prob": 0.80, "atr_qfq": 0.1, "limit_times": "1", "industry_encode": 1},
            {"trade_date": "20260106", "stock_code": "LOW", "name": "Low", "close_qfq": 10.0, "pred_prob": 0.01, "atr_qfq": 0.1, "industry_encode": 2},
            {"trade_date": "20260106", "stock_code": "OK", "name": "Good", "close_qfq": 10.0, "pred_prob": 0.05, "atr_qfq": 0.1, "industry_encode": 3},
        ]

        selected = select_candidates(rows, SelectionConfig(top_k=5, min_pred_prob=0.02))

        self.assertEqual([row["stock_code"] for row in selected], ["OK"])

    def test_select_candidates_prefers_close_qfq_for_atr_ratio(self):
        rows = [
            {
                "trade_date": "20260106",
                "stock_code": "OK",
                "name": "Good",
                "pred_prob": 0.50,
                "close": 4.0,
                "close_qfq": 10.0,
                "atr_qfq": 0.5,
                "industry_encode": 1,
            }
        ]

        selected = select_candidates(rows, SelectionConfig(top_k=3, max_atr_ratio=0.10))

        self.assertEqual([row["stock_code"] for row in selected], ["OK"])

    def test_select_candidates_requires_explicit_close_qfq(self):
        rows = [
            {
                "trade_date": "20260106",
                "stock_code": "OK",
                "name": "Good",
                "pred_prob": 0.50,
                "close": 10.0,
                "atr_qfq": 0.5,
                "industry_encode": 1,
            }
        ]

        with self.assertRaisesRegex(KeyError, "requires explicit front-adjusted column: close_qfq"):
            select_candidates(rows, SelectionConfig(top_k=3, max_atr_ratio=0.10))

    def test_limit_times_zero_is_not_current_limit(self):
        rows = [
            {"trade_date": "20260106", "stock_code": "LIMIT", "name": "Limit", "close_qfq": 10.0, "pred_prob": 0.80, "atr_qfq": 0.1, "limit_times": "1", "industry_encode": 1},
            {"trade_date": "20260106", "stock_code": "ZERO", "name": "Zero", "close_qfq": 10.0, "pred_prob": 0.70, "atr_qfq": 0.1, "limit_times": "0", "industry_encode": 2},
            {"trade_date": "20260106", "stock_code": "OK", "name": "Good", "close_qfq": 10.0, "pred_prob": 0.60, "atr_qfq": 0.1, "industry_encode": 3},
        ]

        selected = select_candidates(rows, SelectionConfig(top_k=5, min_pred_prob=0.02))

        self.assertEqual([row["stock_code"] for row in selected], ["ZERO", "OK"])

    def test_filters_delisting_names(self):
        rows = [
            {"trade_date": "20260106", "stock_code": "BAD1", "name": "\u9000\u5e02\u521b\u5174", "close_qfq": 10.0, "pred_prob": 0.90, "atr_qfq": 0.1, "industry_encode": 1},
            {"trade_date": "20260106", "stock_code": "BAD2", "name": "\u9000A", "close_qfq": 10.0, "pred_prob": 0.80, "atr_qfq": 0.1, "industry_encode": 1},
            {"trade_date": "20260106", "stock_code": "OK", "name": "Good", "close_qfq": 10.0, "pred_prob": 0.70, "atr_qfq": 0.1, "industry_encode": 1},
        ]

        selected = select_candidates(rows, SelectionConfig(top_k=3))

        self.assertEqual([row["stock_code"] for row in selected], ["OK"])

    def test_filters_delisting_suffix_names(self):
        rows = [
            {"trade_date": "20260106", "stock_code": "BAD", "name": "\u5929\u9f99\u9000", "close_qfq": 10.0, "pred_prob": 0.90, "atr_qfq": 0.1, "industry_encode": 1},
            {"trade_date": "20260106", "stock_code": "OK", "name": "Good", "close_qfq": 10.0, "pred_prob": 0.70, "atr_qfq": 0.1, "industry_encode": 1},
        ]

        selected = select_candidates(rows, SelectionConfig(top_k=3))

        self.assertEqual([row["stock_code"] for row in selected], ["OK"])


if __name__ == "__main__":
    unittest.main()
