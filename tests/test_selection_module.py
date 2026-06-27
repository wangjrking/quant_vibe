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
            {"trade_date": "20260105", "stock_code": "OLD", "name": "Old", "pred_prob": 0.90, "close": 10.0, "atr_qfq": 0.1, "industry_encode": 1},
            {"trade_date": "20260106", "stock_code": "ST1", "name": "ST Bad", "pred_prob": 0.80, "close": 10.0, "atr_qfq": 0.1, "industry_encode": 1},
            {"trade_date": "20260106", "stock_code": "VOL", "name": "Volatile", "pred_prob": 0.70, "close": 10.0, "atr_qfq": 2.0, "industry_encode": 1},
            {"trade_date": "20260106", "stock_code": "OK1", "name": "Good One", "pred_prob": 0.60, "close": 10.0, "atr_qfq": 0.3, "industry_encode": 1},
            {"trade_date": "20260106", "stock_code": "OK2", "name": "Good Two", "pred_prob": 0.50, "close": 10.0, "atr_qfq": 0.2, "industry_encode": 1},
            {"trade_date": "20260106", "stock_code": "OK3", "name": "Good Three", "pred_prob": 0.40, "close": 10.0, "atr_qfq": 0.1, "industry_encode": 2},
        ]

        selected = select_candidates(rows, SelectionConfig(top_k=3, max_atr_ratio=0.10, max_per_industry=1))

        self.assertEqual([row["stock_code"] for row in selected], ["OK1", "OK3"])

    def test_filters_limit_rows_and_low_predictions(self):
        rows = [
            {"trade_date": "20260106", "stock_code": "LIMIT", "name": "Limit", "pred_prob": 0.80, "close": 10.0, "atr_qfq": 0.1, "limit_times": "1", "industry_encode": 1},
            {"trade_date": "20260106", "stock_code": "LOW", "name": "Low", "pred_prob": 0.01, "close": 10.0, "atr_qfq": 0.1, "industry_encode": 2},
            {"trade_date": "20260106", "stock_code": "OK", "name": "Good", "pred_prob": 0.05, "close": 10.0, "atr_qfq": 0.1, "industry_encode": 3},
        ]

        selected = select_candidates(rows, SelectionConfig(top_k=5, min_pred_prob=0.02))

        self.assertEqual([row["stock_code"] for row in selected], ["OK"])

    def test_limit_times_zero_is_not_current_limit(self):
        rows = [
            {"trade_date": "20260106", "stock_code": "LIMIT", "name": "Limit", "pred_prob": 0.80, "close": 10.0, "atr_qfq": 0.1, "limit_times": "1", "industry_encode": 1},
            {"trade_date": "20260106", "stock_code": "ZERO", "name": "Zero", "pred_prob": 0.70, "close": 10.0, "atr_qfq": 0.1, "limit_times": "0", "industry_encode": 2},
            {"trade_date": "20260106", "stock_code": "OK", "name": "Good", "pred_prob": 0.60, "close": 10.0, "atr_qfq": 0.1, "industry_encode": 3},
        ]

        selected = select_candidates(rows, SelectionConfig(top_k=5, min_pred_prob=0.02))

        self.assertEqual([row["stock_code"] for row in selected], ["ZERO", "OK"])

    def test_filters_delisting_names(self):
        rows = [
            {"trade_date": "20260106", "stock_code": "BAD1", "name": "\u9000\u5e02\u521b\u5174", "pred_prob": 0.90, "close": 10.0, "atr_qfq": 0.1, "industry_encode": 1},
            {"trade_date": "20260106", "stock_code": "BAD2", "name": "\u9000A", "pred_prob": 0.80, "close": 10.0, "atr_qfq": 0.1, "industry_encode": 1},
            {"trade_date": "20260106", "stock_code": "OK", "name": "Good", "pred_prob": 0.70, "close": 10.0, "atr_qfq": 0.1, "industry_encode": 1},
        ]

        selected = select_candidates(rows, SelectionConfig(top_k=3))

        self.assertEqual([row["stock_code"] for row in selected], ["OK"])

    def test_filters_delisting_suffix_names(self):
        rows = [
            {"trade_date": "20260106", "stock_code": "BAD", "name": "\u5929\u9f99\u9000", "pred_prob": 0.90, "close": 10.0, "atr_qfq": 0.1, "industry_encode": 1},
            {"trade_date": "20260106", "stock_code": "OK", "name": "Good", "pred_prob": 0.70, "close": 10.0, "atr_qfq": 0.1, "industry_encode": 1},
        ]

        selected = select_candidates(rows, SelectionConfig(top_k=3))

        self.assertEqual([row["stock_code"] for row in selected], ["OK"])


if __name__ == "__main__":
    unittest.main()
