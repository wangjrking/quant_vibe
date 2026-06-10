import unittest

from optimizer_module import _score, evaluate_grid


class OptimizerModuleTests(unittest.TestCase):
    def test_evaluate_grid_prefers_threshold_that_removes_bad_trade(self):
        rows = [
            {"trade_date": "20260105", "stock_code": "GOOD", "pred_prob": 0.30, "close": 10.0, "10d_yield_rate": 0.20},
            {"trade_date": "20260105", "stock_code": "BAD", "pred_prob": 0.10, "close": 10.0, "10d_yield_rate": -0.50},
        ]

        results = evaluate_grid(
            rows,
            top_k_values=[2],
            min_pred_values=[None, 0.20],
            max_atr_values=[None],
            return_col="10d_yield_rate",
            commission_rate=0.0,
            sell_tax_rate=0.0,
            slippage_rate=0.0,
        )

        self.assertEqual(results[0]["min_pred_prob"], 0.20)
        self.assertGreater(results[0]["score"], results[1]["score"])

    def test_score_can_optimize_annualized_return(self):
        high_annual = {
            "trade_day_count": 10,
            "annualized_return": 0.50,
            "cumulative_return": 0.02,
            "max_drawdown": -0.30,
            "sharpe": 0.1,
            "calmar": 1.0,
        }
        high_cumulative = {
            "trade_day_count": 10,
            "annualized_return": 0.20,
            "cumulative_return": 0.30,
            "max_drawdown": -0.01,
            "sharpe": 0.1,
            "calmar": 1.0,
        }

        self.assertGreater(_score(high_annual, objective="annualized"), _score(high_cumulative, objective="annualized"))


if __name__ == "__main__":
    unittest.main()
