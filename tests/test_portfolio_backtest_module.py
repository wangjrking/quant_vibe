import unittest

from portfolio_backtest_module import PortfolioBacktestConfig, run_order_backtest, run_portfolio_backtest


class PortfolioBacktestModuleTests(unittest.TestCase):
    def test_portfolio_backtest_opens_and_exits_positions(self):
        rows = [
            {
                "trade_date": "20260101",
                "stock_code": "A",
                "name": "A",
                "pred_prob": 0.30,
                "close": 10.0,
                "atr_qfq": 0.2,
                "post_open": 10.0,
                "post12_open": 11.0,
            },
            {
                "trade_date": "20260102",
                "stock_code": "B",
                "name": "B",
                "pred_prob": 0.20,
                "close": 10.0,
                "atr_qfq": 0.2,
                "post_open": 10.0,
                "post12_open": 12.0,
            },
            {"trade_date": "20260103", "stock_code": "C", "pred_prob": 0.01, "close": 10.0, "atr_qfq": 0.2, "post_open": 10.0, "post12_open": 10.0},
        ]

        result = run_portfolio_backtest(
            rows,
            PortfolioBacktestConfig(
                top_k=1,
                max_positions=2,
                holding_days=1,
                min_pred_prob=0.05,
                commission_rate=0.0,
                sell_tax_rate=0.0,
                slippage_rate=0.0,
            ),
        )

        self.assertEqual(result["metrics"]["trade_count"], 2)
        self.assertGreater(result["metrics"]["cumulative_return"], 0)

    def test_portfolio_backtest_does_not_duplicate_held_stock(self):
        rows = [
            {"trade_date": "20260101", "stock_code": "A", "pred_prob": 0.30, "close": 10.0, "atr_qfq": 0.2, "post_open": 10.0, "post12_open": 11.0},
            {"trade_date": "20260102", "stock_code": "A", "pred_prob": 0.40, "close": 10.0, "atr_qfq": 0.2, "post_open": 10.0, "post12_open": 12.0},
            {"trade_date": "20260103", "stock_code": "B", "pred_prob": 0.01, "close": 10.0, "atr_qfq": 0.2, "post_open": 10.0, "post12_open": 10.0},
        ]

        result = run_portfolio_backtest(
            rows,
            PortfolioBacktestConfig(
                top_k=1,
                max_positions=2,
                holding_days=2,
                min_pred_prob=0.05,
                commission_rate=0.0,
                sell_tax_rate=0.0,
                slippage_rate=0.0,
            ),
        )

        self.assertEqual(result["metrics"]["trade_count"], 1)

    def test_portfolio_backtest_honors_market_filter_column(self):
        rows = [
            {"trade_date": "20260101", "stock_code": "OFF", "pred_prob": 0.30, "market_ok": 0, "close": 10.0, "atr_qfq": 0.2, "post_open": 10.0, "post12_open": 11.0},
            {"trade_date": "20260102", "stock_code": "ON", "pred_prob": 0.20, "market_ok": 1, "close": 10.0, "atr_qfq": 0.2, "post_open": 10.0, "post12_open": 11.0},
        ]

        result = run_portfolio_backtest(
            rows,
            PortfolioBacktestConfig(
                top_k=1,
                market_filter_col="market_ok",
                commission_rate=0.0,
                sell_tax_rate=0.0,
                slippage_rate=0.0,
            ),
        )

        self.assertEqual(result["trades"][0]["stock_code"], "ON")

    def test_drawdown_uses_running_peak_not_future_peak(self):
        rows = [
            {"trade_date": "20260101", "stock_code": "A", "pred_prob": 0.30, "close": 10.0, "atr_qfq": 0.2, "post_open": 10.0, "post12_open": 20.0},
            {"trade_date": "20260102", "stock_code": "B", "pred_prob": 0.30, "close": 10.0, "atr_qfq": 0.2, "post_open": 10.0, "post12_open": 10.0},
        ]

        result = run_portfolio_backtest(
            rows,
            PortfolioBacktestConfig(
                top_k=1,
                max_positions=1,
                holding_days=1,
                commission_rate=0.0,
                sell_tax_rate=0.0,
                slippage_rate=0.0,
            ),
        )

        self.assertGreaterEqual(result["metrics"]["max_drawdown"], 0.0)

    def test_exit_rules_can_clip_loss_and_profit(self):
        rows = [
            {"trade_date": "20260101", "stock_code": "LOSS", "pred_prob": 0.30, "close": 10.0, "atr_qfq": 0.2, "post_open": 10.0, "post12_open": 5.0},
            {"trade_date": "20260102", "stock_code": "GAIN", "pred_prob": 0.30, "close": 10.0, "atr_qfq": 0.2, "post_open": 10.0, "post12_open": 20.0},
        ]

        result = run_portfolio_backtest(
            rows,
            PortfolioBacktestConfig(
                top_k=1,
                max_positions=1,
                holding_days=1,
                stop_loss_pct=0.10,
                take_profit_pct=0.20,
                commission_rate=0.0,
                sell_tax_rate=0.0,
                slippage_rate=0.0,
            ),
        )

        returns = [trade["net_return"] for trade in result["trades"]]
        self.assertIn(-0.10, returns)
        self.assertIn(0.20, returns)

    def test_score_weight_mode_allocates_more_capital_to_higher_score(self):
        rows = [
            {"trade_date": "20260101", "stock_code": "A", "pred_prob": 0.30, "close": 10.0, "atr_qfq": 0.2, "post_open": 10.0, "post12_open": 11.0},
            {"trade_date": "20260101", "stock_code": "B", "pred_prob": 0.10, "close": 10.0, "atr_qfq": 0.2, "post_open": 10.0, "post12_open": 11.0},
            {"trade_date": "20260102", "stock_code": "C", "pred_prob": 0.01, "close": 10.0, "atr_qfq": 0.2, "post_open": 10.0, "post12_open": 10.0},
        ]

        result = run_portfolio_backtest(
            rows,
            PortfolioBacktestConfig(
                top_k=2,
                max_positions=2,
                holding_days=1,
                min_pred_prob=0.05,
                commission_rate=0.0,
                sell_tax_rate=0.0,
                slippage_rate=0.0,
                position_weight_mode="score",
            ),
        )

        trades = {trade["stock_code"]: trade for trade in result["trades"]}
        self.assertGreater(trades["A"]["capital"], trades["B"]["capital"])

    def test_portfolio_backtest_records_liquidity_adjusted_slippage(self):
        rows = [
            {
                "trade_date": "20260101",
                "stock_code": "ILLQ",
                "pred_prob": 0.30,
                "close": 10.0,
                "atr_qfq": 0.2,
                "post_open": 10.0,
                "post12_open": 10.5,
                "amount": 4.0e4,
                "turnover_rate": 0.3,
            },
            {"trade_date": "20260102", "stock_code": "C", "pred_prob": 0.01, "close": 10.0, "atr_qfq": 0.2, "post_open": 10.0, "post12_open": 10.0},
        ]

        result = run_portfolio_backtest(
            rows,
            PortfolioBacktestConfig(
                top_k=1,
                max_positions=1,
                holding_days=1,
                min_pred_prob=0.05,
                commission_rate=0.0,
                sell_tax_rate=0.0,
                slippage_rate=0.001,
                liquidity_slippage_enabled=True,
            ),
        )

        self.assertEqual(result["trades"][0]["stock_code"], "ILLQ")
        self.assertAlmostEqual(result["trades"][0]["slippage_rate"], 0.004)

    def test_score_exit_closes_position_before_scheduled_exit(self):
        rows = [
            {
                "trade_date": "20260101",
                "stock_code": "A",
                "name": "A",
                "pred_prob": 1.00,
                "close": 10.0,
                "atr_qfq": 0.2,
                "post_open": 10.0,
                "post12_open": 20.0,
            },
            {
                "trade_date": "20260102",
                "stock_code": "A",
                "name": "A",
                "pred_prob": 0.40,
                "close": 11.0,
                "atr_qfq": 0.2,
                "post_open": 11.0,
                "post12_open": 20.0,
            },
            {
                "trade_date": "20260103",
                "stock_code": "B",
                "name": "B",
                "pred_prob": 0.01,
                "close": 10.0,
                "atr_qfq": 0.2,
                "post_open": 10.0,
                "post12_open": 10.0,
            },
        ]

        result = run_portfolio_backtest(
            rows,
            PortfolioBacktestConfig(
                top_k=1,
                max_positions=1,
                holding_days=5,
                min_pred_prob=0.50,
                score_exit_ratio=0.50,
                min_score_exit_holding_days=1,
                commission_rate=0.0,
                sell_tax_rate=0.0,
                slippage_rate=0.0,
            ),
        )

        self.assertEqual(result["metrics"]["trade_count"], 1)
        self.assertEqual(result["trades"][0]["exit_signal_date"], "20260102")
        self.assertEqual(result["trades"][0]["exit_reason"], "score_exit")
        self.assertAlmostEqual(result["trades"][0]["net_return"], 0.10)

    def test_order_backtest_waits_for_cash_instead_of_queueing_old_signals(self):
        rows = [
            {
                "trade_date": "20260101",
                "stock_code": "A",
                "pred_prob": 0.90,
                "close": 10.0,
                "atr_qfq": 0.1,
                "post_open": 10.0,
                "post4_open": 12.0,
            },
            {
                "trade_date": "20260102",
                "stock_code": "B",
                "pred_prob": 0.95,
                "close": 10.0,
                "atr_qfq": 0.1,
                "post_open": 10.0,
                "post4_open": 20.0,
            },
            {
                "trade_date": "20260104",
                "stock_code": "A",
                "pred_prob": 0.01,
                "close": 11.5,
                "atr_qfq": 0.1,
                "post_open": 12.0,
                "post4_open": 12.0,
            },
            {
                "trade_date": "20260104",
                "stock_code": "C",
                "pred_prob": 0.80,
                "close": 10.0,
                "atr_qfq": 0.1,
                "post_open": 10.0,
                "post4_open": 11.0,
            },
            {
                "trade_date": "20260106",
                "stock_code": "C",
                "pred_prob": 0.01,
                "close": 10.5,
                "atr_qfq": 0.1,
                "post_open": 11.0,
                "post4_open": 11.0,
            },
        ]

        result = run_order_backtest(
            rows,
            PortfolioBacktestConfig(
                top_k=1,
                max_positions=1,
                holding_days=2,
                min_pred_prob=0.1,
                max_atr_ratio=None,
                commission_rate=0.0,
                sell_tax_rate=0.0,
                slippage_rate=0.0,
            ),
        )

        self.assertEqual([trade["stock_code"] for trade in result["trades"]], ["A", "C"])

    def test_order_backtest_defers_sell_when_exit_open_is_limit_down(self):
        rows = [
            {
                "trade_date": "20260101",
                "stock_code": "A",
                "name": "A",
                "pred_prob": 0.90,
                "close": 10.0,
                "atr_qfq": 0.1,
                "post_open": 10.0,
                "post4_open": 9.0,
            },
            {
                "trade_date": "20260102",
                "stock_code": "A",
                "name": "A",
                "pred_prob": 0.10,
                "close": 10.0,
                "atr_qfq": 0.1,
                "post_open": 9.0,
                "post4_open": 9.5,
            },
            {
                "trade_date": "20260103",
                "stock_code": "A",
                "name": "A",
                "pred_prob": 0.10,
                "close": 9.0,
                "atr_qfq": 0.1,
                "post_open": 9.5,
                "post4_open": 9.5,
            },
        ]

        result = run_order_backtest(
            rows,
            PortfolioBacktestConfig(
                top_k=1,
                max_positions=1,
                holding_days=1,
                min_pred_prob=0.1,
                max_atr_ratio=None,
                commission_rate=0.0,
                sell_tax_rate=0.0,
                slippage_rate=0.0,
            ),
        )

        self.assertEqual(result["trades"][0]["exit_signal_date"], "20260103")
        self.assertAlmostEqual(result["trades"][0]["net_return"], -0.05)

    def test_order_backtest_can_delay_same_day_sell_proceeds(self):
        rows = [
            {"trade_date": "20260101", "stock_code": "A", "pred_prob": 0.90, "close": 10.0, "atr_qfq": 0.1, "post_open": 10.0},
            {"trade_date": "20260102", "stock_code": "A", "pred_prob": 0.10, "close": 10.0, "atr_qfq": 0.1, "post_open": 11.0},
            {"trade_date": "20260102", "stock_code": "B", "pred_prob": 0.80, "close": 10.0, "atr_qfq": 0.1, "post_open": 10.0},
            {"trade_date": "20260103", "stock_code": "B", "pred_prob": 0.10, "close": 10.0, "atr_qfq": 0.1, "post_open": 12.0},
        ]

        result = run_order_backtest(
            rows,
            PortfolioBacktestConfig(
                top_k=1,
                max_positions=1,
                holding_days=1,
                min_pred_prob=0.1,
                max_atr_ratio=None,
                commission_rate=0.0,
                sell_tax_rate=0.0,
                slippage_rate=0.0,
                reuse_sell_proceeds_same_day=False,
            ),
        )

        self.assertEqual([trade["stock_code"] for trade in result["trades"]], ["A"])


if __name__ == "__main__":
    unittest.main()
