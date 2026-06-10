import unittest
import sqlite3
import tempfile
from pathlib import Path

from backtest_module import BacktestConfig, calculate_net_return, read_prediction_rows, resolve_slippage_rate, run_backtest


class BacktestModuleTests(unittest.TestCase):
    def test_calculate_net_return_deducts_round_trip_costs(self):
        net_return = calculate_net_return(
            buy_price=10.0,
            sell_price=11.0,
            commission_rate=0.001,
            sell_tax_rate=0.001,
            slippage_rate=0.001,
        )

        self.assertAlmostEqual(net_return, 0.094511, places=6)

    def test_run_backtest_blocks_unbuyable_limit_open(self):
        rows = [
            {
                "trade_date": "20260105",
                "stock_code": "600001.SH",
                "name": "A",
                "pred_prob": 0.20,
                "close": 10.0,
                "post_open": 11.10,
                "post2_open": 11.30,
            },
            {
                "trade_date": "20260105",
                "stock_code": "600002.SH",
                "name": "B",
                "pred_prob": 0.10,
                "close": 10.0,
                "post_open": 10.20,
                "post2_open": 10.50,
            },
        ]

        result = run_backtest(rows, BacktestConfig(top_k=2, skip_limit_up_open=True, commission_rate=0.0, sell_tax_rate=0.0, slippage_rate=0.0))

        self.assertEqual(result.trades[0]["stock_code"], "600002.SH")
        self.assertEqual(result.metrics["trade_count"], 1)
        self.assertAlmostEqual(result.metrics["cumulative_return"], 0.0294117647)

    def test_run_backtest_reports_rank_ic(self):
        rows = [
            {"trade_date": "20260105", "stock_code": "A", "pred_prob": 0.3, "close": 10.0, "post_open": 10.0, "post2_open": 10.2, "10d_yield_rate": 0.2},
            {"trade_date": "20260105", "stock_code": "B", "pred_prob": 0.2, "close": 10.0, "post_open": 10.0, "post2_open": 10.1, "10d_yield_rate": 0.1},
            {"trade_date": "20260105", "stock_code": "C", "pred_prob": 0.1, "close": 10.0, "post_open": 10.0, "post2_open": 9.9, "10d_yield_rate": -0.1},
        ]

        result = run_backtest(rows, BacktestConfig(top_k=1, commission_rate=0.0, sell_tax_rate=0.0, slippage_rate=0.0))

        self.assertAlmostEqual(result.metrics["mean_spearman_ic"], 1.0)

    def test_run_backtest_reports_negative_max_drawdown(self):
        rows = [
            {"trade_date": "20260105", "stock_code": "A", "pred_prob": 0.3, "close": 10.0, "post_open": 10.0, "post2_open": 12.0},
            {"trade_date": "20260106", "stock_code": "B", "pred_prob": 0.3, "close": 10.0, "post_open": 10.0, "post2_open": 9.0},
        ]

        result = run_backtest(rows, BacktestConfig(top_k=1, commission_rate=0.0, sell_tax_rate=0.0, slippage_rate=0.0))

        self.assertAlmostEqual(result.metrics["max_drawdown"], -0.1)

    def test_run_backtest_can_use_label_return_without_sell_price(self):
        rows = [
            {"trade_date": "20260105", "stock_code": "A", "pred_prob": 0.3, "close": 10.0, "10d_yield_rate": 0.20},
            {"trade_date": "20260105", "stock_code": "B", "pred_prob": 0.2, "close": 10.0, "10d_yield_rate": -0.10},
        ]

        result = run_backtest(rows, BacktestConfig(top_k=1, return_col="10d_yield_rate", commission_rate=0.0, sell_tax_rate=0.0, slippage_rate=0.0))

        self.assertEqual(result.metrics["trade_count"], 1)
        self.assertAlmostEqual(result.metrics["cumulative_return"], 0.2)
        self.assertEqual(result.trades[0]["stock_code"], "A")

    def test_run_backtest_filters_by_prediction_and_atr_ratio(self):
        rows = [
            {"trade_date": "20260105", "stock_code": "LOW", "pred_prob": 0.01, "close": 10.0, "atr_qfq": 0.1, "10d_yield_rate": 0.50},
            {"trade_date": "20260105", "stock_code": "RISK", "pred_prob": 0.30, "close": 10.0, "atr_qfq": 2.0, "10d_yield_rate": 0.50},
            {"trade_date": "20260105", "stock_code": "OK", "pred_prob": 0.20, "close": 10.0, "atr_qfq": 0.2, "10d_yield_rate": 0.10},
        ]

        result = run_backtest(
            rows,
            BacktestConfig(
                top_k=5,
                return_col="10d_yield_rate",
                min_pred_prob=0.02,
                max_atr_ratio=0.10,
                commission_rate=0.0,
                sell_tax_rate=0.0,
                slippage_rate=0.0,
            ),
        )

        self.assertEqual([trade["stock_code"] for trade in result.trades], ["OK"])

    def test_run_backtest_can_filter_by_liquidity(self):
        rows = [
            {"trade_date": "20260105", "stock_code": "LOW", "pred_prob": 0.30, "close": 10.0, "amount": 40000, "turnover_rate": 0.4, "10d_yield_rate": 0.50},
            {"trade_date": "20260105", "stock_code": "OK", "pred_prob": 0.20, "close": 10.0, "amount": 300000, "turnover_rate": 2.0, "10d_yield_rate": 0.10},
        ]

        result = run_backtest(
            rows,
            BacktestConfig(
                top_k=5,
                return_col="10d_yield_rate",
                min_amount=100000,
                min_turnover_rate=1.0,
                commission_rate=0.0,
                sell_tax_rate=0.0,
                slippage_rate=0.0,
            ),
        )

        self.assertEqual([trade["stock_code"] for trade in result.trades], ["OK"])

    def test_label_return_equity_is_scaled_by_holding_period(self):
        rows = [
            {"trade_date": "20260105", "stock_code": "A", "pred_prob": 0.3, "close": 10.0, "10d_yield_rate": 0.21},
        ]

        result = run_backtest(
            rows,
            BacktestConfig(
                top_k=1,
                return_col="10d_yield_rate",
                holding_period_days=2,
                commission_rate=0.0,
                sell_tax_rate=0.0,
                slippage_rate=0.0,
            ),
        )

        self.assertAlmostEqual(result.metrics["avg_trade_return"], 0.21)
        self.assertAlmostEqual(result.metrics["avg_daily_return"], 0.1)
        self.assertAlmostEqual(result.metrics["cumulative_return"], 0.1)

    def test_run_backtest_reports_annualized_metrics(self):
        rows = [
            {"trade_date": "20260105", "stock_code": "A", "pred_prob": 0.3, "close": 10.0, "10d_yield_rate": 0.10},
            {"trade_date": "20260106", "stock_code": "B", "pred_prob": 0.3, "close": 10.0, "10d_yield_rate": 0.10},
        ]

        result = run_backtest(
            rows,
            BacktestConfig(
                top_k=1,
                return_col="10d_yield_rate",
                commission_rate=0.0,
                sell_tax_rate=0.0,
                slippage_rate=0.0,
            ),
        )

        self.assertIn("annualized_return", result.metrics)
        self.assertIn("sharpe", result.metrics)
        self.assertGreater(result.metrics["annualized_return"], result.metrics["cumulative_return"])

    def test_resolve_slippage_rate_increases_for_illiquid_rows(self):
        config = BacktestConfig(slippage_rate=0.001, liquidity_slippage_enabled=True)
        liquid = {"amount": 8.0e5, "turnover_rate": 3.0}
        illiquid = {"amount": 4.0e4, "turnover_rate": 0.3}

        self.assertAlmostEqual(resolve_slippage_rate(liquid, config), 0.001)
        self.assertAlmostEqual(resolve_slippage_rate(illiquid, config), 0.004)

    def test_read_prediction_rows_enriches_liquidity_columns_from_stock_daily_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(db_path)
            conn.execute(
                """
                CREATE TABLE stock_predict_data_2d_yield_rate (
                    trade_date TEXT,
                    stock_code TEXT,
                    pred_prob REAL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE STOCK_DAILY_DATA (
                    trade_date TEXT,
                    stock_code TEXT,
                    amount REAL,
                    turnover_rate REAL,
                    turnover_rate_f REAL,
                    circ_mv REAL,
                    vol REAL,
                    industry TEXT
                )
                """
            )
            conn.execute("INSERT INTO stock_predict_data_2d_yield_rate VALUES ('20260105', '600001.SH', 0.2)")
            conn.execute("INSERT INTO STOCK_DAILY_DATA VALUES ('20260105', '600001.SH', 50000000, 0.8, 1.2, 1000000000, 12345, '电子')")
            conn.commit()
            conn.close()

            rows = read_prediction_rows(db_path, "stock_predict_data_2d_yield_rate")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["amount"], 50000000)
        self.assertEqual(rows[0]["turnover_rate"], 0.8)
        self.assertEqual(rows[0]["industry"], "电子")

    def test_read_prediction_rows_can_filter_by_stock_pool(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            pool_path = Path(tmpdir) / "pool.csv"
            conn = sqlite3.connect(db_path)
            conn.execute(
                """
                CREATE TABLE stock_predict_data_2d_yield_rate (
                    trade_date TEXT,
                    stock_code TEXT,
                    pred_prob REAL
                )
                """
            )
            conn.execute("INSERT INTO stock_predict_data_2d_yield_rate VALUES ('20260105', '600001.SH', 0.2)")
            conn.execute("INSERT INTO stock_predict_data_2d_yield_rate VALUES ('20260105', '000001.SZ', 0.3)")
            conn.commit()
            conn.close()

            pool_path.write_text("stock_code\n000001.SZ\n", encoding="utf-8")

            rows = read_prediction_rows(
                db_path,
                "stock_predict_data_2d_yield_rate",
                stock_pool_path=pool_path,
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["stock_code"], "000001.SZ")


if __name__ == "__main__":
    unittest.main()
