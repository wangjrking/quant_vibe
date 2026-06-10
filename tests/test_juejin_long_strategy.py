import importlib.util
import sys
import types
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


STRATEGY_PATH = Path(r"D:\work\dfcf\juejin\strategy\fb9d4d71-6198-11f1-8a7e-10ffe0295517\main.py")


def _load_strategy_module():
    fake_gm = types.ModuleType("gm")
    fake_api = types.ModuleType("gm.api")
    fake_api.OrderType_Market = "market"
    fake_api.OrderType_Limit = "limit"
    fake_api.PositionSide_Long = "long"
    fake_api.MODE_BACKTEST = "backtest"
    fake_api.ADJUST_PREV = "prev"
    fake_api.ADJUST_NONE = "none"
    fake_api.ADJUST_POST = "post"
    fake_api.OrderSide_Buy = "buy"
    fake_api.OrderSide_Sell = "sell"
    fake_api.PositionEffect_Open = "open"
    fake_api.PositionEffect_Close = "close"
    fake_api.order_target_percent = lambda **kwargs: kwargs
    fake_api.order_volume = lambda **kwargs: kwargs
    fake_api.schedule = lambda **kwargs: None
    fake_api.run = lambda **kwargs: None
    sys.modules["gm"] = fake_gm
    sys.modules["gm.api"] = fake_api

    spec = importlib.util.spec_from_file_location("juejin_long_strategy_test", STRATEGY_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakeAccount:
    def __init__(self, positions, cash=None):
        self._positions = positions
        self.cash = cash or {}

    def positions(self):
        return list(self._positions)


class JuejinLongStrategyTests(unittest.TestCase):
    def setUp(self):
        self.module = _load_strategy_module()

    def test_trade_daily_exits_on_stop_loss_before_holding_period(self):
        context = SimpleNamespace(
            now=datetime(2026, 6, 8, 9, 35, 0),
            trade_index=4,
            held_symbols={"SZSE.000001": {"entry_index": 0, "buy_date": "20260601"}},
            signals_by_buy_date={},
            account=lambda: _FakeAccount([{"symbol": "SZSE.000001", "fpnl_ratio": -0.12}]),
        )

        with patch.object(self.module, "STOP_LOSS_PCT", 0.10), patch.object(self.module, "TAKE_PROFIT_PCT", None), patch.object(
            self.module, "order_target_percent"
        ) as order_target_percent:
            self.module.trade_daily(context)

        order_target_percent.assert_called_once()
        self.assertEqual(order_target_percent.call_args.kwargs["symbol"], "SZSE.000001")
        self.assertEqual(order_target_percent.call_args.kwargs["percent"], 0)
        self.assertEqual(context.held_symbols, {})

    def test_trade_daily_exits_on_take_profit_before_holding_period(self):
        context = SimpleNamespace(
            now=datetime(2026, 6, 8, 9, 35, 0),
            trade_index=4,
            held_symbols={"SZSE.000002": {"entry_index": 0, "buy_date": "20260601"}},
            signals_by_buy_date={},
            account=lambda: _FakeAccount([{"symbol": "SZSE.000002", "fpnl_ratio": 0.22}]),
        )

        with patch.object(self.module, "STOP_LOSS_PCT", None), patch.object(self.module, "TAKE_PROFIT_PCT", 0.20), patch.object(
            self.module, "order_target_percent"
        ) as order_target_percent:
            self.module.trade_daily(context)

        order_target_percent.assert_called_once()
        self.assertEqual(order_target_percent.call_args.kwargs["symbol"], "SZSE.000002")
        self.assertEqual(order_target_percent.call_args.kwargs["percent"], 0)
        self.assertEqual(context.held_symbols, {})

    def test_trade_daily_caps_new_order_by_available_cash(self):
        context = SimpleNamespace(
            now=datetime(2026, 6, 8, 9, 35, 0),
            trade_index=0,
            held_symbols={},
            signals_by_buy_date={
                "20260608": [
                    {
                        "symbol": "SZSE.000001",
                        "target_pct": "0.32666666666666666",
                        "signal_date": "20260607",
                        "pred_prob": "0.02",
                    }
                ]
            },
            pred_scores_by_date={},
            pred_ranks_by_date={},
            account=lambda: _FakeAccount([], cash={"available": 250000.0, "nav": 1000000.0}),
            market_rows_by_date={"20260608": {"000001.SZ": {"stock_code": "000001.SZ", "open": 10.0, "pre_close": 9.8}}},
        )

        with patch.object(self.module, "order_volume") as order_volume:
            self.module.trade_daily(context)

        order_volume.assert_called_once()
        self.assertEqual(order_volume.call_args.kwargs["symbol"], "SZSE.000001")
        self.assertEqual(order_volume.call_args.kwargs["volume"], 24700)
        self.assertEqual(order_volume.call_args.kwargs["price"], 10.0)

    def test_resolve_backtest_adjust_prefers_none_mode(self):
        with patch.object(self.module, "BACKTEST_ADJUST_MODE", "none"):
            self.assertEqual(self.module._resolve_backtest_adjust(), "none")

    def test_trade_daily_skips_buy_when_open_limit_up(self):
        context = SimpleNamespace(
            now=datetime(2026, 6, 8, 9, 35, 0),
            trade_index=0,
            held_symbols={},
            signals_by_buy_date={
                "20260608": [
                    {
                        "symbol": "SZSE.000001",
                        "target_pct": "0.25",
                        "signal_date": "20260607",
                        "pred_prob": "0.02",
                    }
                ]
            },
            pred_scores_by_date={},
            pred_ranks_by_date={},
            account=lambda: _FakeAccount([], cash={"available": 250000.0, "nav": 1000000.0}),
            market_rows_by_date={"20260608": {"000001.SZ": {"stock_code": "000001.SZ", "open": 11.0, "pre_close": 10.0}}},
        )

        with patch.object(self.module, "order_volume") as order_volume:
            self.module.trade_daily(context)

        order_volume.assert_not_called()
        self.assertEqual(context.held_symbols, {})

    def test_trade_daily_skips_sell_when_open_limit_down(self):
        context = SimpleNamespace(
            now=datetime(2026, 6, 8, 9, 35, 0),
            trade_index=3,
            held_symbols={"SZSE.000008": {"entry_index": 0, "buy_date": "20260601", "holding_days": 3}},
            signals_by_buy_date={},
            pred_scores_by_date={},
            pred_ranks_by_date={},
            account=lambda: _FakeAccount([{"symbol": "SZSE.000008", "volume": 1000}], cash={"available": 0.0, "nav": 1000000.0}),
            market_rows_by_date={"20260608": {"000008.SZ": {"stock_code": "000008.SZ", "open": 9.0, "pre_close": 10.0}}},
        )

        with patch.object(self.module, "order_volume") as order_volume:
            self.module.trade_daily(context)

        order_volume.assert_not_called()
        self.assertIn("SZSE.000008", context.held_symbols)

    def test_init_preloads_market_rows(self):
        context = SimpleNamespace()
        with patch.object(self.module, "SIGNAL_FILE", str(STRATEGY_PATH)), patch.object(
            self.module, "_load_signals", return_value={"20260608": [{"stock_code": "000001.SZ", "symbol": "SZSE.000001"}]}
        ), patch.object(self.module, "_preload_market_rows") as preload_market_rows:
            self.module.init(context)

        preload_market_rows.assert_called_once_with(context)
        self.assertEqual(context.trade_index, -1)

    def test_extract_position_return_prefers_fpnl_ratio(self):
        position = {"symbol": "SZSE.000001", "fpnl_ratio": 0.123}
        self.assertAlmostEqual(self.module._extract_position_return(position), 0.123)

    def test_trade_daily_exits_on_low_score_for_losing_position(self):
        context = SimpleNamespace(
            now=datetime(2026, 6, 8, 9, 35, 0),
            trade_index=4,
            held_symbols={"SZSE.000003": {"entry_index": 0, "buy_date": "20260601"}},
            signals_by_buy_date={},
            pred_scores_by_date={"20260608": {"SZSE.000003": 0.004}},
            account=lambda: _FakeAccount([{"symbol": "SZSE.000003", "fpnl_ratio": -0.03}]),
        )

        with patch.object(self.module, "STOP_LOSS_PCT", None), patch.object(self.module, "TAKE_PROFIT_PCT", None), patch.object(
            self.module, "SCORE_STOP_LOSS_PRED", 0.005
        ), patch.object(self.module, "SCORE_TAKE_PROFIT_PRED", None), patch.object(
            self.module, "order_target_percent"
        ) as order_target_percent:
            self.module.trade_daily(context)

        order_target_percent.assert_called_once()
        self.assertEqual(order_target_percent.call_args.kwargs["symbol"], "SZSE.000003")
        self.assertEqual(context.held_symbols, {})

    def test_trade_daily_exits_on_low_score_for_profitable_position(self):
        context = SimpleNamespace(
            now=datetime(2026, 6, 8, 9, 35, 0),
            trade_index=4,
            held_symbols={"SZSE.000004": {"entry_index": 0, "buy_date": "20260601", "entry_pred_prob": 0.02}},
            signals_by_buy_date={},
            pred_scores_by_date={"20260608": {"SZSE.000004": 0.009}},
            account=lambda: _FakeAccount([{"symbol": "SZSE.000004", "fpnl_ratio": 0.06}]),
        )

        with patch.object(self.module, "STOP_LOSS_PCT", None), patch.object(self.module, "TAKE_PROFIT_PCT", None), patch.object(
            self.module, "SCORE_STOP_LOSS_PRED", None
        ), patch.object(self.module, "SCORE_TAKE_PROFIT_PRED", 0.01), patch.object(
            self.module, "order_target_percent"
        ) as order_target_percent:
            self.module.trade_daily(context)

        order_target_percent.assert_called_once()
        self.assertEqual(order_target_percent.call_args.kwargs["symbol"], "SZSE.000004")
        self.assertEqual(context.held_symbols, {})

    def test_trade_daily_exits_when_score_drops_below_entry_ratio_for_loser(self):
        context = SimpleNamespace(
            now=datetime(2026, 6, 8, 9, 35, 0),
            trade_index=4,
            held_symbols={"SZSE.000005": {"entry_index": 0, "buy_date": "20260601", "entry_pred_prob": 0.020}},
            signals_by_buy_date={},
            pred_scores_by_date={"20260608": {"SZSE.000005": 0.005}},
            account=lambda: _FakeAccount([{"symbol": "SZSE.000005", "fpnl_ratio": -0.02}]),
        )

        with patch.object(self.module, "STOP_LOSS_PCT", None), patch.object(self.module, "TAKE_PROFIT_PCT", None), patch.object(
            self.module, "SCORE_STOP_LOSS_PRED", None
        ), patch.object(self.module, "SCORE_TAKE_PROFIT_PRED", None), patch.object(
            self.module, "SCORE_STOP_LOSS_RATIO", 0.4
        ), patch.object(self.module, "SCORE_TAKE_PROFIT_RATIO", None), patch.object(
            self.module, "order_target_percent"
        ) as order_target_percent:
            self.module.trade_daily(context)

        order_target_percent.assert_called_once()
        self.assertEqual(order_target_percent.call_args.kwargs["symbol"], "SZSE.000005")
        self.assertEqual(context.held_symbols, {})

    def test_trade_daily_exits_when_rank_drops_for_loser(self):
        context = SimpleNamespace(
            now=datetime(2026, 6, 8, 9, 35, 0),
            trade_index=4,
            held_symbols={"SZSE.000006": {"entry_index": 0, "buy_date": "20260601"}},
            signals_by_buy_date={},
            pred_scores_by_date={"20260608": {"SZSE.000006": 0.006}},
            pred_ranks_by_date={"20260608": {"SZSE.000006": 0.18}},
            account=lambda: _FakeAccount([{"symbol": "SZSE.000006", "fpnl_ratio": -0.01}]),
        )

        with patch.object(self.module, "STOP_LOSS_PCT", None), patch.object(self.module, "TAKE_PROFIT_PCT", None), patch.object(
            self.module, "SCORE_STOP_LOSS_RANK", 0.20
        ), patch.object(self.module, "order_target_percent") as order_target_percent:
            self.module.trade_daily(context)

        order_target_percent.assert_called_once()
        self.assertEqual(order_target_percent.call_args.kwargs["symbol"], "SZSE.000006")
        self.assertEqual(context.held_symbols, {})

    def test_trade_daily_exits_when_score_drops_vs_previous_day_for_loser(self):
        context = SimpleNamespace(
            now=datetime(2026, 6, 8, 9, 35, 0),
            trade_index=4,
            held_symbols={"SZSE.000007": {"entry_index": 0, "buy_date": "20260601", "last_pred_prob": 0.020}},
            signals_by_buy_date={},
            pred_scores_by_date={"20260608": {"SZSE.000007": 0.007}},
            account=lambda: _FakeAccount([{"symbol": "SZSE.000007", "fpnl_ratio": -0.02}]),
        )

        with patch.object(self.module, "STOP_LOSS_PCT", None), patch.object(self.module, "TAKE_PROFIT_PCT", None), patch.object(
            self.module, "SCORE_STOP_LOSS_DAY_DROP_RATIO", 0.40
        ), patch.object(self.module, "order_target_percent") as order_target_percent:
            self.module.trade_daily(context)

        order_target_percent.assert_called_once()
        self.assertEqual(order_target_percent.call_args.kwargs["symbol"], "SZSE.000007")
        self.assertEqual(context.held_symbols, {})

    def test_trade_daily_uses_per_signal_holding_days_over_global_default(self):
        context = SimpleNamespace(
            now=datetime(2026, 6, 8, 9, 35, 0),
            trade_index=3,
            held_symbols={"SZSE.000008": {"entry_index": 0, "buy_date": "20260601", "holding_days": 3}},
            signals_by_buy_date={},
            pred_scores_by_date={},
            pred_ranks_by_date={},
            account=lambda: _FakeAccount([{"symbol": "SZSE.000008", "fpnl_ratio": 0.01}]),
        )

        with patch.object(self.module, "STOP_LOSS_PCT", None), patch.object(self.module, "TAKE_PROFIT_PCT", None), patch.object(
            self.module, "order_target_percent"
        ) as order_target_percent:
            self.module.trade_daily(context)

        order_target_percent.assert_called_once()
        self.assertEqual(order_target_percent.call_args.kwargs["symbol"], "SZSE.000008")
        self.assertEqual(context.held_symbols, {})

    def test_trade_daily_does_not_exit_on_entry_score_decay_before_min_hold_days(self):
        context = SimpleNamespace(
            now=datetime(2026, 6, 8, 9, 35, 0),
            trade_index=0,
            held_symbols={
                "SZSE.000009": {
                    "entry_index": 0,
                    "buy_date": "20260607",
                    "entry_pred_prob": 0.020,
                }
            },
            signals_by_buy_date={},
            pred_scores_by_date={"20260608": {"SZSE.000009": 0.010}},
            pred_ranks_by_date={},
            account=lambda: _FakeAccount([{"symbol": "SZSE.000009", "fpnl_ratio": -0.01}]),
        )

        with patch.object(self.module, "STOP_LOSS_PCT", None), patch.object(self.module, "TAKE_PROFIT_PCT", None), patch.object(
            self.module, "SCORE_EXIT_ENTRY_RATIO", 0.7
        ), patch.object(self.module, "MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT", 2), patch.object(
            self.module, "order_target_percent"
        ) as order_target_percent:
            self.module.trade_daily(context)

        order_target_percent.assert_not_called()
        self.assertIn("SZSE.000009", context.held_symbols)
        self.assertEqual(context.held_symbols["SZSE.000009"]["last_pred_prob"], 0.010)

    def test_trade_daily_exits_on_entry_score_decay_after_min_hold_days(self):
        context = SimpleNamespace(
            now=datetime(2026, 6, 9, 9, 35, 0),
            trade_index=1,
            held_symbols={
                "SZSE.000010": {
                    "entry_index": 0,
                    "buy_date": "20260607",
                    "entry_pred_prob": 0.020,
                }
            },
            signals_by_buy_date={},
            pred_scores_by_date={"20260609": {"SZSE.000010": 0.010}},
            pred_ranks_by_date={},
            account=lambda: _FakeAccount([{"symbol": "SZSE.000010", "fpnl_ratio": -0.01}]),
        )

        with patch.object(self.module, "STOP_LOSS_PCT", None), patch.object(self.module, "TAKE_PROFIT_PCT", None), patch.object(
            self.module, "SCORE_EXIT_ENTRY_RATIO", 0.7
        ), patch.object(self.module, "MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT", 2), patch.object(
            self.module, "order_target_percent"
        ) as order_target_percent:
            self.module.trade_daily(context)

        order_target_percent.assert_called_once()
        self.assertEqual(order_target_percent.call_args.kwargs["symbol"], "SZSE.000010")
        self.assertEqual(context.held_symbols, {})

    def test_trade_daily_extends_holding_when_score_stays_strong_at_base_horizon(self):
        context = SimpleNamespace(
            now=datetime(2026, 6, 10, 9, 35, 0),
            trade_index=2,
            held_symbols={
                "SZSE.000011": {
                    "entry_index": 0,
                    "buy_date": "20260608",
                    "holding_days": 3,
                    "entry_pred_prob": 0.020,
                }
            },
            signals_by_buy_date={},
            pred_scores_by_date={"20260610": {"SZSE.000011": 0.018}},
            pred_ranks_by_date={},
            account=lambda: _FakeAccount([{"symbol": "SZSE.000011", "fpnl_ratio": 0.01}]),
        )

        with patch.object(self.module, "STOP_LOSS_PCT", None), patch.object(self.module, "TAKE_PROFIT_PCT", None), patch.object(
            self.module, "SCORE_CONTINUE_ENTRY_RATIO", 0.85
        ), patch.object(self.module, "MAX_HOLDING_DAYS", 5), patch.object(
            self.module, "order_target_percent"
        ) as order_target_percent:
            self.module.trade_daily(context)

        order_target_percent.assert_not_called()
        self.assertIn("SZSE.000011", context.held_symbols)
        self.assertEqual(context.held_symbols["SZSE.000011"]["last_pred_prob"], 0.018)

    def test_trade_daily_exits_at_base_horizon_when_score_not_strong_enough(self):
        context = SimpleNamespace(
            now=datetime(2026, 6, 10, 9, 35, 0),
            trade_index=2,
            held_symbols={
                "SZSE.000012": {
                    "entry_index": 0,
                    "buy_date": "20260608",
                    "holding_days": 3,
                    "entry_pred_prob": 0.020,
                }
            },
            signals_by_buy_date={},
            pred_scores_by_date={"20260610": {"SZSE.000012": 0.012}},
            pred_ranks_by_date={},
            account=lambda: _FakeAccount([{"symbol": "SZSE.000012", "fpnl_ratio": 0.01}]),
        )

        with patch.object(self.module, "STOP_LOSS_PCT", None), patch.object(self.module, "TAKE_PROFIT_PCT", None), patch.object(
            self.module, "SCORE_CONTINUE_ENTRY_RATIO", 0.85
        ), patch.object(self.module, "MAX_HOLDING_DAYS", 5), patch.object(
            self.module, "order_target_percent"
        ) as order_target_percent:
            self.module.trade_daily(context)

        order_target_percent.assert_called_once()
        self.assertEqual(order_target_percent.call_args.kwargs["symbol"], "SZSE.000012")
        self.assertEqual(context.held_symbols, {})

    def test_trade_daily_does_not_exit_on_light_stop_before_min_hold_days(self):
        context = SimpleNamespace(
            now=datetime(2026, 6, 8, 9, 35, 0),
            trade_index=0,
            held_symbols={
                "SZSE.000013": {
                    "entry_index": 0,
                    "buy_date": "20260607",
                    "entry_pred_prob": 0.020,
                }
            },
            signals_by_buy_date={},
            pred_scores_by_date={"20260608": {"SZSE.000013": 0.020}},
            pred_ranks_by_date={},
            account=lambda: _FakeAccount([{"symbol": "SZSE.000013", "fpnl_ratio": -0.07}]),
        )

        with patch.object(self.module, "STOP_LOSS_PCT", None), patch.object(self.module, "TAKE_PROFIT_PCT", None), patch.object(
            self.module, "LIGHT_STOP_LOSS_PCT", 0.06
        ), patch.object(self.module, "MIN_HOLDING_DAYS_BEFORE_LIGHT_STOP", 2), patch.object(
            self.module, "order_target_percent"
        ) as order_target_percent:
            self.module.trade_daily(context)

        order_target_percent.assert_not_called()
        self.assertIn("SZSE.000013", context.held_symbols)

    def test_trade_daily_exits_on_light_stop_after_min_hold_days(self):
        context = SimpleNamespace(
            now=datetime(2026, 6, 9, 9, 35, 0),
            trade_index=1,
            held_symbols={
                "SZSE.000014": {
                    "entry_index": 0,
                    "buy_date": "20260607",
                    "entry_pred_prob": 0.020,
                }
            },
            signals_by_buy_date={},
            pred_scores_by_date={"20260609": {"SZSE.000014": 0.020}},
            pred_ranks_by_date={},
            account=lambda: _FakeAccount([{"symbol": "SZSE.000014", "fpnl_ratio": -0.07}]),
        )

        with patch.object(self.module, "STOP_LOSS_PCT", None), patch.object(self.module, "TAKE_PROFIT_PCT", None), patch.object(
            self.module, "LIGHT_STOP_LOSS_PCT", 0.06
        ), patch.object(self.module, "MIN_HOLDING_DAYS_BEFORE_LIGHT_STOP", 2), patch.object(
            self.module, "order_target_percent"
        ) as order_target_percent:
            self.module.trade_daily(context)

        order_target_percent.assert_called_once()
        self.assertEqual(order_target_percent.call_args.kwargs["symbol"], "SZSE.000014")
        self.assertEqual(context.held_symbols, {})

    def test_trade_daily_caps_signal_target_pct_by_runtime_target(self):
        context = SimpleNamespace(
            now=datetime(2026, 6, 8, 9, 35, 0),
            trade_index=-1,
            held_symbols={},
            signals_by_buy_date={
                "20260608": [
                    {
                        "symbol": "SZSE.000015",
                        "target_pct": 0.98,
                        "pred_prob": 0.02,
                        "holding_days": 3,
                    }
                ]
            },
            pred_scores_by_date={},
            pred_ranks_by_date={},
            account=lambda: _FakeAccount([]),
        )

        with patch.object(self.module, "TARGET_POSITION_PCT", 0.33), patch.object(
            self.module, "order_target_percent"
        ) as order_target_percent:
            self.module.trade_daily(context)

        order_target_percent.assert_called_once()
        self.assertEqual(order_target_percent.call_args.kwargs["symbol"], "SZSE.000015")
        self.assertAlmostEqual(order_target_percent.call_args.kwargs["percent"], 0.33)


if __name__ == "__main__":
    unittest.main()
