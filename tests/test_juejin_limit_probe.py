import importlib.util
import sys
import types
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PROBE_PATH = Path(__file__).resolve().parents[1] / "juejin_limit_probe.py"


def _load_probe_module():
    fake_gm = types.ModuleType("gm")
    fake_api = types.ModuleType("gm.api")
    fake_api.OrderType_Limit = "limit"
    fake_api.OrderSide_Buy = "buy"
    fake_api.OrderSide_Sell = "sell"
    fake_api.PositionEffect_Open = "open"
    fake_api.PositionEffect_Close = "close"
    fake_api.MODE_BACKTEST = "backtest"
    fake_api.ADJUST_PREV = "prev"
    fake_api.ADJUST_NONE = "none"
    fake_api.order_volume = lambda **kwargs: kwargs
    fake_api.schedule = lambda **kwargs: None
    fake_api.run = lambda **kwargs: None
    sys.modules["gm"] = fake_gm
    sys.modules["gm.api"] = fake_api

    spec = importlib.util.spec_from_file_location("juejin_limit_probe_test", PROBE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class JuejinLimitProbeTests(unittest.TestCase):
    def setUp(self):
        self.module = _load_probe_module()

    def test_trade_daily_places_limit_buy_on_buy_date(self):
        context = SimpleNamespace(now=datetime(2024, 6, 6, 9, 35, 0), has_bought=False, has_sold=False)
        with patch.object(self.module, "PROBE_BUY_DATE", "2024-06-06"), patch.object(
            self.module, "order_volume"
        ) as order_volume:
            self.module.trade_daily(context)

        order_volume.assert_called_once()
        self.assertEqual(order_volume.call_args.kwargs["side"], "buy")
        self.assertTrue(context.has_bought)

    def test_trade_daily_places_limit_sell_on_sell_date(self):
        context = SimpleNamespace(now=datetime(2024, 6, 10, 9, 35, 0), has_bought=True, has_sold=False)
        with patch.object(self.module, "PROBE_SELL_DATE", "2024-06-10"), patch.object(
            self.module, "order_volume"
        ) as order_volume:
            self.module.trade_daily(context)

        order_volume.assert_called_once()
        self.assertEqual(order_volume.call_args.kwargs["side"], "sell")
        self.assertTrue(context.has_sold)


if __name__ == "__main__":
    unittest.main()
