from __future__ import absolute_import, print_function

import os
from datetime import datetime

from gm.api import *


STRATEGY_ID = os.environ.get("GM_PROBE_STRATEGY_ID", "fb9d4d71-6198-11f1-8a7e-10ffe0295517")
PROBE_SYMBOL = os.environ.get("GM_PROBE_SYMBOL", "SZSE.000001")
PROBE_BUY_DATE = os.environ.get("GM_PROBE_BUY_DATE", "2024-06-06")
PROBE_SELL_DATE = os.environ.get("GM_PROBE_SELL_DATE", "2024-06-10")
PROBE_BUY_PRICE = float(os.environ.get("GM_PROBE_BUY_PRICE", "10.0"))
PROBE_SELL_PRICE = float(os.environ.get("GM_PROBE_SELL_PRICE", "10.5"))
PROBE_VOLUME = int(os.environ.get("GM_PROBE_VOLUME", "100"))
BACKTEST_START = os.environ.get("GM_BACKTEST_START", "2024-06-06 09:00:00")
BACKTEST_END = os.environ.get("GM_BACKTEST_END", "2024-06-20 15:30:00")


def _date_key(value):
    return value.strftime("%Y-%m-%d")


def init(context):
    context.has_bought = False
    context.has_sold = False
    schedule(schedule_func=trade_daily, date_rule="1d", time_rule="09:35:00")
    print("LIMIT_PROBE_INIT symbol={} buy_date={} sell_date={}".format(PROBE_SYMBOL, PROBE_BUY_DATE, PROBE_SELL_DATE))


def trade_daily(context):
    today = _date_key(context.now)
    if not context.has_bought and today == PROBE_BUY_DATE:
        print("LIMIT_PROBE_BUY {} {} {}".format(today, PROBE_SYMBOL, PROBE_BUY_PRICE))
        order_volume(
            symbol=PROBE_SYMBOL,
            volume=PROBE_VOLUME,
            side=OrderSide_Buy,
            order_type=OrderType_Limit,
            position_effect=PositionEffect_Open,
            price=PROBE_BUY_PRICE,
        )
        context.has_bought = True
        return

    if context.has_bought and not context.has_sold and today == PROBE_SELL_DATE:
        print("LIMIT_PROBE_SELL {} {} {}".format(today, PROBE_SYMBOL, PROBE_SELL_PRICE))
        order_volume(
            symbol=PROBE_SYMBOL,
            volume=PROBE_VOLUME,
            side=OrderSide_Sell,
            order_type=OrderType_Limit,
            position_effect=PositionEffect_Close,
            price=PROBE_SELL_PRICE,
        )
        context.has_sold = True


def on_backtest_finished(context, indicator=None, *args, **kwargs):
    print("LIMIT_PROBE_FINISHED")
    if indicator is not None:
        print("LIMIT_PROBE_INDICATOR: {}".format(indicator))


if __name__ == "__main__":
    run(
        strategy_id=STRATEGY_ID,
        filename="juejin_limit_probe.py",
        mode=MODE_BACKTEST,
        token=os.environ.get("GM_TOKEN", ""),
        backtest_start_time=BACKTEST_START,
        backtest_end_time=BACKTEST_END,
        backtest_adjust=globals().get("ADJUST_NONE", ADJUST_PREV),
        backtest_initial_cash=1000000,
        backtest_commission_ratio=0.0003,
        backtest_slippage_ratio=0.0001,
    )
