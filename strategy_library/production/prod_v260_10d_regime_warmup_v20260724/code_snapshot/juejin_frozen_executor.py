# coding=utf-8
from __future__ import absolute_import, print_function

import csv
import os

from gm.api import *


STRATEGY_ID = "ce327750-634e-11f1-b8d7-10ffe0295517"
ACTION_FILE = os.environ["GM_SIGNAL_FILE"]
BACKTEST_START = os.environ["GM_BACKTEST_START"]
BACKTEST_END = os.environ["GM_BACKTEST_END"]
BACKTEST_INITIAL_CASH = float(os.environ.get("GM_BACKTEST_INITIAL_CASH", "700000"))
EXECUTION_SLIPPAGE_RATIO = float(os.environ.get("GM_BACKTEST_SLIPPAGE_RATIO", "0.003"))


def _to_symbol(stock_code):
    code, suffix = str(stock_code).split(".", 1)
    return "{}.{}".format({"SH": "SHSE", "SZ": "SZSE"}[suffix], code)


def _load_actions(path):
    actions = {}
    with open(path, newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            buy_date = str(row.get("buy_date") or "").strip()
            action = str(row.get("action") or "").strip().upper()
            stock_code = str(row.get("stock_code") or "").strip()
            if not buy_date or action not in {"BUY", "SELL"} or not stock_code:
                continue
            actions.setdefault(buy_date, []).append(
                {
                    "action": action,
                    "stock_code": stock_code,
                    "symbol": _to_symbol(stock_code),
                    "target_pct": float(row.get("target_pct") or 0.0),
                    "execution_open_raw": float(row.get("execution_open_raw") or 0.0),
                }
            )
    for rows in actions.values():
        rows.sort(key=lambda item: (0 if item["action"] == "SELL" else 1, item["stock_code"]))
    return actions


def init(context):
    context.actions = _load_actions(ACTION_FILE)
    print("FROZEN_ACTION_FILE {}".format(ACTION_FILE))
    print("SLIPPAGE_CONTRACT open_price_only={:.6f} engine=0".format(EXECUTION_SLIPPAGE_RATIO))
    schedule(schedule_func=execute_sells, date_rule="1d", time_rule="09:30:00")
    schedule(schedule_func=execute_buys, date_rule="1d", time_rule="09:30:01")


def execute_sells(context):
    trade_date = context.now.strftime("%Y%m%d")
    positions = {item.get("symbol"): item for item in context.account().positions()}
    for row in context.actions.get(trade_date, []):
        if row["action"] != "SELL":
            continue
        position = positions.get(row["symbol"])
        volume = int(float((position or {}).get("volume") or 0))
        if volume <= 0 or row["execution_open_raw"] <= 0:
            print("FROZEN_SELL_SKIP {} {} reason=no_position_or_open".format(trade_date, row["stock_code"]))
            continue
        price = row["execution_open_raw"] * (1.0 - EXECUTION_SLIPPAGE_RATIO)
        print("FROZEN_SELL {} {} {:.8f}".format(trade_date, row["stock_code"], price))
        order_volume(
            symbol=row["symbol"], volume=volume, side=OrderSide_Sell,
            order_type=OrderType_Limit, position_effect=PositionEffect_Close, price=price,
        )


def execute_buys(context):
    trade_date = context.now.strftime("%Y%m%d")
    for row in context.actions.get(trade_date, []):
        if row["action"] != "BUY":
            continue
        cash = context.account().cash
        if callable(cash):
            cash = cash()
        available = float(cash.get("available") or cash.get("cash") or 0.0)
        nav = float(cash.get("nav") or cash.get("asset") or cash.get("total_asset") or available)
        price = row["execution_open_raw"] * (1.0 + EXECUTION_SLIPPAGE_RATIO)
        target_value = min(row["target_pct"] * nav, available * 0.98)
        volume = int(target_value / price / 100.0) * 100 if price > 0 else 0
        if volume < 100:
            print("FROZEN_BUY_SKIP {} {} reason=volume_lt_100".format(trade_date, row["stock_code"]))
            continue
        print("FROZEN_BUY {} {} {:.8f}".format(trade_date, row["stock_code"], price))
        order_volume(
            symbol=row["symbol"], volume=volume, side=OrderSide_Buy,
            order_type=OrderType_Limit, position_effect=PositionEffect_Open, price=price,
        )


def on_execution_report(context, execrpt):
    print("FROZEN_EXECUTION {}".format(execrpt))


def on_backtest_finished(context, indicator=None, *args, **kwargs):
    print("GM_BACKTEST_FINISHED frozen_action_executor_open_price_slippage_v1")
    if indicator is not None:
        print("GM_BACKTEST_INDICATOR: {}".format(indicator))


if __name__ == "__main__":
    run(
        strategy_id=STRATEGY_ID,
        filename="main.py",
        mode=MODE_BACKTEST,
        token=os.environ.get("GM_TOKEN", ""),
        backtest_start_time=BACKTEST_START,
        backtest_end_time=BACKTEST_END,
        backtest_adjust=ADJUST_NONE,
        backtest_initial_cash=BACKTEST_INITIAL_CASH,
        backtest_commission_ratio=0.0003,
        backtest_slippage_ratio=0.0,
        backtest_match_mode=0,
    )
