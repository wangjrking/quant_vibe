"""gm.api backtest entrypoint for the IC160 daily stock selection strategy.

The strategy consumes precomputed project signals. A row's signal_date is the
model decision day, and buy_date is the next trading day used for execution.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from gm.api import *  # noqa: F403


DEFAULT_SIGNAL_FILE = str((Path(__file__).resolve().parent / "data_file/gm_signals_2y_ic160.csv").resolve())
SIGNAL_FILE = os.environ.get("GM_SIGNAL_FILE", DEFAULT_SIGNAL_FILE)
MAX_POSITIONS = 10
HOLDING_DAYS = 10
TARGET_POSITION_PCT = 0.095


def _date_key(value) -> str:
    return value.strftime("%Y%m%d")


def _load_signals(path: str) -> dict[str, list[dict]]:
    signals_by_buy_date: dict[str, list[dict]] = {}
    with open(path, newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            buy_date = str(row.get("buy_date", "") or "")
            symbol = str(row.get("symbol", "") or "")
            if not buy_date or not symbol:
                continue
            signals_by_buy_date.setdefault(buy_date, []).append(row)
    for rows in signals_by_buy_date.values():
        rows.sort(key=lambda item: int(float(item.get("rank") or 999999)))
    return signals_by_buy_date


def init(context):
    signal_file = getattr(context, "signal_file", SIGNAL_FILE)
    context.signals_by_buy_date = _load_signals(signal_file)
    context.held_symbols = {}
    context.trade_index = -1
    schedule(schedule_func=trade_daily, date_rule="1d", time_rule="09:35:00")  # noqa: F405


def trade_daily(context):
    context.trade_index += 1
    today = _date_key(context.now)

    for symbol, state in list(context.held_symbols.items()):
        if context.trade_index - state["entry_index"] >= HOLDING_DAYS:
            order_target_percent(  # noqa: F405
                symbol=symbol,
                percent=0,
                order_type=OrderType_Market,  # noqa: F405
                position_side=PositionSide_Long,  # noqa: F405
            )
            del context.held_symbols[symbol]

    open_slots = max(MAX_POSITIONS - len(context.held_symbols), 0)
    if open_slots <= 0:
        return

    for signal in context.signals_by_buy_date.get(today, []):
        if open_slots <= 0:
            break
        symbol = signal["symbol"]
        if symbol in context.held_symbols:
            continue
        order_target_percent(  # noqa: F405
            symbol=symbol,
            percent=TARGET_POSITION_PCT,
            order_type=OrderType_Market,  # noqa: F405
            position_side=PositionSide_Long,  # noqa: F405
        )
        context.held_symbols[symbol] = {
            "entry_index": context.trade_index,
            "signal_date": signal.get("signal_date"),
            "buy_date": today,
            "rank": signal.get("rank"),
        }
        open_slots -= 1


def on_backtest_finished(context, indicator=None, *args, **kwargs):
    print("GM_BACKTEST_FINISHED")
    if indicator is not None:
        print(f"GM_BACKTEST_INDICATOR: {indicator}")
    if args:
        print(f"GM_BACKTEST_ARGS: {args}")
    if kwargs:
        print(f"GM_BACKTEST_KWARGS: {kwargs}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the IC160 strategy in gm.api backtest mode.")
    parser.add_argument("--signal-file", default=SIGNAL_FILE)
    parser.add_argument("--strategy-id", default=os.environ.get("GM_STRATEGY_ID", "quant_mcp_ic160_backtest"))
    parser.add_argument("--token", default=os.environ.get("GM_TOKEN"))
    parser.add_argument("--start", default="2024-06-05 09:00:00")
    parser.add_argument("--end", default="2026-06-04 15:30:00")
    parser.add_argument("--cash", type=float, default=1_000_000)
    parser.add_argument("--commission", type=float, default=0.0003)
    parser.add_argument("--slippage", type=float, default=0.001)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.token:
        raise SystemExit("GM_TOKEN is not set. Set it in the environment or pass --token before running gm backtest.")
    if not Path(args.signal_file).exists():
        raise SystemExit(f"Signal file not found: {args.signal_file}")

    global SIGNAL_FILE
    SIGNAL_FILE = str(Path(args.signal_file).resolve())
    os.environ["GM_SIGNAL_FILE"] = SIGNAL_FILE
    run(  # noqa: F405
        strategy_id=args.strategy_id,
        filename=Path(__file__).name,
        mode=MODE_BACKTEST,  # noqa: F405
        token=args.token,
        backtest_start_time=args.start,
        backtest_end_time=args.end,
        backtest_adjust=ADJUST_PREV,  # noqa: F405
        backtest_initial_cash=args.cash,
        backtest_commission_ratio=args.commission,
        backtest_slippage_ratio=args.slippage,
    )


if __name__ == "__main__":
    main()
