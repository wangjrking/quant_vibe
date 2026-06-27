# coding=utf-8
from __future__ import absolute_import, print_function

import csv
import os
import sqlite3
from datetime import datetime
from pathlib import Path

from gm.api import *


def _optional_signed_float_env(name):
    value = str(os.environ.get(name, "") or "").strip()
    if not value or value.lower() in {"none", "null"}:
        return None
    return float(value)


STRATEGY_NAME = "open_daily_nomarket_top1_q95_hold5_score095_mh2"
STRATEGY_ID = "ce327750-634e-11f1-b8d7-10ffe0295517"
SIGNAL_FILE = os.environ.get(
    "GM_SIGNAL_FILE",
    r"D:\work\quant\quant_mcp\quant\data_file\gm_signals_prod_aligned_4y_liq_enh_top1_q95_amt8e5_turn2_mv1e6_clean_nost_delist.csv",
)
MAX_POSITIONS = int(os.environ.get("GM_MAX_POSITIONS", "1"))
HOLDING_DAYS = int(os.environ.get("GM_HOLDING_DAYS", "5"))
DEFAULT_CONCURRENT_POSITIONS = max(1, min(MAX_POSITIONS, HOLDING_DAYS))
TARGET_POSITION_PCT = float(os.environ.get("GM_TARGET_POSITION_PCT", str(0.98 / DEFAULT_CONCURRENT_POSITIONS)))
CASH_BUFFER = float(os.environ.get("GM_CASH_BUFFER", "0.98"))
SYNC_POSITIONS = str(os.environ.get("GM_SYNC_POSITIONS", "0") or "0").strip().lower() in {"1", "true", "yes"}
RESIZE_HELD_ON_SIGNAL = str(os.environ.get("GM_RESIZE_HELD_ON_SIGNAL", "0") or "0").strip().lower() in {
    "1",
    "true",
    "yes",
}
RESIZE_HELD_TARGET_MULT = float(os.environ.get("GM_RESIZE_HELD_TARGET_MULT", "1.0"))
RESIZE_HELD_MIN_DELTA_PCT = float(os.environ.get("GM_RESIZE_HELD_MIN_DELTA_PCT", "0.015"))
BACKTEST_START = os.environ.get("GM_BACKTEST_START", "2022-06-06 09:00:00")
BACKTEST_END = os.environ.get("GM_BACKTEST_END", "2026-06-13 15:30:00")
BACKTEST_ADJUST_MODE = str(os.environ.get("GM_BACKTEST_ADJUST", "none") or "none").strip().lower()
BACKTEST_INITIAL_CASH = float(os.environ.get("GM_BACKTEST_INITIAL_CASH", "600000"))
BACKTEST_SLIPPAGE_RATIO = float(os.environ.get("GM_BACKTEST_SLIPPAGE_RATIO", "0.0015"))
OPEN_PRICE_DAILY_MODE = str(os.environ.get("GM_OPEN_PRICE_DAILY_MODE", "1") or "1").strip().lower() not in {
    "0",
    "false",
    "no",
}
OPEN_DAILY_SCORE_EXIT = str(os.environ.get("GM_OPEN_DAILY_SCORE_EXIT", "1") or "1").strip().lower() in {
    "1",
    "true",
    "yes",
}
DEFER_EXITS_WITHOUT_BUY_SIGNAL = str(os.environ.get("GM_DEFER_EXITS_WITHOUT_BUY_SIGNAL", "0") or "0").strip().lower() in {
    "1",
    "true",
    "yes",
}
NO_SIGNAL_MAX_HOLDING_DAYS = int(os.environ.get("GM_NO_SIGNAL_MAX_HOLDING_DAYS", "0"))
DEFER_EXIT_MIN_POSITION_RETURN = _optional_signed_float_env("GM_DEFER_EXIT_MIN_POSITION_RETURN")
DEFER_EXIT_MAX_POSITION_RETURN = _optional_signed_float_env("GM_DEFER_EXIT_MAX_POSITION_RETURN")
SCHEDULE_TIME = os.environ.get("GM_SCHEDULE_TIME", "09:30:00" if OPEN_PRICE_DAILY_MODE else "09:35:00")
SELL_SCHEDULE_TIME = os.environ.get("GM_SELL_SCHEDULE_TIME", SCHEDULE_TIME)
BUY_SCHEDULE_TIME = os.environ.get("GM_BUY_SCHEDULE_TIME", "09:31:00" if OPEN_PRICE_DAILY_MODE else SCHEDULE_TIME)
INTRADAY_RISK_MODE = str(os.environ.get("GM_INTRADAY_RISK_MODE", "0") or "0").strip().lower() in {
    "1",
    "true",
    "yes",
}
INTRADAY_REPLACE_BUY = str(os.environ.get("GM_INTRADAY_REPLACE_BUY", "0") or "0").strip().lower() in {
    "1",
    "true",
    "yes",
}
INTRADAY_REPLACE_MAX_BUYS = int(os.environ.get("GM_INTRADAY_REPLACE_MAX_BUYS", "1"))
FORCE_MARKET_ORDER = str(os.environ.get("GM_FORCE_MARKET_ORDER", "0") or "0").strip().lower() in {
    "1",
    "true",
    "yes",
}
FORCE_BUY_MARKET_ORDER = str(
    os.environ.get("GM_FORCE_BUY_MARKET_ORDER", "1" if FORCE_MARKET_ORDER else "0") or "0"
).strip().lower() in {
    "1",
    "true",
    "yes",
}
FORCE_SELL_MARKET_ORDER = str(
    os.environ.get("GM_FORCE_SELL_MARKET_ORDER", "1" if FORCE_MARKET_ORDER else "0") or "0"
).strip().lower() in {
    "1",
    "true",
    "yes",
}
SKIP_OPEN_LIMIT_UP_BUY = str(os.environ.get("GM_SKIP_OPEN_LIMIT_UP_BUY", "1") or "1").strip().lower() in {
    "1",
    "true",
    "yes",
}
VERBOSE_TRADES = str(os.environ.get("GM_VERBOSE_TRADES", "0") or "0").strip().lower() in {
    "1",
    "true",
    "yes",
}
LOG_EXPOSURE = str(os.environ.get("GM_LOG_EXPOSURE", "0") or "0").strip().lower() in {
    "1",
    "true",
    "yes",
}
EQUITY_DD_RISK_MODE = str(os.environ.get("GM_EQUITY_DD_RISK_MODE", "0") or "0").strip().lower() in {
    "1",
    "true",
    "yes",
}
EQUITY_DD_SOFT_TRIGGER = float(os.environ.get("GM_EQUITY_DD_SOFT_TRIGGER", "0.08"))
EQUITY_DD_HARD_TRIGGER = float(os.environ.get("GM_EQUITY_DD_HARD_TRIGGER", "0.14"))
EQUITY_DD_RECOVER_TRIGGER = float(os.environ.get("GM_EQUITY_DD_RECOVER_TRIGGER", "0.04"))
EQUITY_DD_SOFT_SCALE = float(os.environ.get("GM_EQUITY_DD_SOFT_SCALE", "0.85"))
EQUITY_DD_HARD_SCALE = float(os.environ.get("GM_EQUITY_DD_HARD_SCALE", "0.65"))
EQUITY_DD_RESIZE_EXISTING = str(os.environ.get("GM_EQUITY_DD_RESIZE_EXISTING", "0") or "0").strip().lower() in {
    "1",
    "true",
    "yes",
}
DB_IMMUTABLE_READ = str(os.environ.get("GM_DB_IMMUTABLE_READ", "1") or "1").strip().lower() in {
    "1",
    "true",
    "yes",
}
INTRADAY_RISK_TIMES = [
    item.strip()
    for item in os.environ.get("GM_INTRADAY_RISK_TIMES", "10:00:00,11:00:00,14:30:00").split(",")
    if item.strip()
]
INTRADAY_RISK_SELL_FRACTION = max(
    0.0,
    min(float(os.environ.get("GM_INTRADAY_RISK_SELL_FRACTION", "1.0")), 1.0),
)
INDEX_RISK_EXIT_MODE = str(os.environ.get("GM_INDEX_RISK_EXIT_MODE", "0") or "0").strip().lower() in {
    "1",
    "true",
    "yes",
}
INDEX_RISK_CC_THRESHOLD = float(os.environ.get("GM_INDEX_RISK_CC_THRESHOLD", "-0.04"))
INDEX_RISK_INTRADAY_THRESHOLD = float(os.environ.get("GM_INDEX_RISK_INTRADAY_THRESHOLD", "-0.04"))
INDEX_RISK_BUY_SCALE = max(0.0, min(float(os.environ.get("GM_INDEX_RISK_BUY_SCALE", "1.0")), 1.0))
BREADTH_RISK_EXIT_MODE = str(os.environ.get("GM_BREADTH_RISK_EXIT_MODE", "0") or "0").strip().lower() in {
    "1",
    "true",
    "yes",
}
BREADTH_RISK_UP_RATIO_THRESHOLD = float(os.environ.get("GM_BREADTH_RISK_UP_RATIO_THRESHOLD", "0.08"))
BREADTH_RISK_AVG_PCT_THRESHOLD = float(os.environ.get("GM_BREADTH_RISK_AVG_PCT_THRESHOLD", "-3.0"))
BREADTH_RISK_BUY_SCALE = max(0.0, min(float(os.environ.get("GM_BREADTH_RISK_BUY_SCALE", "1.0")), 1.0))


def _optional_float_env(name):
    value = str(os.environ.get(name, "") or "").strip()
    if not value or value.lower() in {"none", "null"}:
        return None
    return abs(float(value))


STOP_LOSS_PCT = _optional_float_env("GM_STOP_LOSS_PCT") if "GM_STOP_LOSS_PCT" in os.environ else 0.08
TAKE_PROFIT_PCT = _optional_float_env("GM_TAKE_PROFIT_PCT")
SCORE_STOP_LOSS_PRED = _optional_float_env("GM_SCORE_STOP_LOSS_PRED")
SCORE_TAKE_PROFIT_PRED = _optional_float_env("GM_SCORE_TAKE_PROFIT_PRED")
SCORE_STOP_LOSS_RATIO = _optional_float_env("GM_SCORE_STOP_LOSS_RATIO")
SCORE_TAKE_PROFIT_RATIO = _optional_float_env("GM_SCORE_TAKE_PROFIT_RATIO")
SCORE_STOP_LOSS_RANK = _optional_float_env("GM_SCORE_STOP_LOSS_RANK")
SCORE_TAKE_PROFIT_RANK = _optional_float_env("GM_SCORE_TAKE_PROFIT_RANK")
SCORE_STOP_LOSS_DAY_DROP_RATIO = _optional_float_env("GM_SCORE_STOP_LOSS_DAY_DROP_RATIO")
SCORE_TAKE_PROFIT_DAY_DROP_RATIO = _optional_float_env("GM_SCORE_TAKE_PROFIT_DAY_DROP_RATIO")
SCORE_EXIT_ENTRY_RATIO = _optional_float_env("GM_SCORE_EXIT_ENTRY_RATIO") if "GM_SCORE_EXIT_ENTRY_RATIO" in os.environ else 0.95
SCORE_EXIT_RANK = _optional_float_env("GM_SCORE_EXIT_RANK")
MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT = int(os.environ.get("GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT", "2"))
SCORE_CONTINUE_ENTRY_RATIO = (
    _optional_float_env("GM_SCORE_CONTINUE_ENTRY_RATIO") if "GM_SCORE_CONTINUE_ENTRY_RATIO" in os.environ else 1.0
)
MAX_HOLDING_DAYS = int(os.environ.get("GM_MAX_HOLDING_DAYS", str(HOLDING_DAYS)))
MAX_DAILY_SELLS = int(os.environ.get("GM_MAX_DAILY_SELLS", "0"))
LIGHT_STOP_LOSS_PCT = _optional_float_env("GM_LIGHT_STOP_LOSS_PCT")
MIN_HOLDING_DAYS_BEFORE_LIGHT_STOP = int(os.environ.get("GM_MIN_HOLDING_DAYS_BEFORE_LIGHT_STOP", "0"))
SCORE_DB = os.environ.get("GM_SCORE_DB", r"D:\work\quant\quant_mcp\quant\data_file\odb.db")
SCORE_TABLE = os.environ.get(
    "GM_SCORE_TABLE",
    "stock_predict_data_executable_5d_open_return_prod_aligned_202206_202606",
)
MARKET_DB = os.environ.get(
    "GM_MARKET_DB",
    SCORE_DB or r"D:\work\quant\quant_mcp\quant\data_file\odb.db",
)


def _resolve_backtest_adjust():
    adjust_none = globals().get("ADJUST_NONE", globals().get("ADJUST_PREV"))
    adjust_prev = globals().get("ADJUST_PREV", adjust_none)
    adjust_post = globals().get("ADJUST_POST", adjust_none)
    mapping = {
        "none": adjust_none,
        "raw": adjust_none,
        "prev": adjust_prev,
        "pre": adjust_prev,
        "qfq": adjust_prev,
        "post": adjust_post,
        "hfq": adjust_post,
    }
    return mapping.get(BACKTEST_ADJUST_MODE, adjust_none)


def _date_key(value):
    return value.strftime("%Y%m%d")


def _restore_stock_code(symbol):
    exchange, code = str(symbol).split(".", 1)
    suffix = {"SHSE": "SH", "SZSE": "SZ"}.get(exchange, exchange)
    return "{}.{}".format(code, suffix)


def _load_signals(path):
    signals_by_buy_date = {}
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


def _build_decision_date_by_buy_date(signals_by_buy_date):
    decision_dates = {}
    for buy_date, rows in (signals_by_buy_date or {}).items():
        signal_dates = sorted({str(row.get("signal_date") or "") for row in rows if row.get("signal_date")})
        if signal_dates:
            decision_dates[buy_date] = signal_dates[-1]
    return decision_dates


def _signal_optional_float(row, *keys):
    for key in keys:
        value = str((row or {}).get(key, "") or "").strip()
        if not value or value.lower() in {"none", "null"}:
            continue
        try:
            return abs(float(value))
        except Exception:
            continue
    return None


def _signal_optional_int(row, *keys):
    for key in keys:
        value = str((row or {}).get(key, "") or "").strip()
        if not value or value.lower() in {"none", "null"}:
            continue
        try:
            return int(float(value))
        except Exception:
            continue
    return None


def _iter_signal_stock_codes(signals_by_buy_date):
    seen = set()
    for rows in signals_by_buy_date.values():
        for row in rows:
            stock_code = str(row.get("stock_code") or "")
            if stock_code and stock_code not in seen:
                seen.add(stock_code)
                yield stock_code


def _normalize_trade_date_for_db(value):
    text = str(value or "").strip()
    if not text:
        return ""
    if len(text) == 8 and text.isdigit():
        return text
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").strftime("%Y%m%d")
    except ValueError:
        return text.replace("-", "")[:8]


def _connect_readonly_db(db_path):
    path = Path(db_path)
    if not DB_IMMUTABLE_READ:
        return sqlite3.connect(str(db_path), timeout=60)
    try:
        uri = path.resolve().as_uri() + "?mode=ro&immutable=1"
        return sqlite3.connect(uri, uri=True, timeout=30)
    except Exception:
        return sqlite3.connect(str(db_path), timeout=30)


def _load_daily_scores(db_path, table):
    scores_by_date = {}
    ranks_by_date = {}
    conn = _connect_readonly_db(db_path)
    try:
        sql = 'SELECT trade_date, stock_code, pred_prob FROM "{}" ORDER BY trade_date'.format(table.replace('"', '""'))
        for trade_date, stock_code, pred_prob in conn.execute(sql):
            symbol = None
            if stock_code:
                code, suffix = str(stock_code).split(".", 1)
                symbol = "{}.{}".format({"SH": "SHSE", "SZ": "SZSE"}.get(suffix, suffix), code)
            if trade_date and symbol and pred_prob is not None:
                scores_by_date.setdefault(str(trade_date), {})[symbol] = float(pred_prob)
    finally:
        conn.close()
    for trade_date, daily_scores in scores_by_date.items():
        ranked = sorted(daily_scores.items(), key=lambda item: item[1], reverse=True)
        size = len(ranked)
        if size == 1:
            ranks_by_date[trade_date] = {ranked[0][0]: 1.0}
            continue
        ranks_by_date[trade_date] = {
            symbol: 1.0 - (idx / float(size - 1))
            for idx, (symbol, _score) in enumerate(ranked)
        }
    return scores_by_date, ranks_by_date


def _extract_position_return(position):
    for key in ("fpnl_ratio", "pnl_ratio", "position_pnl_ratio", "floating_pnl_ratio"):
        value = position.get(key)
        if value not in (None, ""):
            return float(value)

    price = position.get("price")
    vwap = position.get("vwap")
    if price not in (None, "") and vwap not in (None, "", 0):
        return float(price) / float(vwap) - 1.0

    market_value = position.get("market_value")
    volume = position.get("volume")
    if market_value not in (None, "") and volume not in (None, "", 0) and vwap not in (None, "", 0):
        current_price = float(market_value) / float(volume)
        return current_price / float(vwap) - 1.0

    return None


def _get_position_return(context, symbol):
    account = getattr(context, "account", None)
    if account is None:
        return None
    positions = account().positions()
    for position in positions:
        if position.get("symbol") == symbol:
            return _extract_position_return(position)
    return None


def _get_pred_score(context, trade_date, symbol):
    scores_by_date = getattr(context, "pred_scores_by_date", None) or {}
    return scores_by_date.get(trade_date, {}).get(symbol)


def _get_pred_rank(context, trade_date, symbol):
    ranks_by_date = getattr(context, "pred_ranks_by_date", None) or {}
    return ranks_by_date.get(trade_date, {}).get(symbol)


def _as_float(value):
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _get_cash_snapshot(context):
    account_getter = getattr(context, "account", None)
    if account_getter is None:
        return None, None
    account_obj = account_getter()
    cash_obj = getattr(account_obj, "cash", None)
    if callable(cash_obj):
        cash_obj = cash_obj()

    available_cash = None
    nav = None
    if isinstance(cash_obj, dict):
        for key in ("available", "available_cash", "cash", "nav", "asset", "total_asset"):
            if key == "available" and available_cash is None:
                available_cash = _as_float(cash_obj.get(key))
            elif key == "available_cash" and available_cash is None:
                available_cash = _as_float(cash_obj.get(key))
            elif key == "cash" and available_cash is None:
                available_cash = _as_float(cash_obj.get(key))
            elif key in {"nav", "asset", "total_asset"} and nav is None:
                nav = _as_float(cash_obj.get(key))

    positions = account_obj.positions()
    if nav is None:
        market_value = sum(_as_float(position.get("market_value")) or 0.0 for position in positions)
        if available_cash is not None:
            nav = available_cash + market_value
    return available_cash, nav


def _log_exposure(context, trade_date, phase):
    if not LOG_EXPOSURE:
        return
    account_obj = getattr(context, "account", None)
    if account_obj is None:
        return
    account_obj = account_obj()
    positions = account_obj.positions()
    market_value = sum(_as_float(position.get("market_value")) or 0.0 for position in positions)
    available_cash, nav = _get_cash_snapshot(context)
    if nav in (None, 0):
        nav = market_value + (available_cash or 0.0)
    invested_pct = 0.0 if nav in (None, 0) else market_value / nav
    active_positions = sum(1 for position in positions if (_as_float(position.get("volume")) or 0.0) > 0)
    print(
        "EXPOSURE {} {} invested_pct={:.6f} active_positions={} market_value={:.2f} nav={:.2f} cash={}".format(
            trade_date,
            phase,
            float(invested_pct),
            active_positions,
            float(market_value or 0.0),
            float(nav or 0.0),
            "" if available_cash is None else "{:.2f}".format(float(available_cash)),
        )
    )


def _update_equity_drawdown_risk(context):
    if not EQUITY_DD_RISK_MODE:
        return 1.0
    _available_cash, nav = _get_cash_snapshot(context)
    if nav in (None, 0):
        return float(getattr(context, "equity_dd_risk_scale", 1.0) or 1.0)
    nav = float(nav)
    peak = max(float(getattr(context, "equity_peak_nav", BACKTEST_INITIAL_CASH) or BACKTEST_INITIAL_CASH), nav)
    context.equity_peak_nav = peak
    drawdown = max(0.0, 1.0 - nav / peak) if peak > 0 else 0.0
    scale = float(getattr(context, "equity_dd_risk_scale", 1.0) or 1.0)
    if drawdown >= EQUITY_DD_HARD_TRIGGER:
        scale = EQUITY_DD_HARD_SCALE
    elif drawdown >= EQUITY_DD_SOFT_TRIGGER:
        scale = EQUITY_DD_SOFT_SCALE
    elif drawdown <= EQUITY_DD_RECOVER_TRIGGER:
        scale = 1.0
    context.equity_dd_risk_scale = max(0.0, min(1.0, float(scale)))
    context.equity_dd_drawdown = drawdown
    return context.equity_dd_risk_scale


def _resize_existing_for_equity_risk(context):
    if not EQUITY_DD_RISK_MODE or not EQUITY_DD_RESIZE_EXISTING:
        return
    scale = _update_equity_drawdown_risk(context)
    if scale >= 0.999:
        return
    for symbol, state in list(context.held_symbols.items()):
        base_target = _as_float(state.get("entry_target_pct"))
        if base_target is None:
            base_target = TARGET_POSITION_PCT
        target_pct = max(0.0, min(float(base_target), TARGET_POSITION_PCT) * scale)
        order_target_percent(symbol=symbol, percent=target_pct, order_type=OrderType_Market, position_side=PositionSide_Long)


def _symbol_to_stock_code(symbol):
    exchange, code = str(symbol).split(".", 1)
    suffix = {"SHSE": "SH", "SZSE": "SZ", "BJSE": "BJ"}.get(exchange, exchange)
    return "{}.{}".format(code, suffix)


def _limit_up_pct(stock_code):
    code = str(stock_code or "")
    if code.startswith(("300", "301", "688")):
        return 0.20
    return 0.10


def _get_market_rows_for_date(context, trade_date):
    cache = getattr(context, "market_rows_by_date", None)
    if cache is None:
        cache = {}
        context.market_rows_by_date = cache
    if trade_date in cache:
        return cache[trade_date]
    if not MARKET_DB or not Path(MARKET_DB).exists():
        cache[trade_date] = {}
        return cache[trade_date]
    preloaded = getattr(context, "market_rows_preloaded", False)
    if preloaded:
        cache[trade_date] = {}
        return cache[trade_date]
    conn = getattr(context, "market_db_conn", None)
    if conn is None:
        conn = sqlite3.connect(MARKET_DB)
        context.market_db_conn = conn
    cursor = conn.execute(
        """
        SELECT stock_code, open, high, low, close, pre_close
        FROM STOCK_DAILY_DATA
        WHERE trade_date = ?
        """,
        (trade_date,),
    )
    rows = {}
    for stock_code, open_price, high_price, low_price, close_price, pre_close in cursor:
        rows[str(stock_code)] = {
            "stock_code": str(stock_code),
            "open": _as_float(open_price),
            "high": _as_float(high_price),
            "low": _as_float(low_price),
            "close": _as_float(close_price),
            "pre_close": _as_float(pre_close),
        }
    cache[trade_date] = rows
    return rows


def _preload_market_rows(context):
    if not MARKET_DB or not Path(MARKET_DB).exists():
        context.market_rows_by_date = {}
        context.market_rows_preloaded = True
        return
    stock_codes = list(_iter_signal_stock_codes(getattr(context, "signals_by_buy_date", {}) or {}))
    if not stock_codes:
        context.market_rows_by_date = {}
        context.market_rows_preloaded = True
        return
    buy_dates = sorted((getattr(context, "signals_by_buy_date", {}) or {}).keys())
    if not buy_dates:
        context.market_rows_by_date = {}
        context.market_rows_preloaded = True
        return
    start_date = buy_dates[0]
    end_date = _normalize_trade_date_for_db(BACKTEST_END)
    conn = _connect_readonly_db(MARKET_DB)
    try:
        rows_by_date = {}
        chunk_size = 400
        for offset in range(0, len(stock_codes), chunk_size):
            chunk = stock_codes[offset : offset + chunk_size]
            placeholders = ",".join("?" for _ in chunk)
            sql = f"""
                SELECT trade_date, stock_code, open, high, low, close, pre_close
                FROM STOCK_DAILY_DATA
                WHERE trade_date >= ? AND trade_date <= ?
                  AND stock_code IN ({placeholders})
            """
            params = [start_date, end_date] + chunk
            cursor = conn.execute(sql, params)
            for trade_date, stock_code, open_price, high_price, low_price, close_price, pre_close in cursor:
                rows_by_date.setdefault(str(trade_date), {})[str(stock_code)] = {
                    "stock_code": str(stock_code),
                    "open": _as_float(open_price),
                    "high": _as_float(high_price),
                    "low": _as_float(low_price),
                    "close": _as_float(close_price),
                    "pre_close": _as_float(pre_close),
                }
        context.market_rows_by_date = rows_by_date
        context.market_rows_preloaded = True
    finally:
        conn.close()


def _get_market_row(context, trade_date, symbol):
    stock_code = _symbol_to_stock_code(symbol)
    return _get_market_rows_for_date(context, trade_date).get(stock_code)


def _get_index_metrics(context, signal_date):
    if not signal_date or not MARKET_DB or not Path(MARKET_DB).exists():
        return None
    cache = getattr(context, "index_metrics_by_date", None)
    if cache is None:
        cache = {}
        context.index_metrics_by_date = cache
    if signal_date in cache:
        return cache[signal_date]
    conn = getattr(context, "index_market_db_conn", None)
    if conn is None:
        conn = sqlite3.connect(MARKET_DB)
        context.index_market_db_conn = conn
    row = conn.execute(
        """
        SELECT index_2000_open, index_2000_close, index_2000_high, index_2000_low
        FROM STOCK_DAILY_DATA
        WHERE trade_date = ? AND index_2000_close IS NOT NULL
        LIMIT 1
        """,
        (signal_date,),
    ).fetchone()
    prev_date_row = conn.execute(
        """
        SELECT MAX(trade_date)
        FROM STOCK_DAILY_DATA
        WHERE trade_date < ? AND index_2000_close IS NOT NULL
        """,
        (signal_date,),
    ).fetchone()
    prev_close = None
    if prev_date_row and prev_date_row[0]:
        prev_row = conn.execute(
            """
            SELECT index_2000_close
            FROM STOCK_DAILY_DATA
            WHERE trade_date = ? AND index_2000_close IS NOT NULL
            LIMIT 1
            """,
            (prev_date_row[0],),
        ).fetchone()
        prev_close = _as_float(prev_row[0]) if prev_row else None
    metrics = {}
    if row:
        open_price = _as_float(row[0])
        close_price = _as_float(row[1])
        metrics["intraday_return"] = (
            close_price / open_price - 1.0 if open_price not in (None, 0) and close_price is not None else None
        )
        metrics["close_to_close_return"] = (
            close_price / prev_close - 1.0 if prev_close not in (None, 0) and close_price is not None else None
        )
    cache[signal_date] = metrics
    return metrics


def _index_risk_active(context, buy_date):
    if not INDEX_RISK_EXIT_MODE:
        return False
    decision_date = (getattr(context, "decision_date_by_buy_date", None) or {}).get(buy_date)
    metrics = _get_index_metrics(context, decision_date)
    if not metrics:
        return False
    cc_ret = metrics.get("close_to_close_return")
    intraday_ret = metrics.get("intraday_return")
    return (
        (cc_ret is not None and cc_ret <= INDEX_RISK_CC_THRESHOLD)
        or (intraday_ret is not None and intraday_ret <= INDEX_RISK_INTRADAY_THRESHOLD)
    )


def _get_breadth_metrics(context, signal_date):
    if not signal_date or not MARKET_DB or not Path(MARKET_DB).exists():
        return None
    cache = getattr(context, "breadth_metrics_by_date", None)
    if cache is None:
        cache = {}
        context.breadth_metrics_by_date = cache
    if signal_date in cache:
        return cache[signal_date]
    conn = getattr(context, "breadth_market_db_conn", None)
    if conn is None:
        conn = sqlite3.connect(MARKET_DB)
        context.breadth_market_db_conn = conn
    rows = conn.execute(
        """
        SELECT pct_chg
        FROM STOCK_DAILY_DATA
        WHERE trade_date = ? AND pct_chg IS NOT NULL
        """,
        (signal_date,),
    ).fetchall()
    values = [_as_float(row[0]) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        cache[signal_date] = None
        return None
    metrics = {
        "up_ratio": sum(1 for value in values if value > 0) / float(len(values)),
        "avg_pct_chg": sum(values) / float(len(values)),
    }
    cache[signal_date] = metrics
    return metrics


def _breadth_risk_active(context, buy_date):
    if not BREADTH_RISK_EXIT_MODE:
        return False
    decision_date = (getattr(context, "decision_date_by_buy_date", None) or {}).get(buy_date)
    metrics = _get_breadth_metrics(context, decision_date)
    if not metrics:
        return False
    up_ratio = metrics.get("up_ratio")
    avg_pct_chg = metrics.get("avg_pct_chg")
    return (
        (up_ratio is not None and up_ratio <= BREADTH_RISK_UP_RATIO_THRESHOLD)
        or (avg_pct_chg is not None and avg_pct_chg <= BREADTH_RISK_AVG_PCT_THRESHOLD)
    )


def _market_risk_active(context, buy_date):
    return _index_risk_active(context, buy_date) or _breadth_risk_active(context, buy_date)


def _limit_bounds(row):
    pre_close = _as_float((row or {}).get("pre_close"))
    if pre_close in (None, 0):
        return None, None
    pct = _limit_up_pct(row.get("stock_code"))
    return pre_close * (1.0 + pct), pre_close * (1.0 - pct)


def _is_open_limit_up(row):
    open_price = _as_float((row or {}).get("open"))
    upper, _lower = _limit_bounds(row)
    if open_price is None or upper is None:
        return False
    return open_price >= upper * 0.995


def _is_open_limit_down(row):
    open_price = _as_float((row or {}).get("open"))
    _upper, lower = _limit_bounds(row)
    if open_price is None or lower is None:
        return False
    return open_price <= lower * 1.005


def _round_lot_volume(target_value, price):
    if price in (None, 0):
        return 0
    raw = int(float(target_value) / float(price))
    return max((raw // 100) * 100, 0)


def _submit_buy_order(context, symbol, target_pct, trade_date):
    market_row = _get_market_row(context, trade_date, symbol)
    if not market_row:
        if VERBOSE_TRADES:
            print("BUY_ATTEMPT {} {} target={:.6f} mode=market reason=no_market_row".format(trade_date, symbol, float(target_pct)))
        order_target_percent(symbol=symbol, percent=target_pct, order_type=OrderType_Market, position_side=PositionSide_Long)
        return True
    if SKIP_OPEN_LIMIT_UP_BUY and _is_open_limit_up(market_row):
        if VERBOSE_TRADES:
            print("BUY_SKIP {} {} reason=open_limit_up".format(trade_date, symbol))
        return False
    if FORCE_BUY_MARKET_ORDER:
        if VERBOSE_TRADES:
            print("BUY_ATTEMPT {} {} target={:.6f} mode=market open={}".format(trade_date, symbol, float(target_pct), market_row.get("open")))
        order_target_percent(symbol=symbol, percent=target_pct, order_type=OrderType_Market, position_side=PositionSide_Long)
        return True
    open_price = _as_float(market_row.get("open"))
    if open_price in (None, 0):
        if VERBOSE_TRADES:
            print("BUY_SKIP {} {} reason=no_open".format(trade_date, symbol))
        return False
    available_cash, nav = _get_cash_snapshot(context)
    if nav in (None, 0):
        if VERBOSE_TRADES:
            print("BUY_SKIP {} {} reason=no_nav".format(trade_date, symbol))
        return False
    target_value = float(target_pct) * float(nav)
    if available_cash is not None:
        target_value = min(target_value, float(available_cash) * CASH_BUFFER)
    volume = _round_lot_volume(target_value, open_price)
    if volume < 100:
        if VERBOSE_TRADES:
            print("BUY_SKIP {} {} reason=volume_lt_100 target_value={} open={}".format(trade_date, symbol, target_value, open_price))
        return False
    if VERBOSE_TRADES:
        print("BUY_ATTEMPT {} {} volume={} price={} target={:.6f}".format(trade_date, symbol, volume, open_price, float(target_pct)))
    order_volume(
        symbol=symbol,
        volume=volume,
        side=OrderSide_Buy,
        order_type=OrderType_Limit,
        position_effect=PositionEffect_Open,
        price=open_price,
    )
    return True


def _position_market_value(context, symbol, market_row=None):
    account_getter = getattr(context, "account", None)
    if account_getter is None:
        return None
    for position in account_getter().positions():
        if position.get("symbol") != symbol:
            continue
        market_value = _as_float(position.get("market_value"))
        if market_value is not None:
            return market_value
        volume = _as_float(position.get("volume"))
        open_price = _as_float((market_row or {}).get("open"))
        if volume is not None and open_price not in (None, 0):
            return float(volume) * float(open_price)
    return None


def _submit_topup_order(context, symbol, target_total_pct, trade_date):
    market_row = _get_market_row(context, trade_date, symbol)
    if not market_row:
        return False
    if SKIP_OPEN_LIMIT_UP_BUY and _is_open_limit_up(market_row):
        if VERBOSE_TRADES:
            print("BUY_SKIP {} {} reason=open_limit_up_topup".format(trade_date, symbol))
        return False
    open_price = _as_float(market_row.get("open"))
    if open_price in (None, 0):
        if VERBOSE_TRADES:
            print("BUY_SKIP {} {} reason=no_open_topup".format(trade_date, symbol))
        return False
    available_cash, nav = _get_cash_snapshot(context)
    if nav in (None, 0):
        return False
    current_value = _position_market_value(context, symbol, market_row) or 0.0
    target_value = float(target_total_pct) * float(nav)
    delta_value = target_value - float(current_value)
    if delta_value / float(nav) < RESIZE_HELD_MIN_DELTA_PCT:
        return False
    if available_cash is not None:
        delta_value = min(delta_value, float(available_cash) * CASH_BUFFER)
    volume = _round_lot_volume(delta_value, open_price)
    if volume < 100:
        return False
    if VERBOSE_TRADES:
        print(
            "BUY_TOPUP {} {} volume={} price={} target_total={:.6f}".format(
                trade_date, symbol, volume, open_price, float(target_total_pct)
            )
        )
    order_volume(
        symbol=symbol,
        volume=volume,
        side=OrderSide_Buy,
        order_type=OrderType_Limit,
        position_effect=PositionEffect_Open,
        price=open_price,
    )
    return True


def _submit_sell_order(context, symbol, trade_date):
    market_row = _get_market_row(context, trade_date, symbol)
    if not market_row:
        if VERBOSE_TRADES:
            print("SELL_ATTEMPT {} {} target=0 mode=market reason=no_market_row".format(trade_date, symbol))
        order_target_percent(symbol=symbol, percent=0, order_type=OrderType_Market, position_side=PositionSide_Long)
        return True
    if _is_open_limit_down(market_row):
        if VERBOSE_TRADES:
            print("SELL_SKIP {} {} reason=open_limit_down".format(trade_date, symbol))
        return False
    if FORCE_SELL_MARKET_ORDER:
        if VERBOSE_TRADES:
            print("SELL_ATTEMPT {} {} target=0 mode=market open={}".format(trade_date, symbol, market_row.get("open")))
        order_target_percent(symbol=symbol, percent=0, order_type=OrderType_Market, position_side=PositionSide_Long)
        return True
    open_price = _as_float(market_row.get("open"))
    if open_price in (None, 0):
        return False
    account_getter = getattr(context, "account", None)
    if account_getter is None:
        return False
    positions = account_getter().positions()
    for position in positions:
        if position.get("symbol") != symbol:
            continue
        volume = int(float(position.get("volume") or 0))
        if volume <= 0:
            if VERBOSE_TRADES:
                print("SELL_ATTEMPT {} {} target=0 mode=market reason=volume_le_0".format(trade_date, symbol))
            order_target_percent(symbol=symbol, percent=0, order_type=OrderType_Market, position_side=PositionSide_Long)
            return True
        if VERBOSE_TRADES:
            print("SELL_ATTEMPT {} {} volume={} price={}".format(trade_date, symbol, volume, open_price))
        order_volume(
            symbol=symbol,
            volume=volume,
            side=OrderSide_Sell,
            order_type=OrderType_Limit,
            position_effect=PositionEffect_Close,
            price=open_price,
        )
        return True
    return False


def _submit_sell_market_order(context, symbol):
    order_target_percent(symbol=symbol, percent=0, order_type=OrderType_Market, position_side=PositionSide_Long)
    return True


def _submit_intraday_risk_sell_order(context, symbol):
    if INTRADAY_RISK_SELL_FRACTION >= 0.999:
        return _submit_sell_market_order(context, symbol), True
    if INTRADAY_RISK_SELL_FRACTION <= 0:
        return False, False

    account_getter = getattr(context, "account", None)
    if account_getter is None:
        return False, False
    for position in account_getter().positions():
        if position.get("symbol") != symbol:
            continue
        volume = int(float(position.get("volume") or 0))
        if volume <= 0:
            return _submit_sell_market_order(context, symbol), True
        sell_volume = int(volume * INTRADAY_RISK_SELL_FRACTION / 100) * 100
        if sell_volume < 100:
            return False, False
        if volume - sell_volume < 100:
            return _submit_sell_market_order(context, symbol), True
        order_volume(
            symbol=symbol,
            volume=sell_volume,
            side=OrderSide_Sell,
            order_type=OrderType_Market,
            position_effect=PositionEffect_Close,
        )
        return True, False
    return _submit_sell_market_order(context, symbol), True


def _should_score_exit(context, state, decision_date, symbol, holding_days_elapsed):
    pred_score = _get_pred_score(context, decision_date, symbol)
    pred_rank = _get_pred_rank(context, decision_date, symbol)
    entry_pred_prob = state.get("entry_pred_prob")
    score_exit_ratio = _as_float(state.get("signal_score_exit_entry_ratio"))
    if score_exit_ratio is None:
        score_exit_ratio = SCORE_EXIT_ENTRY_RATIO
    min_hold_before_score_exit = state.get("signal_min_holding_days_before_score_exit")
    if min_hold_before_score_exit in (None, "", "None"):
        min_hold_before_score_exit = MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT
    else:
        min_hold_before_score_exit = int(float(min_hold_before_score_exit))
    if (
        pred_score is not None
        and entry_pred_prob not in (None, "", 0)
        and score_exit_ratio is not None
        and holding_days_elapsed >= min_hold_before_score_exit
        and pred_score <= float(entry_pred_prob) * score_exit_ratio
    ):
        return True
    last_pred_prob = state.get("last_pred_prob")
    if (
        pred_score is not None
        and last_pred_prob not in (None, "", 0)
        and SCORE_STOP_LOSS_DAY_DROP_RATIO is not None
        and holding_days_elapsed >= MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT
        and pred_score <= float(last_pred_prob) * SCORE_STOP_LOSS_DAY_DROP_RATIO
    ):
        return True
    if (
        pred_rank is not None
        and SCORE_EXIT_RANK is not None
        and holding_days_elapsed >= MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT
        and pred_rank <= SCORE_EXIT_RANK
    ):
        return True
    if pred_score is not None:
        state["last_pred_prob"] = pred_score
    return False


def _sync_held_symbols_with_positions(context):
    account_getter = getattr(context, "account", None)
    if account_getter is None:
        return
    actual_symbols = {
        position.get("symbol")
        for position in account_getter().positions()
        if _as_float(position.get("volume")) not in (None, 0)
    }
    for symbol in list(context.held_symbols.keys()):
        if symbol not in actual_symbols:
            del context.held_symbols[symbol]
    for symbol in actual_symbols:
        if symbol in context.held_symbols:
            continue
        context.held_symbols[symbol] = {
            "entry_index": getattr(context, "trade_index", 0) - max(int(HOLDING_DAYS), int(MAX_HOLDING_DAYS or HOLDING_DAYS)),
            "signal_date": None,
            "buy_date": getattr(context, "last_trade_date", None),
            "holding_days": HOLDING_DAYS,
            "max_holding_days": MAX_HOLDING_DAYS,
            "entry_pred_prob": None,
            "last_pred_prob": None,
            "entry_target_pct": TARGET_POSITION_PCT,
            "signal_stop_loss_pct": STOP_LOSS_PCT,
            "signal_take_profit_pct": TAKE_PROFIT_PCT,
            "signal_score_exit_entry_ratio": SCORE_EXIT_ENTRY_RATIO,
            "signal_score_continue_entry_ratio": SCORE_CONTINUE_ENTRY_RATIO,
            "signal_min_holding_days_before_score_exit": MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT,
        }


def init(context):
    if not Path(SIGNAL_FILE).exists():
        raise RuntimeError("Signal file not found: {}".format(SIGNAL_FILE))
    context.signals_by_buy_date = _load_signals(SIGNAL_FILE)
    context.decision_date_by_buy_date = _build_decision_date_by_buy_date(context.signals_by_buy_date)
    if SCORE_DB and SCORE_TABLE:
        context.pred_scores_by_date, context.pred_ranks_by_date = _load_daily_scores(SCORE_DB, SCORE_TABLE)
    else:
        context.pred_scores_by_date = {}
        context.pred_ranks_by_date = {}
    context.held_symbols = {}
    context.trade_index = -1
    context.last_trade_date = None
    context.market_rows_by_date = {}
    context.market_rows_preloaded = False
    context.equity_peak_nav = BACKTEST_INITIAL_CASH
    context.equity_dd_risk_scale = 1.0
    context.equity_dd_drawdown = 0.0
    _preload_market_rows(context)
    print("{} loaded signals from {}".format(STRATEGY_NAME, SIGNAL_FILE))
    if context.pred_scores_by_date:
        print("{} loaded daily scores from {} / {}".format(STRATEGY_NAME, SCORE_DB, SCORE_TABLE))
    if OPEN_PRICE_DAILY_MODE:
        schedule(schedule_func=sell_daily, date_rule="1d", time_rule=SELL_SCHEDULE_TIME)
        schedule(schedule_func=buy_daily, date_rule="1d", time_rule=BUY_SCHEDULE_TIME)
    else:
        schedule(schedule_func=trade_daily, date_rule="1d", time_rule=SCHEDULE_TIME)
    if INTRADAY_RISK_MODE and (STOP_LOSS_PCT is not None or TAKE_PROFIT_PCT is not None):
        for risk_time in INTRADAY_RISK_TIMES:
            schedule(schedule_func=intraday_risk_check, date_rule="1d", time_rule=risk_time)


def intraday_risk_check(context):
    today = _start_trade_day(context)
    sold_count = 0
    for symbol in list(context.held_symbols.keys()):
        state = context.held_symbols.get(symbol) or {}
        position_return = _get_position_return(context, symbol)
        should_exit = False
        stop_loss_pct = _as_float(state.get("signal_stop_loss_pct"))
        if stop_loss_pct is None:
            stop_loss_pct = STOP_LOSS_PCT
        take_profit_pct = _as_float(state.get("signal_take_profit_pct"))
        if take_profit_pct is None:
            take_profit_pct = TAKE_PROFIT_PCT
        if position_return is not None and stop_loss_pct is not None and position_return <= -stop_loss_pct:
            should_exit = True
        if position_return is not None and take_profit_pct is not None and position_return >= take_profit_pct:
            should_exit = True
        if should_exit:
            submitted, fully_closed = _submit_intraday_risk_sell_order(context, symbol)
            if submitted:
                if fully_closed:
                    del context.held_symbols[symbol]
                    sold_count += 1
                else:
                    context.held_symbols[symbol]["intraday_risk_reduced_count"] = (
                        int(context.held_symbols[symbol].get("intraday_risk_reduced_count") or 0) + 1
                    )
    if INTRADAY_REPLACE_BUY and sold_count > 0:
        max_buys = min(sold_count, max(INTRADAY_REPLACE_MAX_BUYS, 0))
        if max_buys > 0:
            _buy_from_daily_signals(context, today, max_buys=max_buys)


def _start_trade_day(context):
    today = _date_key(context.now)
    if getattr(context, "last_trade_date", None) != today:
        context.trade_index += 1
        context.last_trade_date = today
    if SYNC_POSITIONS:
        _sync_held_symbols_with_positions(context)
    return today


def sell_daily(context):
    today = _start_trade_day(context)
    sold_count = 0
    if _market_risk_active(context, today):
        for symbol in list(context.held_symbols.keys()):
            if _submit_sell_order(context, symbol, today):
                del context.held_symbols[symbol]
                sold_count += 1
        return
    for symbol, state in list(context.held_symbols.items()):
        if MAX_DAILY_SELLS > 0 and sold_count >= MAX_DAILY_SELLS:
            break
        holding_days_elapsed = context.trade_index - state["entry_index"]
        should_exit = False
        effective_holding_days = int(state.get("holding_days") or HOLDING_DAYS)
        effective_max_holding_days = int(state.get("max_holding_days") or MAX_HOLDING_DAYS or effective_holding_days)
        has_buy_signal_today = bool((getattr(context, "signals_by_buy_date", None) or {}).get(today))
        position_return_for_defer = _get_position_return(context, symbol)
        defer_allowed_by_return = True
        if DEFER_EXIT_MIN_POSITION_RETURN is not None and (
            position_return_for_defer is None or position_return_for_defer < DEFER_EXIT_MIN_POSITION_RETURN
        ):
            defer_allowed_by_return = False
        if DEFER_EXIT_MAX_POSITION_RETURN is not None and (
            position_return_for_defer is None or position_return_for_defer > DEFER_EXIT_MAX_POSITION_RETURN
        ):
            defer_allowed_by_return = False
        if (
            OPEN_PRICE_DAILY_MODE
            and DEFER_EXITS_WITHOUT_BUY_SIGNAL
            and not has_buy_signal_today
            and (NO_SIGNAL_MAX_HOLDING_DAYS <= 0 or holding_days_elapsed < NO_SIGNAL_MAX_HOLDING_DAYS)
            and defer_allowed_by_return
        ):
            continue
        if OPEN_PRICE_DAILY_MODE:
            decision_date = (getattr(context, "decision_date_by_buy_date", None) or {}).get(today)
            pred_score = _get_pred_score(context, decision_date, symbol) if decision_date else None
            if holding_days_elapsed >= effective_holding_days:
                entry_pred_prob = state.get("entry_pred_prob")
                score_continue_entry_ratio = _as_float(state.get("signal_score_continue_entry_ratio"))
                if score_continue_entry_ratio is None:
                    score_continue_entry_ratio = SCORE_CONTINUE_ENTRY_RATIO
                allow_continue = (
                    score_continue_entry_ratio is not None
                    and holding_days_elapsed < effective_max_holding_days
                    and pred_score is not None
                    and entry_pred_prob not in (None, "", 0)
                    and pred_score >= float(entry_pred_prob) * score_continue_entry_ratio
                )
                if not allow_continue:
                    should_exit = True
            elif OPEN_DAILY_SCORE_EXIT:
                if decision_date and _should_score_exit(context, state, decision_date, symbol, holding_days_elapsed):
                    should_exit = True
            if should_exit and _submit_sell_order(context, symbol, today):
                del context.held_symbols[symbol]
                sold_count += 1
            elif pred_score is not None:
                state["last_pred_prob"] = pred_score
            continue

        position_return = _get_position_return(context, symbol)
        pred_score = _get_pred_score(context, today, symbol)
        pred_rank = _get_pred_rank(context, today, symbol)
        if position_return is not None and STOP_LOSS_PCT is not None and position_return <= -STOP_LOSS_PCT:
            should_exit = True
        if position_return is not None and TAKE_PROFIT_PCT is not None and position_return >= TAKE_PROFIT_PCT:
            should_exit = True
        if (
            position_return is not None
            and pred_score is not None
            and position_return < 0
            and SCORE_STOP_LOSS_PRED is not None
            and pred_score <= SCORE_STOP_LOSS_PRED
        ):
            should_exit = True
        entry_pred_prob = state.get("entry_pred_prob")
        if (
            position_return is not None
            and pred_score is not None
            and entry_pred_prob not in (None, "", 0)
            and position_return < 0
            and SCORE_STOP_LOSS_RATIO is not None
            and pred_score <= float(entry_pred_prob) * SCORE_STOP_LOSS_RATIO
        ):
            should_exit = True
        if (
            position_return is not None
            and pred_score is not None
            and position_return > 0
            and SCORE_TAKE_PROFIT_PRED is not None
            and pred_score <= SCORE_TAKE_PROFIT_PRED
        ):
            should_exit = True
        if (
            position_return is not None
            and pred_score is not None
            and entry_pred_prob not in (None, "", 0)
            and position_return > 0
            and SCORE_TAKE_PROFIT_RATIO is not None
            and pred_score <= float(entry_pred_prob) * SCORE_TAKE_PROFIT_RATIO
        ):
            should_exit = True
        if (
            position_return is not None
            and pred_rank is not None
            and position_return < 0
            and SCORE_STOP_LOSS_RANK is not None
            and pred_rank <= SCORE_STOP_LOSS_RANK
        ):
            should_exit = True
        if (
            position_return is not None
            and pred_rank is not None
            and position_return > 0
            and SCORE_TAKE_PROFIT_RANK is not None
            and pred_rank <= SCORE_TAKE_PROFIT_RANK
        ):
            should_exit = True
        last_pred_prob = state.get("last_pred_prob")
        if (
            position_return is not None
            and pred_score is not None
            and last_pred_prob not in (None, "", 0)
            and position_return < 0
            and SCORE_STOP_LOSS_DAY_DROP_RATIO is not None
            and pred_score <= float(last_pred_prob) * SCORE_STOP_LOSS_DAY_DROP_RATIO
        ):
            should_exit = True
        if (
            position_return is not None
            and pred_score is not None
            and last_pred_prob not in (None, "", 0)
            and position_return > 0
            and SCORE_TAKE_PROFIT_DAY_DROP_RATIO is not None
            and pred_score <= float(last_pred_prob) * SCORE_TAKE_PROFIT_DAY_DROP_RATIO
        ):
            should_exit = True
        if (
            pred_score is not None
            and entry_pred_prob not in (None, "", 0)
            and SCORE_EXIT_ENTRY_RATIO is not None
            and holding_days_elapsed >= MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT
            and pred_score <= float(entry_pred_prob) * SCORE_EXIT_ENTRY_RATIO
        ):
            should_exit = True
        if (
            pred_rank is not None
            and SCORE_EXIT_RANK is not None
            and holding_days_elapsed >= MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT
            and pred_rank <= SCORE_EXIT_RANK
        ):
            should_exit = True
        if (
            position_return is not None
            and LIGHT_STOP_LOSS_PCT is not None
            and holding_days_elapsed >= MIN_HOLDING_DAYS_BEFORE_LIGHT_STOP
            and position_return <= -LIGHT_STOP_LOSS_PCT
        ):
            should_exit = True
        if holding_days_elapsed >= effective_holding_days:
            allow_continue = (
                SCORE_CONTINUE_ENTRY_RATIO is not None
                and holding_days_elapsed < effective_max_holding_days
                and pred_score is not None
                and entry_pred_prob not in (None, "", 0)
                and pred_score >= float(entry_pred_prob) * SCORE_CONTINUE_ENTRY_RATIO
            )
            if not allow_continue:
                should_exit = True
        if should_exit:
            if _submit_sell_order(context, symbol, today):
                del context.held_symbols[symbol]
                sold_count += 1
        elif pred_score is not None:
            state["last_pred_prob"] = pred_score


def buy_daily(context):
    today = _start_trade_day(context)
    if SYNC_POSITIONS:
        _sync_held_symbols_with_positions(context)

    _resize_existing_for_equity_risk(context)
    _buy_from_daily_signals(context, today)
    _log_exposure(context, today, "post_buy")


def _buy_from_daily_signals(context, today, max_buys=None):
    open_slots = max(MAX_POSITIONS - len(context.held_symbols), 0)
    if max_buys is not None:
        open_slots = min(open_slots, max(int(max_buys), 0))
    if open_slots <= 0 and not RESIZE_HELD_ON_SIGNAL:
        return 0

    bought_count = 0
    for signal in context.signals_by_buy_date.get(today, []):
        symbol = signal["symbol"]
        signal_target_pct = signal.get("target_pct")
        target_pct = float(signal_target_pct) if signal_target_pct not in (None, "", "None") else TARGET_POSITION_PCT
        target_pct = min(target_pct, TARGET_POSITION_PCT)
        if _market_risk_active(context, today):
            target_pct *= INDEX_RISK_BUY_SCALE
            target_pct *= BREADTH_RISK_BUY_SCALE
        target_pct *= _update_equity_drawdown_risk(context)
        if symbol in context.held_symbols:
            if RESIZE_HELD_ON_SIGNAL:
                target_total_pct = min(target_pct * RESIZE_HELD_TARGET_MULT, TARGET_POSITION_PCT)
                if _submit_topup_order(context, symbol, target_total_pct, today):
                    state = context.held_symbols.get(symbol) or {}
                    state["entry_target_pct"] = max(_as_float(state.get("entry_target_pct")) or 0.0, target_total_pct)
                    state["last_pred_prob"] = signal.get("pred_prob")
                    context.held_symbols[symbol] = state
            continue
        if open_slots <= 0:
            continue
        available_cash, nav = _get_cash_snapshot(context)
        if available_cash is not None and nav not in (None, 0):
            cash_limited_target_pct = max(0.0, min(float(available_cash) / float(nav) * CASH_BUFFER, TARGET_POSITION_PCT))
            target_pct = min(target_pct, cash_limited_target_pct)
        if target_pct <= 0:
            continue
        if _submit_buy_order(context, symbol, target_pct, today):
            context.held_symbols[symbol] = {
                "entry_index": context.trade_index,
                "signal_date": signal.get("signal_date"),
                "buy_date": today,
                "holding_days": signal.get("holding_days"),
                "entry_pred_prob": signal.get("pred_prob"),
                "last_pred_prob": signal.get("pred_prob"),
                "entry_target_pct": target_pct,
                "signal_stop_loss_pct": _signal_optional_float(signal, "signal_stop_loss_pct", "stop_loss_pct"),
                "signal_take_profit_pct": _signal_optional_float(signal, "signal_take_profit_pct", "take_profit_pct"),
                "signal_score_exit_entry_ratio": _signal_optional_float(
                    signal, "signal_score_exit_entry_ratio", "score_exit_entry_ratio"
                ),
                "signal_score_continue_entry_ratio": _signal_optional_float(
                    signal, "signal_score_continue_entry_ratio", "score_continue_entry_ratio"
                ),
                "signal_min_holding_days_before_score_exit": _signal_optional_int(
                    signal, "signal_min_holding_days_before_score_exit", "min_holding_days_before_score_exit"
                ),
            }
            open_slots -= 1
            bought_count += 1
    return bought_count


def trade_daily(context):
    sell_daily(context)
    buy_daily(context)


def on_backtest_finished(context, indicator=None, *args, **kwargs):
    print("GM_BACKTEST_FINISHED {}".format(STRATEGY_NAME))
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
        backtest_adjust=_resolve_backtest_adjust(),
        backtest_initial_cash=BACKTEST_INITIAL_CASH,
        backtest_commission_ratio=0.0003,
        backtest_slippage_ratio=BACKTEST_SLIPPAGE_RATIO,
        backtest_match_mode=0,
    )
