"""Signal export helpers for running the project strategy in gm.api."""

from __future__ import annotations

import csv
import math
import sqlite3
from bisect import bisect_right
from pathlib import Path

from selection_module import SelectionConfig, select_candidates


def to_gm_symbol(stock_code: str) -> str:
    """Convert Tushare-style stock codes to gm.api symbols."""
    code = str(stock_code or "").strip().upper()
    if code.endswith(".SH"):
        return f"SHSE.{code[:6]}"
    if code.endswith(".SZ"):
        return f"SZSE.{code[:6]}"
    if code.endswith(".BJ"):
        return f"BJSE.{code[:6]}"
    if code.startswith(("SHSE.", "SZSE.", "BJSE.")):
        return code
    raise ValueError(f"Unsupported stock code for gm symbol conversion: {stock_code}")


def _group_by_trade_date(rows):
    grouped = {}
    for row in rows:
        date = str(row.get("trade_date", "") or "")
        if date:
            grouped.setdefault(date, []).append(row)
    return grouped


def _next_buy_date_by_signal_date(
    grouped_rows: dict[str, list[dict]],
    market_rows_by_trade_date: dict[str, dict[str, dict]] | None,
) -> dict[str, str]:
    signal_dates = sorted(grouped_rows)
    if not signal_dates:
        return {}
    calendar_dates = set(signal_dates)
    if market_rows_by_trade_date:
        calendar_dates.update(str(date) for date in market_rows_by_trade_date.keys() if date)
    ordered_calendar = sorted(calendar_dates)
    next_date_by_signal_date: dict[str, str] = {}
    for signal_date in signal_dates:
        next_idx = bisect_right(ordered_calendar, signal_date)
        if next_idx < len(ordered_calendar):
            next_date_by_signal_date[signal_date] = ordered_calendar[next_idx]
    return next_date_by_signal_date


def _merge_rows_with_market_snapshot(rows: list[dict], market_rows_for_date: dict[str, dict] | None) -> list[dict]:
    if not rows or not market_rows_for_date:
        return rows
    merged_rows = []
    for row in rows:
        stock_code = str(row.get("stock_code", "") or "")
        market_row = market_rows_for_date.get(stock_code) if stock_code else None
        if not market_row:
            merged_rows.append(row)
            continue
        merged = dict(market_row)
        for key, value in row.items():
            if value in (None, "", "None") and merged.get(key) not in (None, "", "None"):
                continue
            merged[key] = value
        merged_rows.append(merged)
    return merged_rows


def load_market_rows_by_trade_date(db_path: str | Path, start: str, end: str) -> dict[str, dict[str, dict]]:
    """Load daily market rows keyed by trade_date -> stock_code."""
    conn = sqlite3.connect(str(Path(db_path)))
    try:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        if "STOCK_DAILY_DATA" in tables:
            stock_daily_cols = {
                row[1]
                for row in conn.execute("PRAGMA table_info(STOCK_DAILY_DATA)").fetchall()
            }
            atr_expr = "atr_qfq" if "atr_qfq" in stock_daily_cols else "NULL AS atr_qfq"
            st_type_expr = "ST_TYPE AS st_type" if "ST_TYPE" in stock_daily_cols else "NULL AS st_type"
            st_type_name_expr = "ST_TYPE_name AS st_type_name" if "ST_TYPE_name" in stock_daily_cols else "NULL AS st_type_name"
            query = """
                SELECT trade_date, stock_code, name, pre_close, open, close,
                       amount, turnover_rate, total_mv,
                       {atr_expr},
                       {st_type_expr}, {st_type_name_expr}, limit_times
                FROM STOCK_DAILY_DATA
                WHERE trade_date >= ? AND trade_date <= ?
            """.format(atr_expr=atr_expr, st_type_expr=st_type_expr, st_type_name_expr=st_type_name_expr)
        elif "stk_factor" in tables or "STK_FACTOR" in tables:
            factor_table = "stk_factor" if "stk_factor" in tables else "STK_FACTOR"
            query = f"""
                SELECT f.trade_date,
                       CASE
                           WHEN f.ts_code LIKE '%.SH' THEN 'SHSE.' || substr(f.ts_code, 1, 6)
                           WHEN f.ts_code LIKE '%.SZ' THEN 'SZSE.' || substr(f.ts_code, 1, 6)
                           WHEN f.ts_code LIKE '%.BJ' THEN 'BJSE.' || substr(f.ts_code, 1, 6)
                           ELSE f.ts_code
                       END AS stock_code,
                       b.name AS name,
                       f.pre_close AS pre_close,
                       f.open AS open,
                       f.close AS close,
                       NULL AS amount,
                       NULL AS turnover_rate,
                       NULL AS total_mv,
                       NULL AS atr_qfq,
                       NULL AS st_type,
                       NULL AS st_type_name,
                       NULL AS limit_times
                FROM "{factor_table}" f
                LEFT JOIN stock_basic_data b ON f.ts_code = b.ts_code
                WHERE f.trade_date >= ? AND f.trade_date <= ?
            """
        else:
            query = """
                SELECT trade_date,
                       CASE
                           WHEN ts_code LIKE '%.SH' THEN 'SHSE.' || substr(ts_code, 1, 6)
                           WHEN ts_code LIKE '%.SZ' THEN 'SZSE.' || substr(ts_code, 1, 6)
                           WHEN ts_code LIKE '%.BJ' THEN 'BJSE.' || substr(ts_code, 1, 6)
                           ELSE ts_code
                       END AS stock_code,
                       NULL AS name,
                       pre_close,
                       open,
                       close,
                       NULL AS amount,
                       NULL AS turnover_rate,
                       NULL AS total_mv,
                       NULL AS atr_qfq,
                       NULL AS st_type,
                       NULL AS st_type_name,
                       NULL AS limit_times
                FROM daily_data
                WHERE trade_date >= ? AND trade_date <= ?
            """
        cursor = conn.execute(query, (str(start), str(end)))
        grouped: dict[str, dict[str, dict]] = {}
        for (
            trade_date,
            stock_code,
            name,
            pre_close,
            open_price,
            close_price,
            amount,
            turnover_rate,
            total_mv,
            atr_qfq,
            st_type,
            st_type_name,
            limit_times,
        ) in cursor:
            grouped.setdefault(str(trade_date), {})[str(stock_code)] = {
                "trade_date": str(trade_date),
                "stock_code": str(stock_code),
                "name": name,
                "pre_close": pre_close,
                "open": open_price,
                "close": close_price,
                "amount": amount,
                "turnover_rate": turnover_rate,
                "total_mv": total_mv,
                "atr_qfq": atr_qfq,
                "st_type": st_type,
                "st_type_name": st_type_name,
                "limit_times": limit_times,
            }
        return grouped
    finally:
        conn.close()


def _to_float(value):
    if value is None or value == "":
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _is_bj_stock(stock_code: str | None) -> bool:
    code = str(stock_code or "").strip().upper()
    return code.endswith(".BJ") or code.startswith("BJSE.")


def _limit_up_pct(stock_code: str, name: str | None = None, st_type=None) -> float:
    code = str(stock_code or "")
    name_text = str(name or "")
    if st_type or name_text.startswith("ST") or name_text.startswith("*ST"):
        return 0.05
    if code.startswith(("300", "301", "688")):
        return 0.20
    return 0.10


def _is_current_limit(row: dict | None) -> bool:
    if not row:
        return False
    value = row.get("limit_times")
    if value in (None, "", "None"):
        return False
    numeric_value = _to_float(value)
    if numeric_value is not None:
        return numeric_value > 0.0
    return True


def _is_unbuyable_next_day(next_day_row: dict | None) -> bool:
    if not next_day_row:
        return False
    if _is_current_limit(next_day_row):
        return True
    pre_close = _to_float(next_day_row.get("pre_close"))
    open_price = _to_float(next_day_row.get("open"))
    if pre_close is None or open_price is None or pre_close <= 0 or open_price <= 0:
        return False
    pct = _limit_up_pct(next_day_row.get("stock_code"), next_day_row.get("name"), next_day_row.get("st_type"))
    upper = pre_close * (1.0 + pct)
    lower = pre_close * (1.0 - pct)
    return open_price >= upper * 0.995 or open_price <= lower * 1.005


def _signal_weights(candidates: list[dict], pred_col: str, mode: str) -> list[float]:
    if not candidates:
        return []
    normalized_mode = str(mode or "equal").lower()
    if normalized_mode == "equal":
        return [1.0] * len(candidates)
    if normalized_mode == "rank":
        return [float(len(candidates) - idx) for idx in range(len(candidates))]
    if normalized_mode == "score":
        preds = [float(candidate.get(pred_col) or 0.0) for candidate in candidates]
        floor = min(preds)
        return [max(pred - floor, 0.0) + 1e-6 for pred in preds]
    raise ValueError(f"unsupported weight_mode: {mode}")


def _liquidity_weight_multiplier(
    candidate: dict,
    amount_col: str,
    turnover_col: str,
    min_amount: float | None,
    min_turnover_rate: float | None,
    mid_scale: float,
    low_scale: float,
) -> float:
    amount = candidate.get(amount_col)
    turnover = candidate.get(turnover_col)
    try:
        amount = float(amount) if amount not in (None, "", "None") else None
    except (TypeError, ValueError):
        amount = None
    try:
        turnover = float(turnover) if turnover not in (None, "", "None") else None
    except (TypeError, ValueError):
        turnover = None

    misses = 0
    if min_amount is not None and (amount is None or amount < min_amount):
        misses += 1
    if min_turnover_rate is not None and (turnover is None or turnover < min_turnover_rate):
        misses += 1
    if misses >= 2:
        return low_scale
    if misses == 1:
        return mid_scale
    return 1.0


def build_gm_signal_rows(
    rows,
    config: SelectionConfig | None = None,
    market_rows_by_trade_date: dict[str, dict[str, dict]] | None = None,
    holding_days: int | None = None,
    max_positions: int | None = None,
    weight_mode: str = "equal",
    target_total_pct: float | None = None,
    liquidity_target_pct_enabled: bool = False,
    liquidity_amount_col: str = "amount",
    liquidity_turnover_col: str = "turnover_rate",
    liquidity_min_amount: float | None = None,
    liquidity_min_turnover_rate: float | None = None,
    liquidity_mid_scale: float = 0.8,
    liquidity_low_scale: float = 0.6,
) -> list[dict]:
    """Build signal rows keyed by next-day execution date for gm backtests."""
    config = config or SelectionConfig()
    grouped = _group_by_trade_date(rows)
    trade_dates = sorted(grouped)
    next_date_by_signal_date = _next_buy_date_by_signal_date(grouped, market_rows_by_trade_date)
    signals = []

    concurrency_cap = None
    if target_total_pct is not None and holding_days:
        expected_concurrent_positions = max(1, int(config.top_k) * int(holding_days))
        if max_positions is not None:
            expected_concurrent_positions = min(expected_concurrent_positions, int(max_positions))
        concurrency_cap = float(target_total_pct) / float(max(expected_concurrent_positions, 1))

    for signal_date in trade_dates:
        buy_date = next_date_by_signal_date.get(signal_date)
        if not buy_date:
            continue
        next_day_rows = {
            str(row.get("stock_code") or ""): row
            for row in grouped.get(buy_date, [])
            if row.get("stock_code")
        }
        signal_day_rows = grouped[signal_date]
        signal_day_market_rows = None
        if market_rows_by_trade_date and buy_date in market_rows_by_trade_date:
            next_day_rows = market_rows_by_trade_date[buy_date]
        if market_rows_by_trade_date and signal_date in market_rows_by_trade_date:
            signal_day_market_rows = market_rows_by_trade_date[signal_date]
        selection_rows = _merge_rows_with_market_snapshot(signal_day_rows, signal_day_market_rows)
        selection_buffer_k = max(int(config.top_k), min(300, int(config.top_k) * 20 + 20))
        day_config = SelectionConfig(
            top_k=selection_buffer_k,
            trade_date=signal_date,
            pred_col=config.pred_col,
            min_pred_prob=config.min_pred_prob,
            min_pred_quantile=config.min_pred_quantile,
            max_atr_ratio=config.max_atr_ratio,
            min_amount=config.min_amount,
            min_turnover_rate=config.min_turnover_rate,
            max_total_mv=config.max_total_mv,
            max_per_industry=config.max_per_industry,
            exclude_st=config.exclude_st,
            exclude_current_limit=config.exclude_current_limit,
        )
        candidates = select_candidates(selection_rows, day_config)
        candidates = [
            candidate
            for candidate in candidates
            if not _is_bj_stock(candidate.get("stock_code"))
        ]
        candidates = [
            candidate
            for candidate in candidates
            if not _is_unbuyable_next_day(next_day_rows.get(str(candidate.get("stock_code") or "")))
        ]
        candidates = candidates[: int(config.top_k)]
        for rank, candidate in enumerate(candidates, start=1):
            candidate["rank"] = rank
        weights = _signal_weights(candidates, config.pred_col, weight_mode)
        if liquidity_target_pct_enabled and candidates:
            weights = [
                weight
                * _liquidity_weight_multiplier(
                    candidate,
                    liquidity_amount_col,
                    liquidity_turnover_col,
                    liquidity_min_amount,
                    liquidity_min_turnover_rate,
                    liquidity_mid_scale,
                    liquidity_low_scale,
                )
                for candidate, weight in zip(candidates, weights)
            ]
        total_weight = sum(weights) or 1.0
        for candidate, weight in zip(candidates, weights):
            target_pct = (target_total_pct * weight / total_weight) if target_total_pct is not None else None
            if target_pct is not None and concurrency_cap is not None:
                target_pct = min(target_pct, concurrency_cap)
            signals.append(
                {
                    "signal_date": signal_date,
                    "buy_date": buy_date,
                    "symbol": to_gm_symbol(candidate.get("stock_code")),
                    "stock_code": candidate.get("stock_code"),
                    "name": candidate.get("name"),
                    "rank": candidate.get("rank"),
                    "pred_prob": candidate.get(config.pred_col),
                    "atr_ratio": candidate.get("atr_ratio"),
                    "holding_days": holding_days,
                    "target_pct": target_pct,
                }
            )

    return signals


def write_gm_signals_csv(signals: list[dict], output_path: str | Path) -> None:
    if not signals:
        raise ValueError("No gm signals to write.")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for signal in signals:
        for field in signal.keys():
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(signals)
