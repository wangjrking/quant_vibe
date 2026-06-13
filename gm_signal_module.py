"""Signal export helpers for running the project strategy in gm.api."""

from __future__ import annotations

import csv
import math
import sqlite3
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


def load_market_rows_by_trade_date(db_path: str | Path, start: str, end: str) -> dict[str, dict[str, dict]]:
    """Load daily market rows keyed by trade_date -> stock_code."""
    conn = sqlite3.connect(str(Path(db_path)))
    try:
        cursor = conn.execute(
            """
            SELECT trade_date, stock_code, name, pre_close, open
            FROM STOCK_DAILY_DATA
            WHERE trade_date >= ? AND trade_date <= ?
            """,
            (str(start), str(end)),
        )
        grouped: dict[str, dict[str, dict]] = {}
        for trade_date, stock_code, name, pre_close, open_price in cursor:
            grouped.setdefault(str(trade_date), {})[str(stock_code)] = {
                "trade_date": str(trade_date),
                "stock_code": str(stock_code),
                "name": name,
                "pre_close": pre_close,
                "open": open_price,
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
    return row.get("limit_times") not in (None, "", "None")


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
    next_date_by_signal_date = {date: trade_dates[idx + 1] for idx, date in enumerate(trade_dates[:-1])}
    signals = []

    concurrency_cap = None
    if target_total_pct is not None and holding_days:
        expected_concurrent_positions = max(1, int(config.top_k) * int(holding_days))
        if max_positions is not None:
            expected_concurrent_positions = min(expected_concurrent_positions, int(max_positions))
        concurrency_cap = float(target_total_pct) / float(max(expected_concurrent_positions, 1))

    for signal_date in trade_dates[:-1]:
        buy_date = next_date_by_signal_date[signal_date]
        next_day_rows = {
            str(row.get("stock_code") or ""): row
            for row in grouped.get(buy_date, [])
            if row.get("stock_code")
        }
        if market_rows_by_trade_date and buy_date in market_rows_by_trade_date:
            next_day_rows = market_rows_by_trade_date[buy_date]
        day_config = SelectionConfig(
            top_k=config.top_k,
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
        candidates = select_candidates(grouped[signal_date], day_config)
        candidates = [
            candidate
            for candidate in candidates
            if not _is_unbuyable_next_day(next_day_rows.get(str(candidate.get("stock_code") or "")))
        ]
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
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(signals[0].keys()))
        writer.writeheader()
        writer.writerows(signals)
