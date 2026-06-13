"""Daily stock candidate selection from prediction output."""

from __future__ import annotations

import argparse
import csv
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SelectionConfig:
    top_k: int = 3
    trade_date: str | None = None
    pred_col: str = "pred_prob"
    min_pred_prob: float = 0.01
    min_pred_quantile: float | None = None
    max_atr_ratio: float = 0.10
    min_amount: float | None = None
    min_turnover_rate: float | None = None
    max_total_mv: float | None = None
    max_per_industry: int = 2
    exclude_st: bool = True
    exclude_delisting: bool = True
    exclude_current_limit: bool = True


def _value(row, key, default=None):
    if isinstance(row, dict):
        return row.get(key, default)
    try:
        return row[key]
    except Exception:
        return default


def _to_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_st(row):
    name = str(_value(row, "name", "") or "")
    return bool(_value(row, "st_type")) or name.startswith("ST") or name.startswith("*ST")


def _is_delisting(row):
    name = str(_value(row, "name", "") or "")
    return "\u9000\u5e02" in name or name.startswith("\u9000")


def _is_current_limit(row):
    return _value(row, "limit_times") not in (None, "", "None")


def _atr_ratio(row):
    close = _to_float(_value(row, "close"))
    atr = _to_float(_value(row, "atr_qfq"))
    if close is None or atr is None or close <= 0:
        return None
    return atr / close


def latest_trade_date(rows):
    dates = [str(_value(row, "trade_date", "")) for row in rows if _value(row, "trade_date")]
    if not dates:
        return None
    return max(dates)


def _quantile_threshold(values: list[float], quantile: float | None) -> float | None:
    if quantile is None or not values:
        return None
    if quantile < 0 or quantile > 1:
        raise ValueError("min_pred_quantile must be between 0 and 1.")
    ordered = sorted(values)
    index = math.ceil((len(ordered) - 1) * quantile)
    return ordered[index]


def select_candidates(rows, config=None):
    config = config or SelectionConfig()
    target_date = config.trade_date or latest_trade_date(rows)
    if not target_date:
        return []

    pred_values = [
        pred
        for row in rows
        if str(_value(row, "trade_date", "")) == target_date
        for pred in [_to_float(_value(row, config.pred_col))]
        if pred is not None
    ]
    quantile_threshold = _quantile_threshold(pred_values, config.min_pred_quantile)

    candidates = []
    for row in rows:
        if str(_value(row, "trade_date", "")) != target_date:
            continue
        pred = _to_float(_value(row, config.pred_col))
        if pred is None:
            continue
        if config.min_pred_prob is not None and pred < config.min_pred_prob:
            continue
        if quantile_threshold is not None and pred < quantile_threshold:
            continue
        if config.exclude_st and _is_st(row):
            continue
        if config.exclude_delisting and _is_delisting(row):
            continue
        if config.exclude_current_limit and _is_current_limit(row):
            continue
        ratio = _atr_ratio(row)
        if config.max_atr_ratio is not None and (ratio is None or ratio > config.max_atr_ratio):
            continue
        amount = _to_float(_value(row, "amount"))
        if config.min_amount is not None and (amount is None or amount < config.min_amount):
            continue
        turnover_rate = _to_float(_value(row, "turnover_rate"))
        if config.min_turnover_rate is not None and (turnover_rate is None or turnover_rate < config.min_turnover_rate):
            continue
        total_mv = _to_float(_value(row, "total_mv"))
        if config.max_total_mv is not None and (total_mv is None or total_mv > config.max_total_mv):
            continue
        enriched = dict(row)
        enriched["atr_ratio"] = ratio
        candidates.append(enriched)

    candidates.sort(
        key=lambda row: (
            -float(row[config.pred_col]),
            row["atr_ratio"] if row["atr_ratio"] is not None else 999999.0,
            str(row.get("stock_code", "")),
        )
    )
    selected = []
    industry_counts = {}
    for row in candidates:
        industry = row.get("industry_encode")
        industry_counts.setdefault(industry, 0)
        if industry_counts[industry] >= config.max_per_industry:
            continue
        industry_counts[industry] += 1
        row = dict(row)
        row["rank"] = len(selected) + 1
        selected.append(row)
        if len(selected) >= config.top_k:
            break
    return selected


def read_latest_rows(db_path, table, trade_date=None):
    path = Path(db_path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    if trade_date is None:
        trade_date = conn.execute(f'SELECT MAX(trade_date) FROM "{table}"').fetchone()[0]
    rows = [dict(row) for row in conn.execute(f'SELECT * FROM "{table}" WHERE trade_date = ? ORDER BY pred_prob DESC', (trade_date,))]
    conn.close()
    return trade_date, rows


def write_candidates_csv(candidates, output_path):
    if not candidates:
        return
    with open(output_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(candidates[0].keys()))
        writer.writeheader()
        writer.writerows(candidates)


def parse_args(argv=None):
    defaults = SelectionConfig()
    parser = argparse.ArgumentParser(description="Select daily buy candidates from prediction output.")
    parser.add_argument("--db", default="data_file/odb.db")
    parser.add_argument("--table", default="stock_predict_data_10d_yield_rate")
    parser.add_argument("--date", dest="trade_date")
    parser.add_argument("--top-k", type=int, default=defaults.top_k)
    parser.add_argument("--min-pred", type=float, default=defaults.min_pred_prob)
    parser.add_argument("--min-pred-quantile", type=float, default=defaults.min_pred_quantile)
    parser.add_argument("--max-atr-ratio", type=float, default=defaults.max_atr_ratio)
    parser.add_argument("--max-per-industry", type=int, default=defaults.max_per_industry)
    parser.add_argument("--output")
    return parser.parse_args(argv)


def main():
    args = parse_args()

    trade_date, rows = read_latest_rows(args.db, args.table, args.trade_date)
    config = SelectionConfig(
        top_k=args.top_k,
        trade_date=trade_date,
        min_pred_prob=args.min_pred,
        min_pred_quantile=args.min_pred_quantile,
        max_atr_ratio=args.max_atr_ratio,
        max_per_industry=args.max_per_industry,
    )
    candidates = select_candidates(rows, config)
    for row in candidates:
        print(
            f"{row['rank']:>2} {row.get('trade_date')} {row.get('stock_code')} "
            f"{row.get('name')} pred={float(row.get('pred_prob')):.6f} atr_ratio={row.get('atr_ratio'):.4f}"
        )
    if args.output:
        write_candidates_csv(candidates, args.output)
        print(f"selection_csv: {args.output}")


if __name__ == "__main__":
    main()
