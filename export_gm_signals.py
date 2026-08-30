from __future__ import annotations

import argparse
from pathlib import Path

from backtest_module import read_prediction_rows
from gm_signal_module import build_gm_signal_rows, load_market_rows_by_trade_date, write_gm_signals_csv
from prediction_manifest import resolve_market_db_path, resolve_prediction_source
from selection_module import SelectionConfig


def parse_args(argv=None):
    defaults = SelectionConfig()
    parser = argparse.ArgumentParser(description="Export next-day gm.api signals from prediction rows.")
    parser.add_argument("--prediction-manifest")
    parser.add_argument("--legacy-reproduction", action="store_true")
    parser.add_argument("--db")
    parser.add_argument("--table")
    parser.add_argument("--market-db")
    parser.add_argument("--start", default="20240604")
    parser.add_argument("--end", default="20260604")
    parser.add_argument("--stock-pool")
    parser.add_argument("--top-k", type=int, default=defaults.top_k)
    parser.add_argument("--min-pred", default=str(defaults.min_pred_prob))
    parser.add_argument("--min-pred-quantile", type=float, default=defaults.min_pred_quantile)
    parser.add_argument("--max-atr-ratio", default=str(defaults.max_atr_ratio))
    parser.add_argument("--min-amount", type=float)
    parser.add_argument("--min-turnover-rate", type=float)
    parser.add_argument("--max-total-mv", type=float)
    parser.add_argument("--max-per-industry", type=int, default=defaults.max_per_industry)
    parser.add_argument("--weight-mode", default="equal", choices=["equal", "rank", "score"])
    parser.add_argument("--target-total-pct", type=float)
    parser.add_argument("--max-positions", type=int)
    parser.add_argument("--holding-days", type=int)
    parser.add_argument("--liquidity-target-pct", action="store_true")
    parser.add_argument("--liquidity-min-amount", type=float)
    parser.add_argument("--liquidity-min-turnover-rate", type=float)
    parser.add_argument("--liquidity-mid-scale", type=float, default=0.8)
    parser.add_argument("--liquidity-low-scale", type=float, default=0.6)
    parser.add_argument("--output", default="data_file/gm_signals_2y_ic160.csv")
    return parser.parse_args(argv)


def enrich_prediction_rows_with_market_rows(rows, market_rows_by_trade_date):
    if not rows or not market_rows_by_trade_date:
        return rows
    enriched_rows = []
    market_fields = (
        "name",
        "pre_close",
        "open",
        "amount",
        "turnover_rate",
        "total_mv",
        "st_type",
        "limit_times",
    )
    for row in rows:
        enriched = dict(row)
        market_row = (market_rows_by_trade_date.get(str(row.get("trade_date") or ""), {}) or {}).get(
            str(row.get("stock_code") or "")
        )
        if market_row:
            for field in market_fields:
                if enriched.get(field) in (None, "", "None"):
                    enriched[field] = market_row.get(field)
        enriched_rows.append(enriched)
    return enriched_rows


def main(argv=None):
    args = parse_args(argv)
    min_pred = None if str(args.min_pred).strip().lower() in {"none", "null", ""} else float(args.min_pred)
    max_atr = None if str(args.max_atr_ratio).strip().lower() in {"none", "null", ""} else float(args.max_atr_ratio)
    source = resolve_prediction_source(
        prediction_manifest=args.prediction_manifest,
        legacy_reproduction=bool(args.legacy_reproduction),
        db_path=args.db,
        table=args.table,
    )
    market_db = resolve_market_db_path(source, args.market_db)
    rows = read_prediction_rows(
        source["db_path"],
        source["table"],
        args.start,
        args.end,
        stock_pool_path=args.stock_pool,
        source_type=source.get("source_type", "sqlite_table"),
    )
    market_rows_by_trade_date = load_market_rows_by_trade_date(market_db, args.start, args.end)
    rows = enrich_prediction_rows_with_market_rows(rows, market_rows_by_trade_date)
    signals = build_gm_signal_rows(
        rows,
        SelectionConfig(
            top_k=args.top_k,
            min_pred_prob=min_pred,
            min_pred_quantile=args.min_pred_quantile,
            max_atr_ratio=max_atr,
            min_amount=args.min_amount,
            min_turnover_rate=args.min_turnover_rate,
            max_total_mv=args.max_total_mv,
            max_per_industry=args.max_per_industry,
        ),
        market_rows_by_trade_date=market_rows_by_trade_date,
        holding_days=args.holding_days,
        max_positions=args.max_positions,
        weight_mode=args.weight_mode,
        target_total_pct=args.target_total_pct,
        liquidity_target_pct_enabled=args.liquidity_target_pct,
        liquidity_min_amount=args.liquidity_min_amount,
        liquidity_min_turnover_rate=args.liquidity_min_turnover_rate,
        liquidity_mid_scale=args.liquidity_mid_scale,
        liquidity_low_scale=args.liquidity_low_scale,
    )
    write_gm_signals_csv(signals, args.output)
    dates = sorted({row["buy_date"] for row in signals})
    print(f"signals: {len(signals)}")
    print(f"buy_days: {len(dates)}")
    if dates:
        print(f"buy_date_range: {dates[0]}-{dates[-1]}")
    print(f"output: {Path(args.output)}")


if __name__ == "__main__":
    main()
