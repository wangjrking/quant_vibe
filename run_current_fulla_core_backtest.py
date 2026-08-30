"""Fast reproducible core backtests for current full-A model search.

This runner intentionally reads exactly one prediction table from a DuckDB file.
It does not discover or fall back to older temporary prediction tables.
"""

from __future__ import annotations

import argparse
import csv
import duckdb
import json
from pathlib import Path
from typing import Any

from portfolio_backtest_module import PortfolioBacktestConfig, run_portfolio_backtest


def _parse_int_csv(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _parse_float_csv(value: str) -> list[float | None]:
    values: list[float | None] = []
    for item in str(value).split(","):
        text = item.strip()
        if not text:
            continue
        values.append(None if text.lower() in {"none", "null"} else float(text))
    return values


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _sell_col_for_holding_days(days: int) -> str:
    mapping = {1: "post2_open", 3: "post4_open", 5: "post6_open", 10: "post12_open"}
    return mapping.get(days, f"post{days + 1}_open")


def _load_top_rows(db_path: Path, table: str, start: str, end: str, top_n_per_day: int) -> list[dict[str, Any]]:
    sql = f"""
        SELECT *
        FROM (
            SELECT p.*,
                   ROW_NUMBER() OVER (PARTITION BY trade_date ORDER BY pred_prob DESC) AS rn
            FROM {_quote(table)} p
            WHERE trade_date >= ? AND trade_date <= ?
        )
        WHERE rn <= ?
        ORDER BY trade_date, pred_prob DESC
    """
    with duckdb.connect(str(db_path), read_only=True) as conn:
        result = conn.execute(sql, [start, end, int(top_n_per_day)])
        columns = [item[0] for item in result.description]
        rows = [dict(zip(columns, row)) for row in result.fetchall()]
    for row in rows:
        row.pop("rn", None)
    return rows


def _holding_ratio(equity_curve: list[dict[str, Any]]) -> float:
    if not equity_curve:
        return 0.0
    return sum(1 for row in equity_curve if int(row.get("position_count", 0) or 0) > 0) / len(equity_curve)


def _avg_positions(equity_curve: list[dict[str, Any]]) -> float:
    if not equity_curve:
        return 0.0
    return sum(float(row.get("position_count", 0.0) or 0.0) for row in equity_curve) / len(equity_curve)


def _write_rows(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run current full-A core strategy grid on one prediction table.")
    parser.add_argument("--db", required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--top-n-per-day", type=int, default=1000)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", default="1,2,3")
    parser.add_argument("--holding-days", default="1,3,5")
    parser.add_argument("--score-exit-ratio", default="none,0.95")
    parser.add_argument("--min-amount", default="none,800000")
    parser.add_argument("--min-turnover", default="none,2.0")
    parser.add_argument("--max-total-mv", default="none,3000000")
    parser.add_argument("--slippage", type=float, default=0.0015)
    parser.add_argument("--liquidity-slippage", action="store_true")
    parser.add_argument("--max-positions", type=int, default=3)
    parser.add_argument("--min-holding-ratio", type=float, default=0.50)
    parser.add_argument("--min-score-exit-holding-days", default="1")
    args = parser.parse_args(argv)

    db_path = Path(args.db)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_top_rows(db_path, args.table, args.start, args.end, args.top_n_per_day)
    results: list[dict[str, Any]] = []
    for top_k in _parse_int_csv(args.top_k):
        for holding_days in _parse_int_csv(args.holding_days):
            for score_exit_ratio in _parse_float_csv(args.score_exit_ratio):
                for min_amount in _parse_float_csv(args.min_amount):
                        for min_turnover_rate in _parse_float_csv(args.min_turnover):
                            for max_total_mv in _parse_float_csv(args.max_total_mv):
                                for min_score_exit_holding_days in _parse_int_csv(args.min_score_exit_holding_days):
                                    config = PortfolioBacktestConfig(
                                        top_k=top_k,
                                        max_positions=max(top_k, args.max_positions),
                                        holding_days=holding_days,
                                        start_date=args.start,
                                        end_date=args.end,
                                        min_pred_prob=None,
                                        max_atr_ratio=None,
                                        sell_col=_sell_col_for_holding_days(holding_days),
                                        slippage_rate=args.slippage,
                                        liquidity_slippage_enabled=args.liquidity_slippage,
                                        min_amount=min_amount,
                                        min_turnover_rate=min_turnover_rate,
                                        max_total_mv=max_total_mv,
                                        score_exit_ratio=score_exit_ratio,
                                        min_score_exit_holding_days=min_score_exit_holding_days,
                                    )
                                    backtest = run_portfolio_backtest(rows, config)
                                    metrics = backtest["metrics"]
                                    hold_ratio = _holding_ratio(backtest["equity_curve"])
                                    results.append(
                                        {
                                            "table": args.table,
                                            "prefilter": f"top{args.top_n_per_day}_per_day",
                                            "top_k": top_k,
                                            "holding_days": holding_days,
                                            "score_exit_ratio": score_exit_ratio,
                                            "min_score_exit_holding_days": min_score_exit_holding_days,
                                            "min_amount": min_amount,
                                            "min_turnover_rate": min_turnover_rate,
                                            "max_total_mv": max_total_mv,
                                            "slippage": args.slippage,
                                            "liquidity_slippage": bool(args.liquidity_slippage),
                                            "holding_ratio": hold_ratio,
                                            "meets_holding_ratio": hold_ratio >= args.min_holding_ratio,
                                            "avg_positions": _avg_positions(backtest["equity_curve"]),
                                            **metrics,
                                        }
                                    )

    results.sort(key=lambda row: (row["meets_holding_ratio"], row["annualized_return"]), reverse=True)
    _write_rows(results, output_dir / "core_results.csv")
    (output_dir / "top20.json").write_text(
        json.dumps(results[:20], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"rows_loaded": len(rows), "results": len(results), "best": results[0] if results else None}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
