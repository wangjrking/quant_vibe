from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


QUANT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = QUANT_ROOT / "data_file" / "reports" / "strategy_agent_active_l4_prod_repro_grid_20260703"
CACHE_DB = REPORT_DIR / "active_l4_wide_cache.duckdb"
SUMMARY_CSV = REPORT_DIR / "active_l4_wide_sql_stage1_summary.csv"
VARIANT_DIR = REPORT_DIR / "active_l4_wide_sql_stage1_top_variants"
MANIFEST = REPORT_DIR / "active_l4_wide_sql_stage1_top_variant_manifest.csv"


SORT_MODES = {
    "score": "pred_prob desc, buy_amount desc",
    "lowgap": "buy_open_gap_raw_pct asc, pred_prob desc",
}


def add_filter(filters: list[str], col: str, op: str, value) -> None:
    if pd.notna(value):
        filters.append(f"{col} {op} {float(value)}")


def main() -> None:
    VARIANT_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(SUMMARY_CSV).sort_values(["annual", "sharpe"], ascending=False).head(3)
    con = duckdb.connect(str(CACHE_DB), read_only=True)
    rows = []
    for _, r in summary.iterrows():
        name = str(r["name"])
        hold = int(r["hold"])
        topn = int(r["topn"])
        ret_col = f"ret_h{hold}"
        order_expr = SORT_MODES[str(r["sort_mode"])]
        filters = [
            f"score_pct_rank >= {float(r['score_min'])}",
            f"signal_amount >= {float(r['amount_min'])}",
            f"buy_amount >= {float(r['amount_min'])}",
            f"{ret_col} is not null",
        ]
        add_filter(filters, "signal_pct_chg", ">=", r.get("pct_min"))
        add_filter(filters, "signal_pct_chg", "<=", r.get("pct_max"))
        add_filter(filters, "buy_open_gap_raw_pct", ">=", r.get("buy_gap_min"))
        add_filter(filters, "buy_open_gap_raw_pct", "<=", r.get("buy_gap_max"))
        where = " and ".join(filters)
        sig = con.execute(
            f"""
            with picked as (
                select
                    *,
                    row_number() over(partition by buy_date order by {order_expr}) as pick_rank
                from active_l4_wide
                where {where}
            )
            select
                signal_date, buy_date, stock_code, name, market,
                case
                    when stock_code like '%.SH' then 'SHSE.' || split_part(stock_code, '.', 1)
                    when stock_code like '%.SZ' then 'SZSE.' || split_part(stock_code, '.', 1)
                    else stock_code
                end as symbol,
                pick_rank as rank,
                {float(r['target_each'])} as target_pct,
                pred_prob as signal_score,
                pred_prob, score_pct_rank,
                signal_pct_chg, signal_open_gap_raw_pct, buy_open_gap_raw_pct,
                signal_amount, buy_amount, signal_total_mv, buy_total_mv,
                {hold} as holding_days,
                {hold} as max_holding_days,
                9.99 as score_exit_entry_ratio,
                1 as min_holding_days_before_score_exit,
                9.99 as score_continue_entry_ratio,
                true as buy_day_market_available,
                true as buy_day_hard_gate_complete,
                false as buy_day_st_rejected,
                false as buy_day_open_limit_up_rejected,
                '20260702' as latest_market_date
            from picked
            where pick_rank <= {topn}
            order by buy_date, pick_rank
            """
        ).fetchdf()
        sig = sig[
            [
                "signal_date", "buy_date", "symbol", "stock_code", "name", "rank",
                "pred_prob", "signal_score", "score_pct_rank", "signal_pct_chg",
                "signal_open_gap_raw_pct", "buy_open_gap_raw_pct", "signal_amount",
                "buy_amount", "signal_total_mv", "buy_total_mv", "target_pct",
                "holding_days", "max_holding_days", "score_exit_entry_ratio",
                "min_holding_days_before_score_exit", "score_continue_entry_ratio",
                "buy_day_market_available", "buy_day_hard_gate_complete",
                "buy_day_st_rejected", "buy_day_open_limit_up_rejected",
                "latest_market_date",
            ]
        ]
        path = VARIANT_DIR / f"{name}.csv"
        sig.to_csv(path, index=False, encoding="utf-8-sig")
        row = r.to_dict()
        row["signal_file"] = str(path)
        row["signal_rows"] = len(sig)
        row["signal_buy_days"] = sig["buy_date"].nunique()
        rows.append(row)
    con.close()
    manifest = pd.DataFrame(rows)
    manifest.to_csv(MANIFEST, index=False, encoding="utf-8-sig")
    print(manifest[["name", "annual", "sharpe", "max_drawdown", "signal_rows", "signal_buy_days", "signal_file"]].to_string(index=False))
    print(MANIFEST)


if __name__ == "__main__":
    main()
