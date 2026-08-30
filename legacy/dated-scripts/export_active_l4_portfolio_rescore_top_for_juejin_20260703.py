from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


QUANT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = QUANT_ROOT / "data_file" / "reports" / "strategy_agent_active_l4_prod_repro_grid_20260703"
CACHE_DB = REPORT_DIR / "active_l4_wide_cache.duckdb"
RESCORE = REPORT_DIR / "active_l4_wide_portfolio_rescore_summary.csv"
VARIANT_DIR = REPORT_DIR / "active_l4_wide_portfolio_rescore_juejin_variants"
MANIFEST = REPORT_DIR / "active_l4_wide_portfolio_rescore_juejin_manifest.csv"

SORT_MODES = {
    "score": "pred_prob desc, buy_amount desc",
    "lowgap": "buy_open_gap_raw_pct asc, pred_prob desc",
}


def add_filter(filters: list[str], col: str, op: str, value) -> None:
    if pd.notna(value):
        filters.append(f"{col} {op} {float(value)}")


def export_signal(con: duckdb.DuckDBPyConnection, row: pd.Series) -> pd.DataFrame:
    hold = int(row["hold"])
    topn = int(row["topn"])
    ret_col = f"ret_h{hold}"
    order_expr = SORT_MODES[str(row["sort_mode"])]
    filters = [
        f"score_pct_rank >= {float(row['score_min'])}",
        f"signal_amount >= {float(row['amount_min'])}",
        f"buy_amount >= {float(row['amount_min'])}",
        f"{ret_col} is not null",
    ]
    add_filter(filters, "signal_pct_chg", ">=", row.get("pct_min"))
    add_filter(filters, "signal_pct_chg", "<=", row.get("pct_max"))
    add_filter(filters, "buy_open_gap_raw_pct", ">=", row.get("buy_gap_min"))
    add_filter(filters, "buy_open_gap_raw_pct", "<=", row.get("buy_gap_max"))
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
            signal_date, buy_date, stock_code, name, market, pick_rank as rank,
            pred_prob, score_pct_rank, signal_pct_chg, signal_open_gap_raw_pct,
            buy_open_gap_raw_pct, signal_amount, buy_amount, signal_total_mv,
            buy_total_mv
        from picked
        where pick_rank <= {topn}
        order by buy_date, pick_rank
        """
    ).fetchdf()
    sig["symbol"] = np.where(
        sig["stock_code"].str.endswith(".SH"),
        "SHSE." + sig["stock_code"].str.split(".").str[0],
        "SZSE." + sig["stock_code"].str.split(".").str[0],
    )
    sig["target_pct"] = float(row["target_each"])
    sig["holding_days"] = hold
    sig["max_holding_days"] = hold
    sig["score_exit_entry_ratio"] = 9.99
    sig["min_holding_days_before_score_exit"] = hold
    sig["score_continue_entry_ratio"] = 9.99
    sig["buy_day_market_available"] = True
    sig["buy_day_hard_gate_complete"] = True
    sig["buy_day_st_rejected"] = False
    sig["buy_day_open_limit_up_rejected"] = False
    sig["latest_market_date"] = "20260702"
    return sig[
        [
            "signal_date", "buy_date", "symbol", "stock_code", "name", "rank",
            "pred_prob", "score_pct_rank", "signal_pct_chg", "signal_open_gap_raw_pct",
            "buy_open_gap_raw_pct", "signal_amount", "buy_amount", "signal_total_mv",
            "buy_total_mv", "target_pct", "holding_days", "max_holding_days",
            "score_exit_entry_ratio", "min_holding_days_before_score_exit",
            "score_continue_entry_ratio", "buy_day_market_available",
            "buy_day_hard_gate_complete", "buy_day_st_rejected",
            "buy_day_open_limit_up_rejected", "latest_market_date",
        ]
    ]


def main() -> None:
    VARIANT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RESCORE)
    choices = pd.concat(
        [
            df.sort_values(["portfolio_annual", "portfolio_sharpe"], ascending=False).head(2),
            df[df["portfolio_max_drawdown"] < 0.40].sort_values(
                ["portfolio_annual", "portfolio_sharpe"], ascending=False
            ).head(3),
        ],
        ignore_index=True,
    ).drop_duplicates("case_id")
    con = duckdb.connect(str(CACHE_DB), read_only=True)
    rows = []
    for _, row in choices.iterrows():
        sig = export_signal(con, row)
        name = str(row["name"]) + "_portrescore"
        path = VARIANT_DIR / f"{name}.csv"
        sig.to_csv(path, index=False, encoding="utf-8-sig")
        out = row.to_dict()
        out["name"] = name
        out["signal_file"] = str(path)
        out["signal_rows"] = len(sig)
        out["signal_buy_days"] = sig["buy_date"].nunique()
        rows.append(out)
    con.close()
    manifest = pd.DataFrame(rows)
    manifest.to_csv(MANIFEST, index=False, encoding="utf-8-sig")
    print(manifest[[
        "name", "portfolio_annual", "portfolio_sharpe", "portfolio_max_drawdown",
        "topn", "hold", "target_each", "signal_rows", "signal_buy_days", "signal_file"
    ]].to_string(index=False))
    print(MANIFEST)


if __name__ == "__main__":
    main()
