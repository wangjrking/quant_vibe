from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


QUANT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = QUANT_ROOT / "data_file" / "reports" / "strategy_agent_active_l4_prod_repro_grid_20260703"
CACHE_DB = REPORT_DIR / "active_l4_wide_cache.duckdb"
VARIANT_DIR = REPORT_DIR / "variants_0939_real_target_topn"
MANIFEST = REPORT_DIR / "variants_0939_real_target_topn_manifest.csv"


def export_case(con: duckdb.DuckDBPyConnection, case: dict) -> tuple[pd.DataFrame, Path]:
    filters = [
        f"score_pct_rank >= {case['score_min']}",
        "signal_amount >= 50000",
        "buy_amount >= 50000",
        "ret_h2 is not null",
        "buy_open_gap_raw_pct >= -3.0",
        "buy_open_gap_raw_pct <= 1.5",
    ]
    where = " and ".join(filters)
    sig = con.execute(
        f"""
        with picked as (
            select
                *,
                row_number() over(partition by buy_date order by buy_open_gap_raw_pct asc, pred_prob desc) as pick_rank
            from active_l4_wide
            where {where}
        )
        select
            signal_date, buy_date, stock_code, name, market, pick_rank as rank,
            pred_prob, score_pct_rank, signal_pct_chg, signal_open_gap_raw_pct,
            buy_open_gap_raw_pct, signal_amount, buy_amount, signal_total_mv,
            buy_total_mv
        from picked
        where pick_rank <= {case['topn']}
        order by buy_date, pick_rank
        """
    ).fetchdf()
    sig["symbol"] = np.where(
        sig["stock_code"].str.endswith(".SH"),
        "SHSE." + sig["stock_code"].str.split(".").str[0],
        "SZSE." + sig["stock_code"].str.split(".").str[0],
    )
    sig["target_pct"] = case["target_each"]
    sig["holding_days"] = 2
    sig["max_holding_days"] = 2
    sig["score_exit_entry_ratio"] = 9.99
    sig["min_holding_days_before_score_exit"] = 2
    sig["score_continue_entry_ratio"] = 9.99
    sig["buy_day_market_available"] = True
    sig["buy_day_hard_gate_complete"] = True
    sig["buy_day_st_rejected"] = False
    sig["buy_day_open_limit_up_rejected"] = False
    sig["latest_market_date"] = "20260702"
    cols = [
        "signal_date", "buy_date", "symbol", "stock_code", "name", "rank",
        "pred_prob", "score_pct_rank", "signal_pct_chg", "signal_open_gap_raw_pct",
        "buy_open_gap_raw_pct", "signal_amount", "buy_amount", "signal_total_mv",
        "buy_total_mv", "target_pct", "holding_days", "max_holding_days",
        "score_exit_entry_ratio", "min_holding_days_before_score_exit",
        "score_continue_entry_ratio", "buy_day_market_available",
        "buy_day_hard_gate_complete", "buy_day_st_rejected",
        "buy_day_open_limit_up_rejected", "latest_market_date",
    ]
    path = VARIANT_DIR / f"{case['name']}.csv"
    sig[cols].to_csv(path, index=False, encoding="utf-8-sig")
    return sig, path


def main() -> None:
    VARIANT_DIR.mkdir(parents=True, exist_ok=True)
    cases = [
        dict(name="0939_s95_top1_t55_h2", score_min=0.95, topn=1, target_each=0.55),
        dict(name="0939_s95_top1_t60_h2", score_min=0.95, topn=1, target_each=0.60),
        dict(name="0939_s95_top1_t70_h2", score_min=0.95, topn=1, target_each=0.70),
        dict(name="0939_s95_top2_t30_h2", score_min=0.95, topn=2, target_each=0.30),
        dict(name="0939_s95_top2_t25_h2", score_min=0.95, topn=2, target_each=0.25),
        dict(name="0939_s95_top3_t20_h2", score_min=0.95, topn=3, target_each=0.20),
        dict(name="0939_s90_top2_t30_h2", score_min=0.90, topn=2, target_each=0.30),
        dict(name="0939_s90_top3_t20_h2", score_min=0.90, topn=3, target_each=0.20),
    ]
    con = duckdb.connect(str(CACHE_DB), read_only=True)
    rows = []
    for case in cases:
        sig, path = export_case(con, case)
        row = dict(case)
        row["signal_file"] = str(path)
        row["signal_rows"] = len(sig)
        row["signal_buy_days"] = sig["buy_date"].nunique()
        rows.append(row)
    con.close()
    manifest = pd.DataFrame(rows)
    manifest.to_csv(MANIFEST, index=False, encoding="utf-8-sig")
    print(manifest.to_string(index=False))
    print(MANIFEST)


if __name__ == "__main__":
    main()
