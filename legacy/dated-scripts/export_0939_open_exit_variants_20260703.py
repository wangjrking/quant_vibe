from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


QUANT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = QUANT_ROOT / "data_file" / "reports" / "strategy_agent_active_l4_prod_repro_grid_20260703"
CACHE_DB = REPORT_DIR / "active_l4_wide_cache.duckdb"
VARIANT_DIR = REPORT_DIR / "variants_0939_open_exit"
MANIFEST = REPORT_DIR / "variants_0939_open_exit_manifest.csv"


def export_case(con: duckdb.DuckDBPyConnection, case: dict) -> tuple[pd.DataFrame, Path]:
    filters = [
        "score_pct_rank >= 0.95",
        "signal_amount >= 50000",
        "buy_amount >= 50000",
        "ret_h2 is not null",
    ]
    if case["gap_min"] is not None:
        filters.append(f"buy_open_gap_raw_pct >= {case['gap_min']}")
    if case["gap_max"] is not None:
        filters.append(f"buy_open_gap_raw_pct <= {case['gap_max']}")
    if case.get("signal_pct_max") is not None:
        filters.append(f"signal_pct_chg <= {case['signal_pct_max']}")
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
        where pick_rank <= 1
        order by buy_date, pick_rank
        """
    ).fetchdf()
    sig["symbol"] = np.where(
        sig["stock_code"].str.endswith(".SH"),
        "SHSE." + sig["stock_code"].str.split(".").str[0],
        "SZSE." + sig["stock_code"].str.split(".").str[0],
    )
    sig["target_pct"] = case["target"]
    sig["holding_days"] = 2
    sig["max_holding_days"] = case["max_hold"]
    sig["score_exit_entry_ratio"] = case["exit_ratio"]
    sig["min_holding_days_before_score_exit"] = case["min_exit_hold"]
    sig["score_continue_entry_ratio"] = case["continue_ratio"]
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
        # Control around current 0939_t50.
        dict(name="0939_gap_m3_15_t50_h2", gap_min=-3.0, gap_max=1.5, target=0.50, max_hold=2, exit_ratio=9.99, min_exit_hold=2, continue_ratio=9.99, signal_pct_max=None),
        dict(name="0939_gap_m25_12_t50_h2", gap_min=-2.5, gap_max=1.2, target=0.50, max_hold=2, exit_ratio=9.99, min_exit_hold=2, continue_ratio=9.99, signal_pct_max=None),
        dict(name="0939_gap_m2_10_t50_h2", gap_min=-2.0, gap_max=1.0, target=0.50, max_hold=2, exit_ratio=9.99, min_exit_hold=2, continue_ratio=9.99, signal_pct_max=None),
        dict(name="0939_gap_m3_05_t50_h2", gap_min=-3.0, gap_max=0.5, target=0.50, max_hold=2, exit_ratio=9.99, min_exit_hold=2, continue_ratio=9.99, signal_pct_max=None),
        dict(name="0939_gap_m2_05_t50_h2", gap_min=-2.0, gap_max=0.5, target=0.50, max_hold=2, exit_ratio=9.99, min_exit_hold=2, continue_ratio=9.99, signal_pct_max=None),
        dict(name="0939_pull175_gap_m3_15_t50_h2", gap_min=-3.0, gap_max=1.5, target=0.50, max_hold=2, exit_ratio=9.99, min_exit_hold=2, continue_ratio=9.99, signal_pct_max=-1.75),
        # Let strong score continue one more day; this tests sell frequency.
        dict(name="0939_gap_m3_15_t50_h2max3_cont095", gap_min=-3.0, gap_max=1.5, target=0.50, max_hold=3, exit_ratio=9.99, min_exit_hold=2, continue_ratio=0.95, signal_pct_max=None),
        dict(name="0939_gap_m3_15_t50_h2max3_cont100", gap_min=-3.0, gap_max=1.5, target=0.50, max_hold=3, exit_ratio=9.99, min_exit_hold=2, continue_ratio=1.00, signal_pct_max=None),
        # Slightly lower target, around the drawdown boundary.
        dict(name="0939_gap_m3_15_t47_h2", gap_min=-3.0, gap_max=1.5, target=0.47, max_hold=2, exit_ratio=9.99, min_exit_hold=2, continue_ratio=9.99, signal_pct_max=None),
        dict(name="0939_gap_m25_12_t47_h2", gap_min=-2.5, gap_max=1.2, target=0.47, max_hold=2, exit_ratio=9.99, min_exit_hold=2, continue_ratio=9.99, signal_pct_max=None),
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
    print(manifest[["name", "target", "gap_min", "gap_max", "max_hold", "continue_ratio", "signal_rows", "signal_buy_days", "signal_file"]].to_string(index=False))
    print(MANIFEST)


if __name__ == "__main__":
    main()
