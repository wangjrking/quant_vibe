from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(r"D:\work\quant\quant_mcp")
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_active_l4_prod_repro_grid_20260703"
OUT_DIR = REPORT_DIR / "old500_date_state_features"
MARKET_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
PROFILE_RET = REPORT_DIR / "old500_date_selection_bias" / "profile_top1_with_ret_h1.csv"
OLD_SIGNAL = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_latest_l4_weight_candidates_20260702"
    / "postrank_open_filter_candidates"
    / "qfq_rerank_refill_candidates"
    / "reconstruct_500_from_p44_plus_1dge_m002"
    / "signals.csv"
)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    profile = pd.read_csv(PROFILE_RET, dtype={"signal_date": str})
    old = pd.read_csv(OLD_SIGNAL, dtype={"signal_date": str})
    old_days = set(old["signal_date"].astype(str))
    days = pd.DataFrame({"signal_date": sorted(profile["signal_date"].unique())})
    days["is_old_signal_day"] = days["signal_date"].isin(old_days).astype(int)

    con = duckdb.connect(str(MARKET_DB), read_only=True)
    try:
        con.register("days", days)
        state = con.execute(
            """
            WITH universe AS (
              SELECT
                trade_date,
                stock_code,
                pct_chg,
                amount,
                turnover_rate,
                total_mv,
                close,
                close_qfq,
                open,
                pre_close,
                row_number() OVER (PARTITION BY stock_code ORDER BY trade_date) AS rn
              FROM STOCK_DAILY_DATA
              WHERE stock_code NOT LIKE '%.BJ'
                AND coalesce(ST_TYPE, '') IN ('', '0')
                AND coalesce(ST_TYPE_name, '') NOT LIKE '%ST%'
                AND name NOT LIKE 'ST%'
                AND name NOT LIKE '*ST%'
            ),
            idx AS (
              SELECT
                d.signal_date,
                d.is_old_signal_day,
                u.stock_code,
                u.pct_chg,
                u.amount,
                u.turnover_rate,
                u.total_mv,
                u.open / NULLIF(u.pre_close, 0) - 1 AS open_gap,
                u.close_qfq / NULLIF(lag(u.close_qfq, 5) OVER (PARTITION BY u.stock_code ORDER BY u.trade_date), 0) - 1 AS ret5,
                u.close_qfq / NULLIF(lag(u.close_qfq, 20) OVER (PARTITION BY u.stock_code ORDER BY u.trade_date), 0) - 1 AS ret20
              FROM universe u
              JOIN days d ON d.signal_date = u.trade_date
            )
            SELECT
              signal_date,
              max(is_old_signal_day) AS is_old_signal_day,
              count(*) AS stock_count,
              avg(pct_chg) AS mkt_avg_pct_chg,
              median(pct_chg) AS mkt_med_pct_chg,
              avg(case when pct_chg > 0 then 1 else 0 end) AS up_ratio,
              avg(case when pct_chg <= -5 then 1 else 0 end) AS deep_down_ratio,
              avg(case when pct_chg >= 5 then 1 else 0 end) AS big_up_ratio,
              avg(case when open_gap > 0 then 1 else 0 end) AS high_open_ratio,
              avg(case when open_gap < 0 then 1 else 0 end) AS low_open_ratio,
              avg(turnover_rate) AS avg_turnover,
              median(turnover_rate) AS med_turnover,
              avg(amount) AS avg_amount,
              median(amount) AS med_amount,
              avg(ret5) AS avg_ret5,
              median(ret5) AS med_ret5,
              avg(ret20) AS avg_ret20,
              median(ret20) AS med_ret20
            FROM idx
            GROUP BY signal_date
            ORDER BY signal_date
            """
        ).fetchdf()
    finally:
        con.close()

    day_ret = (
        profile.assign(weighted_ret=profile["ret_h1"].astype(float) * profile["target_pct"].astype(float))
        .groupby("signal_date")["weighted_ret"]
        .sum()
        .reset_index()
    )
    state = state.merge(day_ret, on="signal_date", how="left")
    state.to_csv(OUT_DIR / "profile_day_market_state.csv", index=False, encoding="utf-8-sig")

    feature_cols = [
        "mkt_avg_pct_chg",
        "mkt_med_pct_chg",
        "up_ratio",
        "deep_down_ratio",
        "big_up_ratio",
        "high_open_ratio",
        "low_open_ratio",
        "avg_turnover",
        "med_turnover",
        "avg_amount",
        "med_amount",
        "avg_ret5",
        "med_ret5",
        "avg_ret20",
        "med_ret20",
    ]
    compare = (
        state.groupby("is_old_signal_day")[feature_cols + ["weighted_ret"]]
        .agg(["mean", "median"])
        .T
    )
    compare.to_csv(OUT_DIR / "old_vs_nonold_market_state_compare.csv", encoding="utf-8-sig")

    corr = state[feature_cols + ["is_old_signal_day", "weighted_ret"]].corr(numeric_only=True)
    corr.to_csv(OUT_DIR / "market_state_correlation.csv", encoding="utf-8-sig")

    summary = {
        "days": int(len(state)),
        "old_days": int(state["is_old_signal_day"].sum()),
        "non_old_days": int((state["is_old_signal_day"] == 0).sum()),
        "corr_with_old_day": corr["is_old_signal_day"].drop("is_old_signal_day").sort_values(ascending=False).head(10).to_dict(),
        "corr_with_profile_weighted_ret": corr["weighted_ret"].drop("weighted_ret").sort_values(ascending=False).head(10).to_dict(),
    }
    (OUT_DIR / "date_state_feature_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
