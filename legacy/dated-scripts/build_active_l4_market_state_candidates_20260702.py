from __future__ import annotations

import itertools
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(r"D:\work\quant\quant_mcp")
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_latest_l4_weight_candidates_20260702"
OUT_DIR = REPORT_DIR / "active_l4_market_state_candidates"
MARKET_DB_COPY = REPORT_DIR / "active_l4_qfq_event_search" / "runtime_l2_stock_daily_data_copy.duckdb"
L4 = {
    "1d": (
        ROOT / "quant/data_file/production_assets/duckdb/l4_executable_1d_open_return_formal.duckdb",
        "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_1d_open_return_formal_candidate",
    ),
    "3d": (
        ROOT / "quant/data_file/production_assets/duckdb/l4_executable_3d_open_return_formal.duckdb",
        "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_3d_open_return_formal_candidate",
    ),
    "5d": (
        ROOT / "quant/data_file/production_assets/duckdb/l4_executable_5d_open_return_formal.duckdb",
        "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_5d_open_return_formal_candidate",
    ),
    "10d": (
        ROOT / "quant/data_file/production_assets/duckdb/l4_executable_10d_open_return_formal.duckdb",
        "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_10d_open_return_formal_candidate",
    ),
}


def _symbol(stock_code: str) -> str:
    if stock_code.endswith(".SH"):
        return "SHSE." + stock_code[:6]
    if stock_code.endswith(".SZ"):
        return "SZSE." + stock_code[:6]
    return stock_code


def _annualized(mean_daily: float) -> float:
    return (1.0 + mean_daily) ** 252 - 1.0


def _sharpe(ret: pd.Series) -> float | None:
    std = float(ret.std(ddof=1))
    if not std:
        return None
    return float(ret.mean()) / std * (252**0.5)


def _mdd(ret: pd.Series) -> float:
    nav = (1.0 + ret.fillna(0.0)).cumprod()
    return float(-(nav / nav.cummax() - 1.0).min())


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        for label, (db, _table) in L4.items():
            con.execute(f"ATTACH '{db.as_posix()}' AS l4_{label} (READ_ONLY)")
        con.execute(f"ATTACH '{MARKET_DB_COPY.as_posix()}' AS marketdb (READ_ONLY)")
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE cal AS
            SELECT
                trade_date,
                lead(trade_date) OVER (ORDER BY trade_date) AS buy_date,
                lead(trade_date, 2) OVER (ORDER BY trade_date) AS sell_ref_date
            FROM (
                SELECT DISTINCT trade_date
                FROM marketdb.STOCK_DAILY_DATA
                WHERE trade_date BETWEEN '20220606' AND '20260702'
            )
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE breadth AS
            SELECT
                trade_date,
                avg(CASE WHEN pct_chg > 0 THEN 1.0 ELSE 0.0 END) AS up_ratio,
                avg(CASE WHEN pct_chg <= -5 THEN 1.0 ELSE 0.0 END) AS down5_ratio,
                avg(CASE WHEN pct_chg >= 5 THEN 1.0 ELSE 0.0 END) AS up5_ratio,
                median(pct_chg) AS median_pct,
                avg(pct_chg) AS avg_pct,
                sum(amount) AS total_amount
            FROM marketdb.STOCK_DAILY_DATA
            WHERE trade_date BETWEEN '20220606' AND '20260702'
              AND stock_code NOT LIKE '%.BJ'
              AND coalesce(ST_TYPE, '') IN ('', '0')
              AND coalesce(ST_TYPE_name, '') NOT LIKE '%ST%'
              AND coalesce(name, '') NOT LIKE 'ST%'
              AND coalesce(name, '') NOT LIKE '*ST%'
              AND amount IS NOT NULL
              AND pct_chg IS NOT NULL
            GROUP BY trade_date
            """
        )
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE base AS
            WITH preds AS (
                SELECT
                    p10.trade_date,
                    p10.stock_code,
                    p1.pred_prob AS pred_1d,
                    p3.pred_prob AS pred_3d,
                    p5.pred_prob AS pred_5d,
                    p10.pred_prob AS pred_10d
                FROM l4_10d."{L4['10d'][1]}" p10
                JOIN l4_5d."{L4['5d'][1]}" p5 USING (trade_date, stock_code)
                JOIN l4_3d."{L4['3d'][1]}" p3 USING (trade_date, stock_code)
                JOIN l4_1d."{L4['1d'][1]}" p1 USING (trade_date, stock_code)
                WHERE p10.trade_date BETWEEN '20220606' AND '20260630'
                  AND p10.stock_code NOT LIKE '%.BJ'
            ),
            ranked AS (
                SELECT
                    *,
                    percent_rank() OVER (PARTITION BY trade_date ORDER BY pred_1d) AS r1,
                    percent_rank() OVER (PARTITION BY trade_date ORDER BY pred_3d) AS r3,
                    percent_rank() OVER (PARTITION BY trade_date ORDER BY pred_5d) AS r5,
                    percent_rank() OVER (PARTITION BY trade_date ORDER BY pred_10d) AS r10
                FROM preds
            )
            SELECT
                r.*,
                cal.buy_date,
                cal.sell_ref_date,
                md.name,
                md.market,
                md.industry,
                md.amount,
                md.turnover_rate,
                md.total_mv,
                md.atr_qfq,
                md.pct_chg,
                md.close_qfq,
                buy.open_qfq AS buy_open_qfq,
                sellref.open_qfq AS sell_open_qfq,
                buy.open_qfq / NULLIF(md.close_qfq, 0) - 1 AS buy_open_gap_qfq,
                sellref.open_qfq / NULLIF(buy.open_qfq, 0) - 1 AS ret_open_to_next_open_qfq,
                md.atr_qfq / NULLIF(md.close_qfq, 0) AS atr_pct_qfq,
                b.up_ratio,
                b.down5_ratio,
                b.up5_ratio,
                b.median_pct,
                b.avg_pct,
                b.total_amount,
                percent_rank() OVER (PARTITION BY r.trade_date ORDER BY md.amount) AS amount_rank,
                percent_rank() OVER (PARTITION BY r.trade_date ORDER BY md.turnover_rate) AS turnover_rank,
                percent_rank() OVER (PARTITION BY r.trade_date ORDER BY md.total_mv) AS mv_rank,
                percent_rank() OVER (PARTITION BY r.trade_date ORDER BY md.atr_qfq / NULLIF(md.close_qfq, 0)) AS atr_rank
            FROM ranked r
            JOIN cal ON cal.trade_date = r.trade_date
            JOIN marketdb.STOCK_DAILY_DATA md ON md.trade_date = r.trade_date AND md.stock_code = r.stock_code
            JOIN marketdb.STOCK_DAILY_DATA buy ON buy.trade_date = cal.buy_date AND buy.stock_code = r.stock_code
            JOIN marketdb.STOCK_DAILY_DATA sellref ON sellref.trade_date = cal.sell_ref_date AND sellref.stock_code = r.stock_code
            JOIN breadth b ON b.trade_date = r.trade_date
            WHERE cal.buy_date IS NOT NULL
              AND cal.sell_ref_date IS NOT NULL
              AND md.amount >= 90000
              AND md.total_mv >= 200000
              AND md.atr_qfq IS NOT NULL
              AND md.close_qfq IS NOT NULL
              AND buy.open_qfq IS NOT NULL
              AND sellref.open_qfq IS NOT NULL
              AND coalesce(md.ST_TYPE, '') IN ('', '0')
              AND coalesce(md.ST_TYPE_name, '') NOT LIKE '%ST%'
              AND coalesce(md.name, '') NOT LIKE 'ST%'
              AND coalesce(md.name, '') NOT LIKE '*ST%'
            """
        )
        base = con.execute("SELECT * FROM base").fetchdf()
    finally:
        con.close()

    base["score_10d"] = base["r10"]
    base["score_5d10d"] = 0.5 * base["r5"] + 0.5 * base["r10"]
    base["score_mix"] = 0.10 * base["r1"] + 0.15 * base["r3"] + 0.25 * base["r5"] + 0.50 * base["r10"]
    base["score_liq_mix"] = base["score_mix"] + 0.04 * base["amount_rank"] + 0.02 * base["turnover_rank"] - 0.03 * base["atr_rank"]
    base.to_parquet(OUT_DIR / "market_state_base.parquet", index=False)

    rows = []
    score_cols = ["score_10d", "score_5d10d", "score_mix", "score_liq_mix"]
    for score_col, up_min, down5_max, med_min, gap_low, gap_high, pct_min, pct_max, atr_max, topn, target in itertools.product(
        score_cols,
        [0.35, 0.45, 0.55],
        [0.08, 0.15, 0.25],
        [-1.0, 0.0, 0.5],
        [-0.06, -0.02, 0.0],
        [0.005, 0.015, 0.035],
        [-8.0, -4.0, -1.0],
        [3.0, 6.0, 10.0],
        [0.70, 0.85, 1.0],
        [1, 2, 3],
        [0.25, 0.35, 0.50],
    ):
        if target * topn > 1.05:
            continue
        x = base[
            (base["up_ratio"] >= up_min)
            & (base["down5_ratio"] <= down5_max)
            & (base["median_pct"] >= med_min)
            & (base["buy_open_gap_qfq"] >= gap_low)
            & (base["buy_open_gap_qfq"] <= gap_high)
            & (base["pct_chg"] >= pct_min)
            & (base["pct_chg"] <= pct_max)
            & (base["atr_rank"] <= atr_max)
        ].copy()
        if x.empty:
            continue
        x = x.sort_values(["trade_date", score_col, "stock_code"], ascending=[True, False, True])
        x = x.groupby("trade_date", group_keys=False).head(topn)
        daily = x.groupby("trade_date").agg(
            daily_ret=("ret_open_to_next_open_qfq", lambda s: float(s.mean()) * target),
            names=("stock_code", "count"),
        )
        if len(daily) < 80:
            continue
        if float(daily["names"].mean()) < min(topn, 1.2):
            continue
        ret = daily["daily_ret"].astype(float)
        rows.append(
            {
                "name": f"ms_{score_col}_up{int(up_min*100)}_d5{int(down5_max*100)}_med{str(med_min).replace('.','p').replace('-','m')}_g{int(gap_low*1000)}to{int(gap_high*1000)}_p{str(pct_min).replace('.','p').replace('-','m')}to{str(pct_max).replace('.','p')}_atr{int(atr_max*100)}_top{topn}_pos{int(target*100)}",
                "score_col": score_col,
                "up_min": up_min,
                "down5_max": down5_max,
                "med_min": med_min,
                "gap_low": gap_low,
                "gap_high": gap_high,
                "pct_min": pct_min,
                "pct_max": pct_max,
                "atr_max": atr_max,
                "topn": topn,
                "target_pct": target,
                "days": int(len(daily)),
                "avg_names": float(daily["names"].mean()),
                "local_annual": _annualized(float(ret.mean())),
                "local_sharpe": _sharpe(ret),
                "local_mdd": _mdd(ret),
                "recent60": _annualized(float(ret.tail(60).mean())) if len(ret) >= 60 else None,
                "recent120": _annualized(float(ret.tail(120).mean())) if len(ret) >= 120 else None,
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise SystemExit("no candidates")
    summary = summary.sort_values(["local_sharpe", "local_annual"], ascending=[False, False])
    summary.to_csv(OUT_DIR / "local_market_state_summary.csv", index=False, encoding="utf-8-sig")
    selected = summary[
        (summary["local_annual"] >= 1.0)
        & (summary["local_sharpe"] >= 1.5)
        & (summary["local_mdd"] <= 0.45)
        & (summary["recent60"] > 0)
    ].head(8)
    if selected.empty:
        selected = summary.head(8)
    selected.to_csv(OUT_DIR / "selected_for_juejin.csv", index=False, encoding="utf-8-sig")

    manifest = []
    for row in selected.to_dict("records"):
        x = base[
            (base["up_ratio"] >= row["up_min"])
            & (base["down5_ratio"] <= row["down5_max"])
            & (base["median_pct"] >= row["med_min"])
            & (base["buy_open_gap_qfq"] >= row["gap_low"])
            & (base["buy_open_gap_qfq"] <= row["gap_high"])
            & (base["pct_chg"] >= row["pct_min"])
            & (base["pct_chg"] <= row["pct_max"])
            & (base["atr_rank"] <= row["atr_max"])
        ].copy()
        x = x.sort_values(["trade_date", row["score_col"], "stock_code"], ascending=[True, False, True])
        x = x.groupby("trade_date", group_keys=False).head(int(row["topn"]))
        x["rank"] = x.groupby("trade_date").cumcount() + 1
        out_dir = OUT_DIR / row["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        score_db = out_dir / "score.duckdb"
        if score_db.exists():
            score_db.unlink()
        scon = duckdb.connect(str(score_db))
        try:
            score_df = base[["trade_date", "stock_code", row["score_col"]]].rename(columns={row["score_col"]: "pred_prob"})
            scon.register("score_df", score_df)
            scon.execute("CREATE TABLE score AS SELECT trade_date, stock_code, pred_prob FROM score_df")
            scon.execute("CHECKPOINT")
        finally:
            scon.close()
        x.insert(2, "symbol", x["stock_code"].map(_symbol))
        x["pred_prob"] = x[row["score_col"]]
        x["target_pct"] = f"{float(row['target_pct']):.5f}"
        x["holding_days"] = 1
        x["max_holding_days"] = 1
        x["score_exit_entry_ratio"] = "9.99000"
        x["min_holding_days_before_score_exit"] = 1
        x["score_continue_entry_ratio"] = "9.99000"
        x["strategy_variant"] = row["name"]
        x["filter_name"] = "market_state_qfq"
        x["entry_weight_name"] = row["score_col"]
        x["dynamic_hold_name"] = "h1m1_market_state"
        x["buy_day_market_available"] = True
        x["buy_day_hard_gate_complete"] = True
        x["buy_day_st_rejected"] = False
        x["buy_day_open_limit_up_rejected"] = False
        x["latest_market_date"] = "20260701"
        x["buy_open_gap_pct"] = x["buy_open_gap_qfq"] * 100.0
        out_cols = [
            "trade_date",
            "buy_date",
            "symbol",
            "stock_code",
            "name",
            "rank",
            "pred_prob",
            "pred_1d",
            "pred_3d",
            "pred_5d",
            "pred_10d",
            "amount",
            "turnover_rate",
            "total_mv",
            "atr_qfq",
            "pct_chg",
            "buy_open_gap_pct",
            "target_pct",
            "holding_days",
            "max_holding_days",
            "score_exit_entry_ratio",
            "min_holding_days_before_score_exit",
            "score_continue_entry_ratio",
            "strategy_variant",
            "filter_name",
            "entry_weight_name",
            "dynamic_hold_name",
            "buy_day_market_available",
            "buy_day_hard_gate_complete",
            "buy_day_st_rejected",
            "buy_day_open_limit_up_rejected",
            "latest_market_date",
        ]
        signal = x[out_cols].rename(columns={"trade_date": "signal_date"})
        signal_file = out_dir / "signals.csv"
        signal.to_csv(signal_file, index=False, encoding="utf-8")
        counts = signal.groupby("signal_date").size()
        manifest.append(
            {
                "name": row["name"],
                "signal_file": str(signal_file),
                "score_db": str(score_db),
                "score_table": "score",
                "rows": int(len(signal)),
                "signal_days": int(counts.size),
                "avg_names": float(counts.mean()),
                "days_below_topn": int((counts < int(row["topn"])).sum()),
                **row,
            }
        )
    pd.DataFrame(manifest).to_csv(OUT_DIR / "juejin_candidate_manifest.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(manifest).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
