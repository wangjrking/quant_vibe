from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(r"D:\work\quant\quant_mcp")
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_latest_l4_weight_candidates_20260702"
OUT_DIR = REPORT_DIR / "active_l4_clean_direct_rank_candidates"
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


CASES = [
    {"name": "clean_w000_000_000_100_top3_p25", "weights": {"1d": 0, "3d": 0, "5d": 0, "10d": 1.0}, "topn": 3, "target": 0.25},
    {"name": "clean_w000_000_000_100_top5_p18", "weights": {"1d": 0, "3d": 0, "5d": 0, "10d": 1.0}, "topn": 5, "target": 0.18},
    {"name": "clean_w000_000_050_050_top3_p25", "weights": {"1d": 0, "3d": 0, "5d": 0.5, "10d": 0.5}, "topn": 3, "target": 0.25},
    {"name": "clean_w010_015_025_050_top3_p25", "weights": {"1d": 0.10, "3d": 0.15, "5d": 0.25, "10d": 0.5}, "topn": 3, "target": 0.25},
    {"name": "clean_w025_025_000_050_top3_p25", "weights": {"1d": 0.25, "3d": 0.25, "5d": 0, "10d": 0.5}, "topn": 3, "target": 0.25},
    {"name": "clean_w050_000_000_050_top3_p25", "weights": {"1d": 0.5, "3d": 0, "5d": 0, "10d": 0.5}, "topn": 3, "target": 0.25},
    {"name": "clean_w000_000_100_000_top3_p25", "weights": {"1d": 0, "3d": 0, "5d": 1.0, "10d": 0}, "topn": 3, "target": 0.25},
    {"name": "clean_w100_000_000_000_top3_p25", "weights": {"1d": 1.0, "3d": 0, "5d": 0, "10d": 0}, "topn": 3, "target": 0.25},
    {"name": "clean_rev_w000_000_000_100_top3_p25", "weights": {"1d": 0, "3d": 0, "5d": 0, "10d": 1.0}, "topn": 3, "target": 0.25, "reverse": True},
    {"name": "clean_rev_w000_000_050_050_top3_p25", "weights": {"1d": 0, "3d": 0, "5d": 0.5, "10d": 0.5}, "topn": 3, "target": 0.25, "reverse": True},
    {"name": "clean_rev_w010_015_025_050_top3_p25", "weights": {"1d": 0.10, "3d": 0.15, "5d": 0.25, "10d": 0.5}, "topn": 3, "target": 0.25, "reverse": True},
]


def _symbol(stock_code: str) -> str:
    if stock_code.endswith(".SH"):
        return "SHSE." + stock_code[:6]
    if stock_code.endswith(".SZ"):
        return "SZSE." + stock_code[:6]
    return stock_code


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        for label, (db, table) in L4.items():
            con.execute(f"ATTACH '{db.as_posix()}' AS l4_{label} (READ_ONLY)")
        con.execute(f"ATTACH '{MARKET_DB_COPY.as_posix()}' AS marketdb (READ_ONLY)")
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE cal AS
            SELECT
                trade_date,
                lead(trade_date) OVER (ORDER BY trade_date) AS buy_date
            FROM (
                SELECT DISTINCT trade_date
                FROM marketdb.STOCK_DAILY_DATA
                WHERE trade_date BETWEEN '20220606' AND '20260702'
            )
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
                buy.open_qfq / NULLIF(md.close_qfq, 0) - 1 AS buy_open_gap_qfq
            FROM ranked r
            JOIN cal ON cal.trade_date = r.trade_date
            JOIN marketdb.STOCK_DAILY_DATA md ON md.trade_date = r.trade_date AND md.stock_code = r.stock_code
            JOIN marketdb.STOCK_DAILY_DATA buy ON buy.trade_date = cal.buy_date AND buy.stock_code = r.stock_code
            WHERE cal.buy_date IS NOT NULL
              AND md.amount >= 90000
              AND md.total_mv >= 200000
              AND md.atr_qfq IS NOT NULL
              AND md.close_qfq IS NOT NULL
              AND buy.open_qfq IS NOT NULL
              AND coalesce(md.ST_TYPE, '') IN ('', '0')
              AND coalesce(md.ST_TYPE_name, '') NOT LIKE '%ST%'
              AND coalesce(md.name, '') NOT LIKE 'ST%'
              AND coalesce(md.name, '') NOT LIKE '*ST%'
            """
        )
        manifest = []
        for case in CASES:
            w = case["weights"]
            score_expr = f"({w['1d']} * r1 + {w['3d']} * r3 + {w['5d']} * r5 + {w['10d']} * r10)"
            out_dir = OUT_DIR / case["name"]
            out_dir.mkdir(parents=True, exist_ok=True)
            score_db = out_dir / "score.duckdb"
            if score_db.exists():
                score_db.unlink()
            con.execute(f"ATTACH '{score_db.as_posix()}' AS scoreout")
            con.execute(
                f"""
                CREATE TABLE scoreout.score AS
                SELECT trade_date, stock_code, {score_expr}::DOUBLE AS pred_prob
                FROM base
                """
            )
            con.execute("DETACH scoreout")
            df = con.execute(
                f"""
                WITH scored AS (
                    SELECT *, {score_expr}::DOUBLE AS entry_score
                    FROM base
                ),
                selected AS (
                    SELECT
                        *,
                        row_number() OVER (
                            PARTITION BY trade_date
                            ORDER BY entry_score {"ASC" if case.get("reverse") else "DESC"}, stock_code
                        ) AS rn
                    FROM scored
                )
                SELECT
                    trade_date AS signal_date,
                    buy_date,
                    stock_code,
                    name,
                    rn AS rank,
                    entry_score AS pred_prob,
                    entry_score,
                    pred_1d,
                    pred_3d,
                    pred_5d,
                    pred_10d,
                    amount,
                    turnover_rate,
                    total_mv,
                    atr_qfq,
                    pct_chg,
                    buy_open_gap_qfq
                FROM selected
                WHERE rn <= {int(case['topn'])}
                ORDER BY signal_date, rn
                """
            ).fetchdf()
            df.insert(2, "symbol", df["stock_code"].map(_symbol))
            df["target_pct"] = f"{float(case['target']):.5f}"
            df["holding_days"] = 1
            df["max_holding_days"] = 1
            df["score_exit_entry_ratio"] = "9.99000"
            df["min_holding_days_before_score_exit"] = 1
            df["score_continue_entry_ratio"] = "9.99000"
            df["strategy_variant"] = case["name"]
            df["filter_name"] = "clean_active_l4_direct_rank"
            df["entry_weight_name"] = case["name"]
            df["dynamic_hold_name"] = "h1m1_clean_direct"
            df["buy_day_market_available"] = True
            df["buy_day_hard_gate_complete"] = True
            df["buy_day_st_rejected"] = False
            df["buy_day_open_limit_up_rejected"] = False
            df["latest_market_date"] = "20260701"
            df["buy_open_gap_pct"] = df["buy_open_gap_qfq"] * 100.0
            df = df.drop(columns=["buy_open_gap_qfq"])
            signal_file = out_dir / "signals.csv"
            df.to_csv(signal_file, index=False, encoding="utf-8")
            counts = df.groupby("signal_date").size()
            manifest.append(
                {
                    "name": case["name"],
                    "signal_file": str(signal_file),
                    "score_db": str(score_db),
                    "score_table": "score",
                    "rows": int(len(df)),
                    "signal_days": int(counts.size),
                    "avg_names": float(counts.mean()),
                    "days_below_topn": int((counts < int(case["topn"])).sum()),
                    **case,
                }
            )
        pd.DataFrame(manifest).to_csv(OUT_DIR / "candidate_manifest.csv", index=False, encoding="utf-8-sig")
        print(pd.DataFrame(manifest).to_string(index=False))
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
