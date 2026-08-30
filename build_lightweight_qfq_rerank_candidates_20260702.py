from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(r"D:\work\quant\quant_mcp")
BASE_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_latest_l4_weight_candidates_20260702"
OUT_DIR = BASE_DIR / "postrank_open_filter_candidates" / "lightweight_qfq_rerank_candidates"
SOURCE_SIGNAL_FILE = BASE_DIR / "w25_25_00_50_gapm8p3_top8" / "signals.csv"
MARKET_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"


CASES = [
    {
        "name": "lowflat_atr10_top3_p25",
        "gap_low": -0.05,
        "gap_high": 0.00,
        "pct_abs_cap": 12.0,
        "atr_pct_cap": 0.10,
        "amount_min": 90_000.0,
        "mv_min": 200_000.0,
        "topn": 3,
        "target_pct": "0.25000",
        "rank_expr": "entry_score",
    },
    {
        "name": "lowflat_atr8_top3_p24",
        "gap_low": -0.05,
        "gap_high": 0.00,
        "pct_abs_cap": 10.0,
        "atr_pct_cap": 0.08,
        "amount_min": 150_000.0,
        "mv_min": 300_000.0,
        "topn": 3,
        "target_pct": "0.24000",
        "rank_expr": "entry_score",
    },
    {
        "name": "lowflat_liqbonus_top3_p25",
        "gap_low": -0.05,
        "gap_high": 0.00,
        "pct_abs_cap": 12.0,
        "atr_pct_cap": 0.12,
        "amount_min": 90_000.0,
        "mv_min": 200_000.0,
        "topn": 3,
        "target_pct": "0.25000",
        "rank_expr": "entry_score + 0.035 * amount_rank + 0.020 * mv_rank - 0.040 * atr_rank",
    },
    {
        "name": "lowflat_liqbonus_top4_p22",
        "gap_low": -0.05,
        "gap_high": 0.00,
        "pct_abs_cap": 12.0,
        "atr_pct_cap": 0.12,
        "amount_min": 90_000.0,
        "mv_min": 200_000.0,
        "topn": 4,
        "target_pct": "0.22000",
        "rank_expr": "entry_score + 0.035 * amount_rank + 0.020 * mv_rank - 0.040 * atr_rank",
    },
    {
        "name": "flat_mild_top3_p25",
        "gap_low": -0.04,
        "gap_high": 0.005,
        "pct_abs_cap": 10.0,
        "atr_pct_cap": 0.11,
        "amount_min": 120_000.0,
        "mv_min": 250_000.0,
        "topn": 3,
        "target_pct": "0.25000",
        "rank_expr": "entry_score",
    },
    {
        "name": "flat_mild_liq_top4_p20",
        "gap_low": -0.04,
        "gap_high": 0.005,
        "pct_abs_cap": 10.0,
        "atr_pct_cap": 0.11,
        "amount_min": 120_000.0,
        "mv_min": 250_000.0,
        "topn": 4,
        "target_pct": "0.20000",
        "rank_expr": "entry_score + 0.030 * amount_rank + 0.020 * mv_rank - 0.035 * atr_rank",
    },
    {
        "name": "lowflat_no_weakday_top3_p25",
        "gap_low": -0.05,
        "gap_high": 0.00,
        "pct_abs_cap": 9.0,
        "pct_min": -6.0,
        "atr_pct_cap": 0.11,
        "amount_min": 90_000.0,
        "mv_min": 200_000.0,
        "topn": 3,
        "target_pct": "0.25000",
        "rank_expr": "entry_score",
    },
    {
        "name": "lowflat_top5_p15_liq",
        "gap_low": -0.05,
        "gap_high": 0.00,
        "pct_abs_cap": 12.0,
        "atr_pct_cap": 0.12,
        "amount_min": 90_000.0,
        "mv_min": 200_000.0,
        "topn": 5,
        "target_pct": "0.15000",
        "rank_expr": "entry_score + 0.035 * amount_rank + 0.020 * mv_rank - 0.040 * atr_rank",
    },
]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sig = pd.read_csv(SOURCE_SIGNAL_FILE, dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    con = duckdb.connect(str(MARKET_DB), read_only=True)
    try:
        con.register("sig", sig)
        enriched = con.execute(
            """
            WITH md AS (
                SELECT
                    trade_date,
                    stock_code,
                    open_qfq,
                    close_qfq,
                    amount,
                    total_mv,
                    atr_qfq,
                    pct_chg
                FROM STOCK_DAILY_DATA
                WHERE trade_date BETWEEN '20220606' AND '20260701'
            ),
            joined AS (
                SELECT
                    sig.*,
                    sigmd.pct_chg AS signal_pct_chg,
                    buy.open_qfq / NULLIF(sigmd.close_qfq, 0) - 1 AS buy_open_gap_qfq,
                    sigmd.atr_qfq / NULLIF(sigmd.close_qfq, 0) AS atr_pct_qfq,
                    percent_rank() OVER (PARTITION BY sig.signal_date ORDER BY sig.amount) AS amount_rank,
                    percent_rank() OVER (PARTITION BY sig.signal_date ORDER BY sig.total_mv) AS mv_rank,
                    percent_rank() OVER (PARTITION BY sig.signal_date ORDER BY sigmd.atr_qfq / NULLIF(sigmd.close_qfq, 0)) AS atr_rank
                FROM sig
                JOIN md sigmd ON sigmd.trade_date = sig.signal_date AND sigmd.stock_code = sig.stock_code
                JOIN md buy ON buy.trade_date = sig.buy_date AND buy.stock_code = sig.stock_code
            )
            SELECT * FROM joined
            """
        ).fetchdf()
    finally:
        con.close()

    rows = []
    for case in CASES:
        x = enriched[
            (enriched["buy_open_gap_qfq"] >= float(case["gap_low"]))
            & (enriched["buy_open_gap_qfq"] <= float(case["gap_high"]))
            & (enriched["signal_pct_chg"].abs() <= float(case["pct_abs_cap"]))
            & (enriched["atr_pct_qfq"] <= float(case["atr_pct_cap"]))
            & (enriched["amount"] >= float(case["amount_min"]))
            & (enriched["total_mv"] >= float(case["mv_min"]))
        ].copy()
        if "pct_min" in case:
            x = x[x["signal_pct_chg"] >= float(case["pct_min"])].copy()
        if x.empty:
            continue
        x["rerank_score"] = x.eval(case["rank_expr"])
        x = x.sort_values(["signal_date", "rerank_score", "stock_code"], ascending=[True, False, True])
        x = x.groupby("signal_date", group_keys=False).head(int(case["topn"]))
        x["rank"] = x.groupby("signal_date").cumcount() + 1
        x["target_pct"] = case["target_pct"]
        x["holding_days"] = 1
        x["max_holding_days"] = 1
        x["score_exit_entry_ratio"] = "0.98000"
        x["min_holding_days_before_score_exit"] = 1
        x["score_continue_entry_ratio"] = "9.99000"
        x["strategy_variant"] = case["name"]
        x["filter_name"] = case["name"]
        x["dynamic_hold_name"] = "h1m1_qfq_open_rerank"
        x = x.drop(columns=["signal_pct_chg", "buy_open_gap_qfq", "atr_pct_qfq", "amount_rank", "mv_rank", "atr_rank", "rerank_score"])

        out_dir = OUT_DIR / case["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        signal_file = out_dir / "signals.csv"
        x.to_csv(signal_file, index=False, encoding="utf-8")
        counts = x.groupby("signal_date").size()
        rows.append(
            {
                "name": case["name"],
                "signal_file": str(signal_file),
                "rows": int(len(x)),
                "signal_days": int(counts.size),
                "avg_names": float(counts.mean()) if not counts.empty else 0.0,
                "days_below_target": int((counts < int(case["topn"])).sum()) if not counts.empty else 0,
                **case,
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(OUT_DIR / "candidate_manifest.csv", index=False, encoding="utf-8-sig")
    print(manifest.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
