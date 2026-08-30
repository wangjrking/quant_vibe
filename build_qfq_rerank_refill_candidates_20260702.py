from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(r"D:\work\quant\quant_mcp")
BASE_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_latest_l4_weight_candidates_20260702"
OUT_DIR = BASE_DIR / "postrank_open_filter_candidates" / "qfq_rerank_refill_candidates"
SOURCE_SIGNAL_FILE = BASE_DIR / "w25_25_00_50_gapm8p3_top8" / "signals.csv"
MARKET_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"


CASES = [
    {
        "name": "atr10_refill_top3_p34",
        "topn": 3,
        "target_pct": "0.34000",
        "primary": {"gap_low": -0.05, "gap_high": 0.00, "pct_abs_cap": 12.0, "atr_cap": 0.10, "amount_min": 90_000.0, "mv_min": 200_000.0},
        "refill": {"gap_low": -0.08, "gap_high": 0.03, "pct_abs_cap": 12.0, "atr_cap": 0.16, "amount_min": 90_000.0, "mv_min": 200_000.0},
        "rank_expr": "entry_score + 0.015 * amount_rank - 0.025 * atr_rank",
    },
    {
        "name": "liq_refill_top3_p40",
        "topn": 3,
        "target_pct": "0.40000",
        "primary": {"gap_low": -0.05, "gap_high": 0.00, "pct_abs_cap": 12.0, "atr_cap": 0.12, "amount_min": 90_000.0, "mv_min": 200_000.0},
        "refill": {"gap_low": -0.08, "gap_high": 0.03, "pct_abs_cap": 12.0, "atr_cap": 0.18, "amount_min": 90_000.0, "mv_min": 200_000.0},
        "rank_expr": "entry_score + 0.035 * amount_rank + 0.020 * mv_rank - 0.040 * atr_rank",
    },
    {
        "name": "liq_refill_top4_p30",
        "topn": 4,
        "target_pct": "0.30000",
        "primary": {"gap_low": -0.05, "gap_high": 0.00, "pct_abs_cap": 12.0, "atr_cap": 0.12, "amount_min": 90_000.0, "mv_min": 200_000.0},
        "refill": {"gap_low": -0.08, "gap_high": 0.03, "pct_abs_cap": 12.0, "atr_cap": 0.18, "amount_min": 90_000.0, "mv_min": 200_000.0},
        "rank_expr": "entry_score + 0.035 * amount_rank + 0.020 * mv_rank - 0.040 * atr_rank",
    },
    {
        "name": "strict_refill_top5_p22",
        "topn": 5,
        "target_pct": "0.22000",
        "primary": {"gap_low": -0.04, "gap_high": 0.005, "pct_abs_cap": 10.0, "atr_cap": 0.10, "amount_min": 150_000.0, "mv_min": 300_000.0},
        "refill": {"gap_low": -0.06, "gap_high": 0.02, "pct_abs_cap": 12.0, "atr_cap": 0.14, "amount_min": 90_000.0, "mv_min": 200_000.0},
        "rank_expr": "entry_score + 0.030 * amount_rank + 0.020 * mv_rank - 0.035 * atr_rank",
    },
]


def _mask(df: pd.DataFrame, cfg: dict) -> pd.Series:
    return (
        (df["buy_open_gap_qfq"] >= float(cfg["gap_low"]))
        & (df["buy_open_gap_qfq"] <= float(cfg["gap_high"]))
        & (df["signal_pct_chg"].abs() <= float(cfg["pct_abs_cap"]))
        & (df["atr_pct_qfq"] <= float(cfg["atr_cap"]))
        & (df["amount"] >= float(cfg["amount_min"]))
        & (df["total_mv"] >= float(cfg["mv_min"]))
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sig = pd.read_csv(SOURCE_SIGNAL_FILE, dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    con = duckdb.connect(str(MARKET_DB), read_only=True)
    try:
        con.register("sig", sig)
        enriched = con.execute(
            """
            WITH md AS (
                SELECT trade_date, stock_code, open_qfq, close_qfq, atr_qfq, pct_chg
                FROM STOCK_DAILY_DATA
                WHERE trade_date BETWEEN '20220606' AND '20260701'
            )
            SELECT
                sig.*,
                sigmd.pct_chg AS signal_pct_chg,
                buy.open_qfq / NULLIF(sigmd.close_qfq, 0) - 1 AS buy_open_gap_qfq,
                sigmd.atr_qfq / NULLIF(sigmd.close_qfq, 0) AS atr_pct_qfq,
                percent_rank() OVER (PARTITION BY sig.signal_date ORDER BY sig.amount) AS amount_rank,
                percent_rank() OVER (PARTITION BY sig.signal_date ORDER BY sig.total_mv) AS mv_rank,
                percent_rank() OVER (
                    PARTITION BY sig.signal_date
                    ORDER BY sigmd.atr_qfq / NULLIF(sigmd.close_qfq, 0)
                ) AS atr_rank
            FROM sig
            JOIN md sigmd ON sigmd.trade_date = sig.signal_date AND sigmd.stock_code = sig.stock_code
            JOIN md buy ON buy.trade_date = sig.buy_date AND buy.stock_code = sig.stock_code
            """
        ).fetchdf()
    finally:
        con.close()

    manifest_rows = []
    for case in CASES:
        x = enriched.copy()
        x["rerank_score"] = x.eval(case["rank_expr"])
        primary = x[_mask(x, case["primary"])].copy()
        primary["tier"] = 0
        refill = x[_mask(x, case["refill"])].copy()
        refill["tier"] = 1
        combined = pd.concat([primary, refill], ignore_index=True)
        combined = combined.sort_values(
            ["signal_date", "tier", "rerank_score", "stock_code"],
            ascending=[True, True, False, True],
        )
        combined = combined.drop_duplicates(["signal_date", "stock_code"], keep="first")
        combined = combined.groupby("signal_date", group_keys=False).head(int(case["topn"]))
        combined["rank"] = combined.groupby("signal_date").cumcount() + 1
        combined["target_pct"] = case["target_pct"]
        combined["holding_days"] = 1
        combined["max_holding_days"] = 1
        combined["score_exit_entry_ratio"] = "0.98000"
        combined["min_holding_days_before_score_exit"] = 1
        combined["score_continue_entry_ratio"] = "9.99000"
        combined["strategy_variant"] = case["name"]
        combined["filter_name"] = case["name"]
        combined["dynamic_hold_name"] = "h1m1_qfq_open_refill"
        combined = combined.drop(
            columns=[
                "signal_pct_chg",
                "buy_open_gap_qfq",
                "atr_pct_qfq",
                "amount_rank",
                "mv_rank",
                "atr_rank",
                "rerank_score",
                "tier",
            ]
        )
        out_dir = OUT_DIR / case["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        signal_file = out_dir / "signals.csv"
        combined.to_csv(signal_file, index=False, encoding="utf-8")
        counts = combined.groupby("signal_date").size()
        manifest_rows.append(
            {
                "name": case["name"],
                "signal_file": str(signal_file),
                "rows": int(len(combined)),
                "signal_days": int(counts.size),
                "avg_names": float(counts.mean()) if not counts.empty else 0.0,
                "days_below_target": int((counts < int(case["topn"])).sum()) if not counts.empty else 0,
                "topn": int(case["topn"]),
                "target_pct": case["target_pct"],
            }
        )
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(OUT_DIR / "candidate_manifest.csv", index=False, encoding="utf-8-sig")
    print(manifest.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
