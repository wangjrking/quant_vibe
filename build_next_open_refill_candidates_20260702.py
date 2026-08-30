from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(r"D:\work\quant\quant_mcp")
BASE_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_latest_l4_weight_candidates_20260702"
SOURCE_SIGNAL_FILE = BASE_DIR / "w25_25_00_50_gapm8p3_top8" / "signals.csv"
OUT_DIR = BASE_DIR / "postrank_open_filter_candidates" / "next_open_refill_candidates"
MARKET_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"


CASES = [
    {
        "name": "norf_top3_p44_sig230_gap025",
        "topn": 3,
        "target_pct": "0.44000",
        "primary": {"pct_chg_max": -2.30, "gap_high": 0.0025, "gap_low": -0.10, "p10d_min": 0.70},
        "refill": {"pct_chg_max": -1.75, "gap_high": 0.0150, "gap_low": -0.10, "p10d_min": 0.70},
        "rank_expr": "entry_score + 0.035 * amount_rank + 0.020 * turnover_rank - 0.020 * gap_rank",
        "exit_ratio": "0.96000",
    },
    {
        "name": "norf_top3_p455_sig230_gap000",
        "topn": 3,
        "target_pct": "0.45500",
        "primary": {"pct_chg_max": -2.30, "gap_high": 0.0000, "gap_low": -0.10, "p10d_min": 0.70},
        "refill": {"pct_chg_max": -1.75, "gap_high": 0.0100, "gap_low": -0.10, "p10d_min": 0.70},
        "rank_expr": "entry_score + 0.030 * amount_rank + 0.020 * turnover_rank - 0.025 * gap_rank",
        "exit_ratio": "0.96000",
    },
    {
        "name": "norf_top3_p47_sig300_gap050",
        "topn": 3,
        "target_pct": "0.47000",
        "primary": {"pct_chg_max": -3.00, "gap_high": 0.0050, "gap_low": -0.10, "p10d_min": 0.70},
        "refill": {"pct_chg_max": -2.00, "gap_high": 0.0150, "gap_low": -0.10, "p10d_min": 0.70},
        "rank_expr": "entry_score + 0.025 * amount_rank + 0.025 * turnover_rank - 0.025 * gap_rank",
        "exit_ratio": "0.95500",
    },
    {
        "name": "norf_top3_p435_sig230_gapneg_liq",
        "topn": 3,
        "target_pct": "0.43500",
        "primary": {"pct_chg_max": -2.30, "gap_high": 0.0000, "gap_low": -0.10, "p10d_min": 0.70, "amount_rank_min": 0.20},
        "refill": {"pct_chg_max": -1.75, "gap_high": 0.0150, "gap_low": -0.10, "p10d_min": 0.70, "amount_rank_min": 0.10},
        "rank_expr": "entry_score + 0.045 * amount_rank + 0.025 * turnover_rank - 0.020 * gap_rank",
        "exit_ratio": "0.96000",
    },
    {
        "name": "norf_top4_p34_sig230_gap025",
        "topn": 4,
        "target_pct": "0.34000",
        "primary": {"pct_chg_max": -2.30, "gap_high": 0.0025, "gap_low": -0.10, "p10d_min": 0.70},
        "refill": {"pct_chg_max": -1.75, "gap_high": 0.0150, "gap_low": -0.10, "p10d_min": 0.70},
        "rank_expr": "entry_score + 0.035 * amount_rank + 0.020 * turnover_rank - 0.020 * gap_rank",
        "exit_ratio": "0.96000",
    },
]


def _mask(df: pd.DataFrame, cfg: dict) -> pd.Series:
    mask = (
        (df["signal_pct_chg"] <= float(cfg["pct_chg_max"]))
        & (df["buy_open_gap_qfq"] >= float(cfg["gap_low"]))
        & (df["buy_open_gap_qfq"] <= float(cfg["gap_high"]))
        & (df["pred_10d"] >= float(cfg["p10d_min"]))
    )
    if "amount_rank_min" in cfg:
        mask &= df["amount_rank"] >= float(cfg["amount_rank_min"])
    return mask


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sig = pd.read_csv(SOURCE_SIGNAL_FILE, dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    con = duckdb.connect(str(MARKET_DB), read_only=True)
    try:
        con.register("sig", sig)
        enriched = con.execute(
            """
            WITH md AS (
                SELECT trade_date, stock_code, open_qfq, close_qfq, pct_chg
                FROM STOCK_DAILY_DATA
                WHERE trade_date BETWEEN '20220606' AND '20260701'
            )
            SELECT
                sig.*,
                sigmd.pct_chg AS signal_pct_chg,
                buy.open_qfq / NULLIF(sigmd.close_qfq, 0) - 1 AS buy_open_gap_qfq,
                percent_rank() OVER (PARTITION BY sig.signal_date ORDER BY sig.amount) AS amount_rank,
                percent_rank() OVER (PARTITION BY sig.signal_date ORDER BY sig.turnover_rate) AS turnover_rank,
                percent_rank() OVER (
                    PARTITION BY sig.signal_date
                    ORDER BY buy.open_qfq / NULLIF(sigmd.close_qfq, 0) - 1
                ) AS gap_rank
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
        combined["score_exit_entry_ratio"] = case["exit_ratio"]
        combined["min_holding_days_before_score_exit"] = 1
        combined["score_continue_entry_ratio"] = "9.99000"
        combined["strategy_variant"] = case["name"]
        combined["filter_name"] = case["name"]
        combined["dynamic_hold_name"] = "h1m1_next_open_refill"
        out_dir = OUT_DIR / case["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        signal_file = out_dir / "signals.csv"
        drop_cols = ["signal_pct_chg", "buy_open_gap_qfq", "amount_rank", "turnover_rank", "gap_rank", "rerank_score", "tier"]
        combined.drop(columns=[c for c in drop_cols if c in combined.columns]).to_csv(signal_file, index=False, encoding="utf-8")
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
