from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(r"D:\work\quant\quant_mcp")
BASE_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_latest_l4_weight_candidates_20260702"
OUT_DIR = BASE_DIR / "postrank_open_filter_candidates" / "gap_sharpe_refine_candidates"
SOURCE_SIGNAL_FILE = BASE_DIR / "w25_25_00_50_gapm8p3_top8" / "signals.csv"
MARKET_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"


CASES = [
    {
        "name": "gsr_p50_sig175_bog0_p10d70",
        "topn": 3,
        "target_pct": "0.50000",
        "signal_pct_max": -1.75,
        "buy_open_gap_raw_max": 0.000,
        "pred_10d_min": 0.70,
    },
    {
        "name": "gsr_p52_sig175_bog0_p10d70",
        "topn": 3,
        "target_pct": "0.52000",
        "signal_pct_max": -1.75,
        "buy_open_gap_raw_max": 0.000,
        "pred_10d_min": 0.70,
    },
    {
        "name": "gsr_p55_sig175_bogm05_p10d70",
        "topn": 3,
        "target_pct": "0.55000",
        "signal_pct_max": -1.75,
        "buy_open_gap_raw_max": -0.005,
        "pred_10d_min": 0.70,
    },
    {
        "name": "gsr_p60_sig175_bogm05_p10d70",
        "topn": 3,
        "target_pct": "0.60000",
        "signal_pct_max": -1.75,
        "buy_open_gap_raw_max": -0.005,
        "pred_10d_min": 0.70,
    },
    {
        "name": "gsr_p60_sig225_bog0_p10d70",
        "topn": 3,
        "target_pct": "0.60000",
        "signal_pct_max": -2.25,
        "buy_open_gap_raw_max": 0.000,
        "pred_10d_min": 0.70,
    },
    {
        "name": "gsr_p65_sig250_bog0_p10d70",
        "topn": 3,
        "target_pct": "0.65000",
        "signal_pct_max": -2.50,
        "buy_open_gap_raw_max": 0.000,
        "pred_10d_min": 0.70,
    },
    {
        "name": "gsr_p58_sig175_bog0_amt150_p10d70",
        "topn": 3,
        "target_pct": "0.58000",
        "signal_pct_max": -1.75,
        "buy_open_gap_raw_max": 0.000,
        "signal_amount_min": 150_000.0,
        "pred_10d_min": 0.70,
    },
    {
        "name": "gsr_p60_sig175_bog0_amt300_p10d70",
        "topn": 3,
        "target_pct": "0.60000",
        "signal_pct_max": -1.75,
        "buy_open_gap_raw_max": 0.000,
        "signal_amount_min": 300_000.0,
        "pred_10d_min": 0.70,
    },
    {
        "name": "gsr_p58_sig175_bog0_p10d75",
        "topn": 3,
        "target_pct": "0.58000",
        "signal_pct_max": -1.75,
        "buy_open_gap_raw_max": 0.000,
        "pred_10d_min": 0.75,
    },
    {
        "name": "gsr_p62_sig175_bog0_p10d80",
        "topn": 3,
        "target_pct": "0.62000",
        "signal_pct_max": -1.75,
        "buy_open_gap_raw_max": 0.000,
        "pred_10d_min": 0.80,
    },
]


def _case_mask(df: pd.DataFrame, case: dict) -> pd.Series:
    mask = (
        (df["signal_pct_chg"] <= float(case["signal_pct_max"]))
        & (df["buy_open_gap_raw"] <= float(case["buy_open_gap_raw_max"]))
        & (df["pred_10d"] >= float(case["pred_10d_min"]))
        & (df["amount"] >= float(case.get("signal_amount_min", 90_000.0)))
        & (df["total_mv"] >= float(case.get("signal_mv_min", 200_000.0)))
        & (df["atr_pct_qfq"] <= float(case.get("atr_pct_qfq_max", 0.18)))
    )
    if "buy_open_gap_raw_min" in case:
        mask &= df["buy_open_gap_raw"] >= float(case["buy_open_gap_raw_min"])
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
                SELECT
                    trade_date,
                    stock_code,
                    open,
                    pre_close,
                    open_qfq,
                    close_qfq,
                    atr_qfq,
                    pct_chg
                FROM STOCK_DAILY_DATA
                WHERE trade_date BETWEEN '20220606' AND '20260701'
            )
            SELECT
                sig.*,
                sigmd.pct_chg AS signal_pct_chg,
                buy.open / NULLIF(buy.pre_close, 0) - 1 AS buy_open_gap_raw,
                buy.open_qfq / NULLIF(sigmd.close_qfq, 0) - 1 AS buy_open_gap_qfq_from_signal_close,
                sigmd.atr_qfq / NULLIF(sigmd.close_qfq, 0) AS atr_pct_qfq
            FROM sig
            JOIN md sigmd ON sigmd.trade_date = sig.signal_date AND sigmd.stock_code = sig.stock_code
            JOIN md buy ON buy.trade_date = sig.buy_date AND buy.stock_code = sig.stock_code
            """
        ).fetchdf()
    finally:
        con.close()

    manifest_rows = []
    for case in CASES:
        x = enriched[_case_mask(enriched, case)].copy()
        x = x.sort_values(
            ["signal_date", "entry_score", "pred_10d", "amount", "stock_code"],
            ascending=[True, False, False, False, True],
        )
        x = x.groupby("signal_date", group_keys=False).head(int(case["topn"]))
        x["rank"] = x.groupby("signal_date").cumcount() + 1
        x["target_pct"] = case["target_pct"]
        x["holding_days"] = 1
        x["max_holding_days"] = 1
        x["score_exit_entry_ratio"] = "0.96000"
        x["min_holding_days_before_score_exit"] = 1
        x["score_continue_entry_ratio"] = "9.99000"
        x["strategy_variant"] = case["name"]
        x["filter_name"] = case["name"]
        x["entry_weight_name"] = "w25_25_00_50"
        x["dynamic_hold_name"] = "h1m1_raw_open_gap_sharpe_refine"
        x["buy_open_gap_pct"] = x["buy_open_gap_raw"] * 100.0
        drop_cols = ["signal_pct_chg", "buy_open_gap_raw", "buy_open_gap_qfq_from_signal_close", "atr_pct_qfq"]
        x = x.drop(columns=[col for col in drop_cols if col in x.columns])

        out_dir = OUT_DIR / case["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        signal_file = out_dir / "signals.csv"
        x.to_csv(signal_file, index=False, encoding="utf-8")

        counts = x.groupby("signal_date").size()
        manifest_rows.append(
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

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(OUT_DIR / "candidate_manifest.csv", index=False, encoding="utf-8-sig")
    print(manifest.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
