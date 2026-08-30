from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(r"D:\work\quant\quant_mcp")
BASE_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_latest_l4_weight_candidates_20260702"
OUT_DIR = BASE_DIR / "postrank_open_filter_candidates" / "soft_penalty_candidates"
SOURCE_SIGNAL_FILE = BASE_DIR / "w25_25_00_50_gapm8p3_top8" / "signals.csv"
MARKET_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"


CASES = [
    {
        "name": "softv1_top3_p40_force1d",
        "gap_low": -0.05,
        "gap_high": 0.0,
        "pct_abs_cap": 12.0,
        "atr_penalty": 0.08,
        "abs_pct_penalty": 0.03,
        "amount_bonus": 0.02,
        "keep_topn": 3,
        "target_pct": "0.40000",
    },
    {
        "name": "softv2_top3_p40_force1d",
        "gap_low": -0.05,
        "gap_high": 0.0,
        "pct_abs_cap": 12.0,
        "atr_penalty": 0.14,
        "abs_pct_penalty": 0.05,
        "amount_bonus": 0.04,
        "keep_topn": 3,
        "target_pct": "0.40000",
    },
    {
        "name": "softv3_top3_p34_force1d",
        "gap_low": -0.05,
        "gap_high": 0.0,
        "pct_abs_cap": 12.0,
        "atr_penalty": 0.10,
        "abs_pct_penalty": 0.04,
        "amount_bonus": 0.04,
        "keep_topn": 3,
        "target_pct": "0.34000",
    },
    {
        "name": "softv4_top5_p20_force1d",
        "gap_low": -0.05,
        "gap_high": 0.0,
        "pct_abs_cap": 12.0,
        "atr_penalty": 0.10,
        "abs_pct_penalty": 0.04,
        "amount_bonus": 0.04,
        "keep_topn": 5,
        "target_pct": "0.20000",
    },
    {
        "name": "softv5_top5_p22_force1d",
        "gap_low": -0.05,
        "gap_high": 0.0,
        "pct_abs_cap": 10.0,
        "atr_penalty": 0.06,
        "abs_pct_penalty": 0.03,
        "amount_bonus": 0.03,
        "keep_topn": 5,
        "target_pct": "0.22000",
    },
    {
        "name": "softv6_top8_p12_force1d",
        "gap_low": -0.08,
        "gap_high": 0.01,
        "pct_abs_cap": 12.0,
        "atr_penalty": 0.08,
        "abs_pct_penalty": 0.03,
        "amount_bonus": 0.05,
        "keep_topn": 8,
        "target_pct": "0.12000",
    },
]


def _rank01(frame: pd.DataFrame, col: str, ascending: bool) -> pd.Series:
    return frame.groupby("signal_date")[col].rank(method="average", pct=True, ascending=ascending).fillna(0.5)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sig = pd.read_csv(SOURCE_SIGNAL_FILE, dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    con = duckdb.connect(str(MARKET_DB), read_only=True)
    try:
        con.register("sig", sig)
        enriched = con.execute(
            """
            WITH md AS (
                SELECT trade_date, stock_code, open, close, pct_chg, amount, atr_qfq
                FROM STOCK_DAILY_DATA
                WHERE trade_date BETWEEN '20220606' AND '20260701'
            )
            SELECT
                sig.*,
                sigmd.close AS signal_close,
                sigmd.pct_chg AS signal_pct_chg,
                sigmd.amount AS signal_amount,
                sigmd.atr_qfq / NULLIF(sigmd.close, 0) AS signal_atr_ratio,
                buy.open / NULLIF(sigmd.close, 0) - 1 AS open_gap
            FROM sig
            JOIN md sigmd ON sigmd.trade_date = sig.signal_date AND sigmd.stock_code = sig.stock_code
            JOIN md buy ON buy.trade_date = sig.buy_date AND buy.stock_code = sig.stock_code
            """
        ).fetchdf()
    finally:
        con.close()

    enriched["atr_rank_bad"] = _rank01(enriched, "signal_atr_ratio", ascending=True)
    enriched["abs_pct"] = enriched["signal_pct_chg"].abs()
    enriched["abs_pct_rank_bad"] = _rank01(enriched, "abs_pct", ascending=True)
    enriched["amount_rank_good"] = _rank01(enriched, "signal_amount", ascending=True)

    rows = []
    for case in CASES:
        x = enriched[
            (enriched["open_gap"] >= float(case["gap_low"]))
            & (enriched["open_gap"] <= float(case["gap_high"]))
            & (enriched["abs_pct"] <= float(case["pct_abs_cap"]))
        ].copy()
        x["soft_score"] = (
            x["entry_score"]
            - float(case["atr_penalty"]) * x["atr_rank_bad"]
            - float(case["abs_pct_penalty"]) * x["abs_pct_rank_bad"]
            + float(case["amount_bonus"]) * x["amount_rank_good"]
        )
        x = x.sort_values(["signal_date", "soft_score", "entry_score"], ascending=[True, False, False])
        x = x.groupby("signal_date", group_keys=False).head(int(case["keep_topn"]))
        x["rank"] = x.groupby("signal_date").cumcount() + 1
        x["pred_prob"] = x["soft_score"]
        x["target_pct"] = case["target_pct"]
        x["holding_days"] = 1
        x["max_holding_days"] = 1
        x["score_continue_entry_ratio"] = "9.99000"
        x["strategy_variant"] = case["name"]
        x = x.drop(
            columns=[
                "signal_close",
                "signal_pct_chg",
                "signal_amount",
                "signal_atr_ratio",
                "open_gap",
                "atr_rank_bad",
                "abs_pct",
                "abs_pct_rank_bad",
                "amount_rank_good",
                "soft_score",
            ]
        )

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
                "buy_days": int(x["buy_date"].nunique()),
                "signal_days": int(counts.size),
                "avg_names": float(counts.mean()) if not counts.empty else 0.0,
                "days_below_target": int((counts < int(case["keep_topn"])).sum()) if not counts.empty else 0,
                **case,
            }
        )

    manifest = pd.DataFrame(rows)
    manifest.to_csv(OUT_DIR / "candidate_manifest.csv", index=False, encoding="utf-8")
    print(manifest.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
