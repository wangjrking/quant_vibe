from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(r"D:\work\quant\quant_mcp")
BASE_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_latest_l4_weight_candidates_20260702"
OUT_DIR = BASE_DIR / "postrank_open_filter_candidates" / "safe_midrange_candidates"
SOURCE_SIGNAL_FILE = BASE_DIR / "w25_25_00_50_gapm8p3_top8" / "signals.csv"
MARKET_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"


CASES = [
    {"name": "lowflat12_top4_p22", "gap_low": -0.05, "gap_high": 0.0, "pct_cap": 12.0, "keep_topn": 4, "target_pct": "0.22000"},
    {"name": "lowflat12_top4_p24", "gap_low": -0.05, "gap_high": 0.0, "pct_cap": 12.0, "keep_topn": 4, "target_pct": "0.24000"},
    {"name": "lowflat10_top4_p22", "gap_low": -0.05, "gap_high": 0.0, "pct_cap": 10.0, "keep_topn": 4, "target_pct": "0.22000"},
    {"name": "lowflat10_top5_p18", "gap_low": -0.05, "gap_high": 0.0, "pct_cap": 10.0, "keep_topn": 5, "target_pct": "0.18000"},
    {"name": "lowflat12_top6_p15", "gap_low": -0.05, "gap_high": 0.0, "pct_cap": 12.0, "keep_topn": 6, "target_pct": "0.15000"},
    {"name": "lowflat10_top6_p15", "gap_low": -0.05, "gap_high": 0.0, "pct_cap": 10.0, "keep_topn": 6, "target_pct": "0.15000"},
    {"name": "nohigh12_top6_p14", "gap_low": -0.08, "gap_high": 0.01, "pct_cap": 12.0, "keep_topn": 6, "target_pct": "0.14000"},
    {"name": "nohigh10_top6_p14", "gap_low": -0.08, "gap_high": 0.01, "pct_cap": 10.0, "keep_topn": 6, "target_pct": "0.14000"},
    {"name": "nohigh12_top7_p12", "gap_low": -0.08, "gap_high": 0.01, "pct_cap": 12.0, "keep_topn": 7, "target_pct": "0.12000"},
    {"name": "mildlow10_top6_p14", "gap_low": -0.03, "gap_high": 0.01, "pct_cap": 10.0, "keep_topn": 6, "target_pct": "0.14000"},
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
                SELECT trade_date, stock_code, open, close, pct_chg
                FROM STOCK_DAILY_DATA
                WHERE trade_date BETWEEN '20220606' AND '20260701'
            )
            SELECT
                sig.*,
                sigmd.pct_chg AS signal_pct_chg,
                buy.open / NULLIF(sigmd.close, 0) - 1 AS open_gap
            FROM sig
            JOIN md sigmd ON sigmd.trade_date = sig.signal_date AND sigmd.stock_code = sig.stock_code
            JOIN md buy ON buy.trade_date = sig.buy_date AND buy.stock_code = sig.stock_code
            """
        ).fetchdf()
    finally:
        con.close()

    rows = []
    for case in CASES:
        x = enriched[
            (enriched["open_gap"] >= float(case["gap_low"]))
            & (enriched["open_gap"] <= float(case["gap_high"]))
            & (enriched["signal_pct_chg"].abs() <= float(case["pct_cap"]))
        ].copy()
        x = x.sort_values(["signal_date", "entry_score"], ascending=[True, False])
        x = x.groupby("signal_date", group_keys=False).head(int(case["keep_topn"]))
        x["rank"] = x.groupby("signal_date").cumcount() + 1
        x["target_pct"] = case["target_pct"]
        x["holding_days"] = 1
        x["max_holding_days"] = 1
        x["score_continue_entry_ratio"] = "9.99000"
        x["strategy_variant"] = case["name"]
        x = x.drop(columns=["signal_pct_chg", "open_gap"])

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
    manifest.to_csv(OUT_DIR / "candidate_manifest.csv", index=False, encoding="utf-8-sig")
    print(manifest.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
