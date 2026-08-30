from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(r"D:\work\quant\quant_mcp")
BASE_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_latest_l4_weight_candidates_20260702"
OUT_DIR = BASE_DIR / "postrank_open_filter_candidates"
SIGNAL_FILE = BASE_DIR / "w25_25_00_50_gapm8p3_top8" / "signals.csv"
MARKET_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"


CASES = [
    {
        "name": "low_to_flat_pct12_top3",
        "gap_low": -0.05,
        "gap_high": 0.0,
        "pct_cap": 12.0,
        "keep_topn": 3,
        "target_pct": "0.30000",
    },
    {
        "name": "low_to_flat_pct12_top5",
        "gap_low": -0.05,
        "gap_high": 0.0,
        "pct_cap": 12.0,
        "keep_topn": 5,
        "target_pct": "0.18000",
    },
    {
        "name": "no_high_open_pct12_top8",
        "gap_low": -0.08,
        "gap_high": 0.01,
        "pct_cap": 12.0,
        "keep_topn": 8,
        "target_pct": "0.11000",
    },
]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sig = pd.read_csv(SIGNAL_FILE, dtype={"signal_date": str, "buy_date": str, "stock_code": str})
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
        out_dir = OUT_DIR / case["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        x = enriched[
            (enriched["open_gap"] >= float(case["gap_low"]))
            & (enriched["open_gap"] <= float(case["gap_high"]))
            & (enriched["pct_chg"].abs() <= float(case["pct_cap"]))
        ].copy()
        x = x.sort_values(["signal_date", "entry_score"], ascending=[True, False])
        x = x.groupby("signal_date", group_keys=False).head(int(case["keep_topn"]))
        x["rank"] = x.groupby("signal_date").cumcount() + 1
        x["target_pct"] = case["target_pct"]
        x["strategy_variant"] = case["name"]
        x = x.drop(columns=["open_gap"])
        signal_file = out_dir / "signals.csv"
        x.to_csv(signal_file, index=False, encoding="utf-8")
        counts = x.groupby("signal_date").size()
        rows.append(
            {
                "name": case["name"],
                "signal_file": str(signal_file),
                "rows": int(len(x)),
                "days": int(counts.size),
                "avg_names": float(counts.mean()),
                "days_below_target": int((counts < int(case["keep_topn"])).sum()),
                "target_pct": case["target_pct"],
            }
        )
    pd.DataFrame(rows).to_csv(OUT_DIR / "postrank_candidate_manifest.csv", index=False, encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
