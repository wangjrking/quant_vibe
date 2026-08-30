from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(r"D:\work\quant\quant_mcp")
BASE_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_latest_l4_weight_candidates_20260702"
OUT_DIR = BASE_DIR / "postrank_open_filter_candidates" / "safe_day_gate_candidates"
MARKET_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"


SOURCES = {
    "lowflat12_top6_p18": BASE_DIR
    / "postrank_open_filter_candidates"
    / "safe_midrange_refine_candidates"
    / "lowflat12_top6_p18"
    / "signals.csv",
    "nohigh12_top8_p11": BASE_DIR
    / "postrank_open_filter_candidates"
    / "safe_midrange_refine_candidates"
    / "nohigh12_top8_p11"
    / "signals.csv",
}


CASES = [
    {"name": "lf6p18_idx_ge_m1", "source": "lowflat12_top6_p18", "idx_min": -0.01},
    {"name": "lf6p18_idx_ge_m2", "source": "lowflat12_top6_p18", "idx_min": -0.02},
    {"name": "lf6p18_breadth_up40", "source": "lowflat12_top6_p18", "up_ratio_min": 0.40},
    {"name": "lf6p18_avg_sigpct_ge_m2", "source": "lowflat12_top6_p18", "avg_signal_pct_min": -2.0},
    {"name": "lf6p18_avg_opengap_ge_m3", "source": "lowflat12_top6_p18", "avg_open_gap_min": -0.03},
    {"name": "nh8p11_idx_ge_m1", "source": "nohigh12_top8_p11", "idx_min": -0.01},
    {"name": "nh8p11_breadth_up40", "source": "nohigh12_top8_p11", "up_ratio_min": 0.40},
]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(MARKET_DB), read_only=True)
    try:
        breadth = con.execute(
            """
            SELECT
                trade_date,
                AVG(CASE WHEN pct_chg > 0 THEN 1.0 ELSE 0.0 END) AS up_ratio,
                AVG(pct_chg) AS avg_market_pct
            FROM STOCK_DAILY_DATA
            WHERE trade_date BETWEEN '20220606' AND '20260701'
              AND pct_chg IS NOT NULL
            GROUP BY trade_date
            """
        ).fetchdf()
        md = con.execute(
            """
            SELECT trade_date, stock_code, open, close, pct_chg, index_2000_open, index_2000_close
            FROM STOCK_DAILY_DATA
            WHERE trade_date BETWEEN '20220606' AND '20260701'
            """
        ).fetchdf()
    finally:
        con.close()

    rows = []
    for case in CASES:
        sig = pd.read_csv(SOURCES[case["source"]], dtype={"signal_date": str, "buy_date": str, "stock_code": str})
        sig_md = md.rename(
            columns={
                "trade_date": "signal_date",
                "open": "signal_open",
                "close": "signal_close",
                "pct_chg": "signal_pct_chg",
                "index_2000_close": "signal_index_2000_close",
            }
        )[["signal_date", "stock_code", "signal_close", "signal_pct_chg", "signal_index_2000_close"]]
        buy_md = md.rename(
            columns={
                "trade_date": "buy_date",
                "open": "buy_open",
                "index_2000_open": "buy_index_2000_open",
            }
        )[["buy_date", "stock_code", "buy_open", "buy_index_2000_open"]]
        x = sig.merge(sig_md, on=["signal_date", "stock_code"], how="inner")
        x = x.merge(buy_md, on=["buy_date", "stock_code"], how="inner")
        x["open_gap"] = x["buy_open"] / x["signal_close"] - 1
        x["idx_open_gap"] = x["buy_index_2000_open"] / x["signal_index_2000_close"] - 1
        day = (
            x.groupby(["signal_date", "buy_date"])
            .agg(
                idx_open_gap=("idx_open_gap", "mean"),
                avg_signal_pct=("signal_pct_chg", "mean"),
                avg_open_gap=("open_gap", "mean"),
                names=("stock_code", "count"),
            )
            .reset_index()
        )
        day = day.merge(breadth.rename(columns={"trade_date": "signal_date"}), on="signal_date", how="left")
        keep = pd.Series(True, index=day.index)
        if "idx_min" in case:
            keep &= day["idx_open_gap"] >= float(case["idx_min"])
        if "up_ratio_min" in case:
            keep &= day["up_ratio"] >= float(case["up_ratio_min"])
        if "avg_signal_pct_min" in case:
            keep &= day["avg_signal_pct"] >= float(case["avg_signal_pct_min"])
        if "avg_open_gap_min" in case:
            keep &= day["avg_open_gap"] >= float(case["avg_open_gap_min"])
        keep_days = day.loc[keep, ["signal_date", "buy_date"]]
        out = sig.merge(keep_days, on=["signal_date", "buy_date"], how="inner")
        out["strategy_variant"] = case["name"]
        out_dir = OUT_DIR / case["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        signal_file = out_dir / "signals.csv"
        out.to_csv(signal_file, index=False, encoding="utf-8")
        counts = out.groupby("signal_date").size()
        rows.append(
            {
                "name": case["name"],
                "source": case["source"],
                "signal_file": str(signal_file),
                "rows": int(len(out)),
                "signal_days": int(counts.size),
                "avg_names": float(counts.mean()) if not counts.empty else 0.0,
                "dropped_days": int(sig["signal_date"].nunique() - counts.size),
                **case,
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(OUT_DIR / "candidate_manifest.csv", index=False, encoding="utf-8-sig")
    print(manifest.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
