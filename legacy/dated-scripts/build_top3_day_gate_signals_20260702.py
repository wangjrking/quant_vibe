from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(r"D:\work\quant\quant_mcp")
BASE_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_latest_l4_weight_candidates_20260702"
OUT_DIR = BASE_DIR / "postrank_open_filter_candidates" / "top3_day_gate_candidates"
SIGNAL_FILE = BASE_DIR / "postrank_open_filter_candidates" / "low_to_flat_pct12_top3_pos40" / "signals.csv"
MARKET_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sig = pd.read_csv(SIGNAL_FILE, dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    con = duckdb.connect(str(MARKET_DB), read_only=True)
    try:
        con.register("sig", sig)
        enriched = con.execute(
            """
            WITH md AS (
                SELECT
                    trade_date,
                    stock_code,
                    close,
                    open,
                    pct_chg,
                    index_2000_close,
                    index_2000_open
                FROM STOCK_DAILY_DATA
                WHERE trade_date BETWEEN '20220606' AND '20260701'
            )
            SELECT
                sig.*,
                buy.open / NULLIF(sigmd.close, 0) - 1 AS open_gap,
                buy.index_2000_open / NULLIF(sigmd.index_2000_close, 0) - 1 AS idx_open_gap,
                sigmd.pct_chg AS signal_pct_chg
            FROM sig
            JOIN md sigmd ON sigmd.trade_date = sig.signal_date AND sigmd.stock_code = sig.stock_code
            JOIN md buy ON buy.trade_date = sig.buy_date AND buy.stock_code = sig.stock_code
            """
        ).fetchdf()
    finally:
        con.close()

    daily = (
        enriched.groupby("buy_date")
        .agg(
            idx_open_gap=("idx_open_gap", "mean"),
            avg_open_gap=("open_gap", "mean"),
            avg_signal_pct=("signal_pct_chg", "mean"),
            names=("stock_code", "count"),
        )
        .reset_index()
    )
    keep_days = daily[
        (daily["idx_open_gap"] >= -0.02)
        & (daily["idx_open_gap"] <= 0.02)
        & (daily["avg_open_gap"] >= -0.03)
        & (daily["avg_open_gap"] <= 0.0)
        & (daily["avg_signal_pct"] >= -8)
        & (daily["avg_signal_pct"] <= 5)
    ]["buy_date"]
    out = enriched[enriched["buy_date"].isin(set(keep_days))].copy()
    out = out.drop(columns=["open_gap", "idx_open_gap", "signal_pct_chg"])
    out["strategy_variant"] = "top3_pos40_daygate_idxm2p2_gapm3p0_pctm8p5"
    signal_file = OUT_DIR / "signals.csv"
    out.to_csv(signal_file, index=False, encoding="utf-8")
    summary = {
        "signal_file": str(signal_file),
        "source_signal_file": str(SIGNAL_FILE),
        "rows": int(len(out)),
        "buy_days": int(out["buy_date"].nunique()),
        "avg_names": float(out.groupby("buy_date").size().mean()),
        "rule": "idx_open_gap[-2%,2%], avg_open_gap[-3%,0], avg_signal_pct[-8%,5%]",
    }
    pd.DataFrame([summary]).to_csv(OUT_DIR / "candidate_manifest.csv", index=False, encoding="utf-8")
    print(pd.DataFrame([summary]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
