from __future__ import annotations

import itertools
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(r"D:\work\quant\quant_mcp")
BASE_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_latest_l4_weight_candidates_20260702"
REPORT_DIR = BASE_DIR / "postrank_open_filter_candidates" / "day_gate_search"
SIGNAL_FILE = BASE_DIR / "postrank_open_filter_candidates" / "low_to_flat_pct12_top3_pos40" / "signals.csv"
MARKET_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"


def ann(x: float) -> float:
    return (1 + x) ** 252 - 1


def shp(s: pd.Series) -> float:
    return float(s.mean() / s.std() * (252**0.5))


def dd(s: pd.Series) -> float:
    nav = (1 + s.fillna(0)).cumprod()
    return float(-(nav / nav.cummax() - 1).min())


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    sig = pd.read_csv(SIGNAL_FILE, dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    con = duckdb.connect(str(MARKET_DB), read_only=True)
    try:
        con.register("sig", sig)
        trades = con.execute(
            """
            WITH md AS (
                SELECT
                    trade_date,
                    stock_code,
                    open,
                    close,
                    pct_chg,
                    index_2000_open,
                    index_2000_close
                FROM STOCK_DAILY_DATA
                WHERE trade_date BETWEEN '20220606' AND '20260701'
            ),
            cal AS (
                SELECT trade_date, lead(trade_date) OVER (ORDER BY trade_date) AS next_date
                FROM (SELECT DISTINCT trade_date FROM md)
            )
            SELECT
                sig.signal_date,
                sig.buy_date,
                sig.stock_code,
                CAST(sig.target_pct AS DOUBLE) AS target_pct,
                sigmd.close AS signal_close,
                sigmd.pct_chg AS signal_pct_chg,
                sigmd.index_2000_close AS signal_idx_close,
                buy.open AS buy_open,
                buy.index_2000_open AS buy_idx_open,
                nxt.open AS next_open,
                buy.open / NULLIF(sigmd.close, 0) - 1 AS open_gap,
                nxt.open / NULLIF(buy.open, 0) - 1 AS ret_oo,
                buy.index_2000_open / NULLIF(sigmd.index_2000_close, 0) - 1 AS idx_open_gap
            FROM sig
            JOIN md sigmd ON sigmd.trade_date = sig.signal_date AND sigmd.stock_code = sig.stock_code
            JOIN md buy ON buy.trade_date = sig.buy_date AND buy.stock_code = sig.stock_code
            JOIN cal ON cal.trade_date = sig.buy_date
            JOIN md nxt ON nxt.trade_date = cal.next_date AND nxt.stock_code = sig.stock_code
            """
        ).fetchdf()
    finally:
        con.close()

    daily = (
        trades.assign(weighted=trades["ret_oo"] * trades["target_pct"])
        .groupby("buy_date")
        .agg(
            ret=("weighted", "sum"),
            names=("stock_code", "count"),
            avg_open_gap=("open_gap", "mean"),
            min_open_gap=("open_gap", "min"),
            max_open_gap=("open_gap", "max"),
            avg_signal_pct=("signal_pct_chg", "mean"),
            idx_open_gap=("idx_open_gap", "mean"),
        )
        .reset_index()
    )

    rows = []
    idx_lows = [-0.02, -0.015, -0.01, -0.005, 0.0]
    idx_highs = [0.01, 0.02, 0.03]
    avg_gap_lows = [-0.04, -0.035, -0.03, -0.025]
    avg_gap_highs = [-0.005, 0.0]
    pct_lows = [-8, -5, -3, -1]
    pct_highs = [5, 8, 12]
    for idx_low, idx_high, avg_low, avg_high, pct_low, pct_high in itertools.product(
        idx_lows, idx_highs, avg_gap_lows, avg_gap_highs, pct_lows, pct_highs
    ):
        x = daily[
            (daily["idx_open_gap"] >= idx_low)
            & (daily["idx_open_gap"] <= idx_high)
            & (daily["avg_open_gap"] >= avg_low)
            & (daily["avg_open_gap"] <= avg_high)
            & (daily["avg_signal_pct"] >= pct_low)
            & (daily["avg_signal_pct"] <= pct_high)
        ]
        if len(x) < 500:
            continue
        rows.append(
            {
                "case_name": f"idx{idx_low}_{idx_high}_gap{avg_low}_{avg_high}_pct{pct_low}_{pct_high}",
                "idx_low": idx_low,
                "idx_high": idx_high,
                "avg_gap_low": avg_low,
                "avg_gap_high": avg_high,
                "pct_low": pct_low,
                "pct_high": pct_high,
                "buy_days": len(x),
                "annual": ann(float(x["ret"].mean())),
                "sharpe": shp(x["ret"]),
                "mdd": dd(x["ret"]),
                "mean_ret": float(x["ret"].mean()),
            }
        )
    out = pd.DataFrame(rows).sort_values(["sharpe", "annual"], ascending=False)
    out.to_csv(REPORT_DIR / "day_gate_summary.csv", index=False, encoding="utf-8")
    daily.to_csv(REPORT_DIR / "daily_features.csv", index=False, encoding="utf-8")
    print(out.head(30).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
