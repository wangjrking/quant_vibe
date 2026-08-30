from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(r"D:\work\quant\quant_mcp")
BASE_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_latest_l4_weight_candidates_20260702"
REPORT_DIR = BASE_DIR / "postrank_open_filter_candidates" / "repro_top3_pos40"
SIGNAL_FILE = BASE_DIR / "postrank_open_filter_candidates" / "low_to_flat_pct12_top3_pos40" / "signals.csv"
MARKET_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"


def annualized(mean_daily: float) -> float:
    return (1 + mean_daily) ** 252 - 1


def sharpe(s: pd.Series) -> float:
    return float(s.mean() / s.std() * (252**0.5))


def mdd(s: pd.Series) -> float:
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
                SELECT trade_date, stock_code, open, close
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
                sig.name,
                CAST(sig.target_pct AS DOUBLE) AS target_pct,
                buy.open AS buy_open,
                nxt.open AS next_open,
                buy.close AS buy_close,
                nxt.open / NULLIF(buy.open, 0) - 1 AS ret_oo,
                buy.close / NULLIF(buy.open, 0) - 1 AS ret_oc
            FROM sig
            JOIN md buy ON buy.trade_date = sig.buy_date AND buy.stock_code = sig.stock_code
            JOIN cal ON cal.trade_date = sig.buy_date
            JOIN md nxt ON nxt.trade_date = cal.next_date AND nxt.stock_code = sig.stock_code
            """
        ).fetchdf()
    finally:
        con.close()

    daily = (
        trades.assign(weighted_oo=trades["ret_oo"] * trades["target_pct"])
        .groupby("buy_date")
        .agg(
            daily_oo_equal=("ret_oo", "mean"),
            daily_oo_target=("weighted_oo", "sum"),
            names=("stock_code", "count"),
            target_sum=("target_pct", "sum"),
        )
        .reset_index()
    )
    summary = {
        "signal_rows": int(len(sig)),
        "trade_rows_with_next_open": int(len(trades)),
        "buy_days": int(len(daily)),
        "avg_names": float(daily["names"].mean()),
        "avg_target_sum": float(daily["target_sum"].mean()),
        "equal_annual": annualized(float(daily["daily_oo_equal"].mean())),
        "equal_sharpe": sharpe(daily["daily_oo_equal"]),
        "equal_mdd": mdd(daily["daily_oo_equal"]),
        "target_annual": annualized(float(daily["daily_oo_target"].mean())),
        "target_sharpe": sharpe(daily["daily_oo_target"]),
        "target_mdd": mdd(daily["daily_oo_target"]),
    }
    pd.DataFrame([summary]).to_csv(REPORT_DIR / "local_return_profile_summary.csv", index=False, encoding="utf-8")
    daily.sort_values("daily_oo_target").head(30).to_csv(REPORT_DIR / "worst_30_days.csv", index=False, encoding="utf-8")
    daily.sort_values("daily_oo_target", ascending=False).head(30).to_csv(
        REPORT_DIR / "best_30_days.csv", index=False, encoding="utf-8"
    )
    trades.to_csv(REPORT_DIR / "trade_return_rows.csv", index=False, encoding="utf-8")
    print(pd.DataFrame([summary]).to_string(index=False))
    print("worst")
    print(daily.sort_values("daily_oo_target").head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
