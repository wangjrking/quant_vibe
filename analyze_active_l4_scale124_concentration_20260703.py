from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(r"D:\work\quant\quant_mcp")
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_active_l4_prod_repro_grid_20260703"
CACHE_DB = REPORT_DIR / "active_l4_wide_cache.duckdb"
SIGNAL_FILE = REPORT_DIR / "variants_refill_best_position_scale_fine" / "scale124_cap82.csv"
OUT_DIR = REPORT_DIR / "scale124_cap82_concentration"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    signal = pd.read_csv(SIGNAL_FILE, dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    con = duckdb.connect(str(CACHE_DB), read_only=True)
    try:
        wide = con.execute(
            """
            SELECT
                signal_date,
                buy_date,
                stock_code,
                ret_h1,
                ret_h2,
                ret_h3,
                ret_h5
            FROM active_l4_wide
            """
        ).fetchdf()
    finally:
        con.close()
    wide["signal_date"] = wide["signal_date"].astype(str)
    wide["buy_date"] = wide["buy_date"].astype(str)
    wide["stock_code"] = wide["stock_code"].astype(str)
    df = signal.merge(wide, on=["signal_date", "buy_date", "stock_code"], how="left", validate="one_to_one")
    df["target_pct"] = df["target_pct"].astype(float)
    for h in [1, 2, 3, 5]:
        df[f"proxy_pnl_h{h}"] = df["target_pct"] * df[f"ret_h{h}"]
    df["year"] = df["buy_date"].str.slice(0, 4)
    df["month"] = df["buy_date"].str.slice(0, 6)
    df["recent_rank"] = df["buy_date"].rank(method="dense").astype(int)
    max_rank = int(df["recent_rank"].max())
    df["gap_bucket"] = pd.cut(
        df["buy_open_gap_raw_pct"].astype(float),
        [-100, -2.5, -1.5, 0, 100],
        labels=["deep<=-2.5", "mid(-2.5,-1.5]", "mild(-1.5,0]", "pos>0"],
    )
    df["sigchg_bucket"] = pd.cut(
        df["signal_pct_chg"].astype(float),
        [-100, -2, 0, 2, 100],
        labels=["sig<=-2", "sig(-2,0]", "sig(0,2]", "sig>2"],
    )
    df["atr_bucket"] = pd.cut(
        df["buy_atr_qfq"].astype(float),
        [-1, 1, 2, 4, 1000],
        labels=["atr<1", "atr1-2", "atr2-4", "atr>=4"],
    )
    df.to_csv(OUT_DIR / "signal_with_proxy_returns.csv", index=False, encoding="utf-8-sig")

    def agg(group_cols: list[str], path: str) -> pd.DataFrame:
        out = (
            df.groupby(group_cols, dropna=False)
            .agg(
                rows=("stock_code", "size"),
                stocks=("stock_code", "nunique"),
                avg_target=("target_pct", "mean"),
                avg_h1=("ret_h1", "mean"),
                avg_h2=("ret_h2", "mean"),
                avg_h3=("ret_h3", "mean"),
                avg_h5=("ret_h5", "mean"),
                proxy_pnl_h2=("proxy_pnl_h2", "sum"),
                proxy_pnl_h5=("proxy_pnl_h5", "sum"),
            )
            .reset_index()
            .sort_values("proxy_pnl_h2", ascending=False)
        )
        out.to_csv(OUT_DIR / path, index=False, encoding="utf-8-sig")
        return out

    year = agg(["year"], "by_year.csv")
    month = agg(["month"], "by_month.csv")
    stock = agg(["stock_code", "name"], "by_stock.csv")
    gap = agg(["gap_bucket"], "by_gap_bucket.csv")
    sigchg = agg(["sigchg_bucket"], "by_sigchg_bucket.csv")
    atr = agg(["atr_bucket"], "by_atr_bucket.csv")

    recent_rows = []
    for n in [20, 60, 120, 240]:
        part = df[df["recent_rank"] > max_rank - n].copy()
        recent_rows.append(
            {
                "recent_days": n,
                "rows": len(part),
                "stocks": part["stock_code"].nunique(),
                "avg_target": part["target_pct"].mean(),
                "avg_h1": part["ret_h1"].mean(),
                "avg_h2": part["ret_h2"].mean(),
                "avg_h3": part["ret_h3"].mean(),
                "avg_h5": part["ret_h5"].mean(),
                "proxy_pnl_h2": part["proxy_pnl_h2"].sum(),
                "proxy_pnl_h5": part["proxy_pnl_h5"].sum(),
            }
        )
    recent = pd.DataFrame(recent_rows)
    recent.to_csv(OUT_DIR / "recent_windows.csv", index=False, encoding="utf-8-sig")

    total_h2 = df["proxy_pnl_h2"].sum()
    top_stock = stock.head(20).copy()
    top_month = month.head(20).copy()
    summary = {
        "signal_file": str(SIGNAL_FILE),
        "rows": int(len(df)),
        "buy_days": int(df["buy_date"].nunique()),
        "stocks": int(df["stock_code"].nunique()),
        "proxy_pnl_h2": float(total_h2),
        "proxy_pnl_h5": float(df["proxy_pnl_h5"].sum()),
        "top1_stock_h2_share": float(top_stock.iloc[0]["proxy_pnl_h2"] / total_h2) if total_h2 else None,
        "top5_stock_h2_share": float(top_stock.head(5)["proxy_pnl_h2"].sum() / total_h2) if total_h2 else None,
        "top1_month_h2_share": float(top_month.iloc[0]["proxy_pnl_h2"] / total_h2) if total_h2 else None,
        "top5_month_h2_share": float(top_month.head(5)["proxy_pnl_h2"].sum() / total_h2) if total_h2 else None,
        "worst_month_proxy_h2": float(month.tail(1).iloc[0]["proxy_pnl_h2"]),
        "worst_stock_proxy_h2": float(stock.tail(1).iloc[0]["proxy_pnl_h2"]),
    }
    pd.DataFrame([summary]).to_csv(OUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")

    print("SUMMARY")
    for k, v in summary.items():
        print(k, v)
    print("\nBY_YEAR")
    print(year.to_string(index=False))
    print("\nRECENT")
    print(recent.to_string(index=False))
    print("\nBY_GAP")
    print(gap.to_string(index=False))
    print("\nBY_SIGCHG")
    print(sigchg.to_string(index=False))
    print("\nBY_ATR")
    print(atr.to_string(index=False))
    print("\nTOP_STOCK")
    print(top_stock.head(10).to_string(index=False))
    print("\nTOP_MONTH")
    print(top_month.head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
