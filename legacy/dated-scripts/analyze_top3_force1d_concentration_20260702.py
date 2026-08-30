from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(r"D:\work\quant\quant_mcp")
BASE_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_latest_l4_weight_candidates_20260702"
REPORT_DIR = BASE_DIR / "postrank_open_filter_candidates" / "top3_force1d_concentration"
TRADE_ROWS = BASE_DIR / "postrank_open_filter_candidates" / "repro_top3_pos40" / "trade_return_rows.csv"
SIGNAL_FILE = BASE_DIR / "postrank_open_filter_candidates" / "low_to_flat_pct12_top3_pos40" / "signals.csv"
MARKET_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"


def annualized(mean_daily: float) -> float:
    return (1 + mean_daily) ** 252 - 1


def sharpe(s: pd.Series) -> float:
    std = s.std()
    if std == 0 or pd.isna(std):
        return 0.0
    return float(s.mean() / std * (252**0.5))


def mdd(s: pd.Series) -> float:
    nav = (1 + s.fillna(0)).cumprod()
    return float(-(nav / nav.cummax() - 1).min())


def metrics(daily_ret: pd.Series) -> dict[str, float]:
    return {
        "days": int(daily_ret.count()),
        "annual": annualized(float(daily_ret.mean())),
        "sharpe": sharpe(daily_ret),
        "mdd": mdd(daily_ret),
        "mean": float(daily_ret.mean()),
        "std": float(daily_ret.std()),
        "win_rate": float((daily_ret > 0).mean()),
    }


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    trades = pd.read_csv(TRADE_ROWS, dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    trades["target_pct"] = pd.to_numeric(trades["target_pct"], errors="coerce")
    trades["ret_oo"] = pd.to_numeric(trades["ret_oo"], errors="coerce")
    trades["weighted_oo"] = trades["ret_oo"] * trades["target_pct"]
    trades["month"] = trades["buy_date"].str.slice(0, 6)
    trades["year"] = trades["buy_date"].str.slice(0, 4)

    daily = (
        trades.groupby("buy_date")
        .agg(
            daily_ret=("weighted_oo", "sum"),
            equal_ret=("ret_oo", "mean"),
            names=("stock_code", "count"),
            target_sum=("target_pct", "sum"),
        )
        .reset_index()
        .sort_values("buy_date")
    )
    daily["month"] = daily["buy_date"].str.slice(0, 6)
    daily["year"] = daily["buy_date"].str.slice(0, 4)

    base = metrics(daily["daily_ret"])
    rows = [{"case": "base", **base}]
    for n in [1, 3, 5, 10, 20, 30]:
        worst_dates = set(daily.nsmallest(n, "daily_ret")["buy_date"])
        best_dates = set(daily.nlargest(n, "daily_ret")["buy_date"])
        rows.append({"case": f"remove_worst_{n}", **metrics(daily.loc[~daily["buy_date"].isin(worst_dates), "daily_ret"])})
        rows.append({"case": f"remove_best_{n}", **metrics(daily.loc[~daily["buy_date"].isin(best_dates), "daily_ret"])})

    month = daily.groupby("month").agg(month_ret=("daily_ret", lambda s: float((1 + s).prod() - 1)), days=("daily_ret", "count"))
    month = month.reset_index().sort_values("month_ret")
    year = daily.groupby("year").agg(year_ret=("daily_ret", lambda s: float((1 + s).prod() - 1)), days=("daily_ret", "count"))
    year = year.reset_index().sort_values("year")

    stock = (
        trades.groupby(["stock_code", "name"])
        .agg(
            trade_count=("weighted_oo", "count"),
            contribution=("weighted_oo", "sum"),
            avg_ret=("ret_oo", "mean"),
            win_rate=("ret_oo", lambda s: float((s > 0).mean())),
        )
        .reset_index()
        .sort_values("contribution", ascending=False)
    )

    # Local observable gates. These are research diagnostics only; Juejin validation is required before conclusions.
    sig = pd.read_csv(SIGNAL_FILE, dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    con = duckdb.connect(str(MARKET_DB), read_only=True)
    try:
        con.register("sig", sig)
        feats = con.execute(
            """
            WITH md AS (
                SELECT trade_date, stock_code, open, close, pct_chg, amount, turnover_rate, total_mv, atr_qfq,
                       index_2000_open, index_2000_close
                FROM STOCK_DAILY_DATA
                WHERE trade_date BETWEEN '20220606' AND '20260701'
            )
            SELECT
                sig.signal_date,
                sig.buy_date,
                sig.stock_code,
                buy.open / NULLIF(sigmd.close, 0) - 1 AS open_gap,
                sigmd.pct_chg AS signal_pct_chg,
                sigmd.atr_qfq / NULLIF(sigmd.close, 0) AS atr_ratio,
                sigmd.amount AS amount,
                sigmd.turnover_rate AS turnover_rate,
                buy.index_2000_open / NULLIF(sigmd.index_2000_close, 0) - 1 AS idx_open_gap
            FROM sig
            JOIN md sigmd ON sigmd.trade_date = sig.signal_date AND sigmd.stock_code = sig.stock_code
            JOIN md buy ON buy.trade_date = sig.buy_date AND buy.stock_code = sig.stock_code
            """
        ).fetchdf()
    finally:
        con.close()

    trade_feats = trades.merge(feats, on=["signal_date", "buy_date", "stock_code"], how="left")
    gate_cases = [
        ("base_local", pd.Series(True, index=trade_feats.index)),
        ("skip_idx_gap_lt_m2", trade_feats["idx_open_gap"] >= -0.02),
        ("skip_idx_gap_lt_m1", trade_feats["idx_open_gap"] >= -0.01),
        ("skip_open_gap_lt_m4", trade_feats["open_gap"] >= -0.04),
        ("skip_open_gap_lt_m3", trade_feats["open_gap"] >= -0.03),
        ("skip_signal_pct_abs_gt10", trade_feats["signal_pct_chg"].abs() <= 10),
        ("skip_signal_pct_abs_gt8", trade_feats["signal_pct_chg"].abs() <= 8),
        ("skip_atr_ratio_gt10", trade_feats["atr_ratio"] <= 0.10),
        ("skip_turnover_gt25", trade_feats["turnover_rate"] <= 25),
    ]
    gate_rows = []
    for name, mask in gate_cases:
        x = trade_feats[mask.fillna(False)].copy()
        d = x.groupby("buy_date").agg(daily_ret=("weighted_oo", "sum"), names=("stock_code", "count")).reset_index()
        m = metrics(d["daily_ret"]) if not d.empty else {"days": 0, "annual": 0, "sharpe": 0, "mdd": 0, "mean": 0, "std": 0, "win_rate": 0}
        gate_rows.append(
            {
                "case": name,
                "trade_rows": int(len(x)),
                "buy_days": int(d["buy_date"].nunique()) if not d.empty else 0,
                "avg_names": float(d["names"].mean()) if not d.empty else 0.0,
                **m,
            }
        )

    pd.DataFrame(rows).to_csv(REPORT_DIR / "concentration_sensitivity.csv", index=False, encoding="utf-8-sig")
    daily.sort_values("daily_ret").to_csv(REPORT_DIR / "daily_returns_sorted.csv", index=False, encoding="utf-8-sig")
    month.to_csv(REPORT_DIR / "monthly_contribution.csv", index=False, encoding="utf-8-sig")
    year.to_csv(REPORT_DIR / "yearly_contribution.csv", index=False, encoding="utf-8-sig")
    stock.head(50).to_csv(REPORT_DIR / "top_stock_contributors.csv", index=False, encoding="utf-8-sig")
    stock.tail(50).to_csv(REPORT_DIR / "worst_stock_contributors.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(gate_rows).to_csv(REPORT_DIR / "local_observable_gate_probe.csv", index=False, encoding="utf-8-sig")

    print("base", base)
    print("sensitivity")
    print(pd.DataFrame(rows).to_string(index=False))
    print("gates")
    print(pd.DataFrame(gate_rows).sort_values("sharpe", ascending=False).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
