from __future__ import annotations

import itertools
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(r"D:\work\quant\quant_mcp")
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_latest_l4_weight_candidates_20260702" / "postrank_open_filter_search"
SIGNAL_FILE = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_latest_l4_weight_candidates_20260702"
    / "w25_25_00_50_gapm8p3_top8"
    / "signals.csv"
)
MARKET_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"


def _annualized(mean_daily: float) -> float:
    return (1 + mean_daily) ** 252 - 1


def _sharpe(mean_daily: float, std_daily: float) -> float | None:
    if std_daily == 0 or pd.isna(std_daily):
        return None
    return mean_daily / std_daily * (252**0.5)


def _mdd(s: pd.Series) -> float:
    nav = (1 + s.fillna(0)).cumprod()
    peak = nav.cummax()
    return float(-(nav / peak - 1).min())


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    sig = pd.read_csv(SIGNAL_FILE, dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    con = duckdb.connect(str(MARKET_DB), read_only=True)
    try:
        con.register("sig", sig)
        df = con.execute(
            """
            WITH md AS (
                SELECT trade_date, stock_code, open, close, amount, total_mv, turnover_rate, atr_qfq, pct_chg
                FROM STOCK_DAILY_DATA
                WHERE trade_date BETWEEN '20220606' AND '20260701'
            ),
            cal AS (
                SELECT trade_date, lead(trade_date) OVER (ORDER BY trade_date) AS next_date
                FROM (SELECT DISTINCT trade_date FROM md)
            )
            SELECT
                sig.*,
                sigmd.close AS signal_close,
                buy.open AS buy_open,
                buy.close AS buy_close,
                nxt.open AS next_open,
                buy.open / NULLIF(sigmd.close, 0) - 1 AS open_gap,
                nxt.open / NULLIF(buy.open, 0) - 1 AS ret_oo,
                buy.close / NULLIF(buy.open, 0) - 1 AS ret_oc,
                sigmd.atr_qfq / NULLIF(sigmd.close, 0) AS atr_pct
            FROM sig
            JOIN md sigmd ON sigmd.trade_date = sig.signal_date AND sigmd.stock_code = sig.stock_code
            JOIN md buy ON buy.trade_date = sig.buy_date AND buy.stock_code = sig.stock_code
            JOIN cal ON cal.trade_date = sig.buy_date
            JOIN md nxt ON nxt.trade_date = cal.next_date AND nxt.stock_code = sig.stock_code
            """
        ).fetchdf()
    finally:
        con.close()

    gap_rules = [
        ("base", -0.08, 0.03),
        ("no_high_open", -0.08, 0.01),
        ("low_to_flat", -0.05, 0.00),
        ("mild_low", -0.03, 0.01),
        ("flat_high", 0.00, 0.03),
    ]
    amount_mins = [0, 90_000, 150_000, 300_000]
    atr_caps = [None, 0.06, 0.09]
    pct_caps = [None, 8, 12]
    keep_topns = [3, 5, 8]
    rows = []
    for (gname, gl, gh), amount_min, atr_cap, pct_cap, keep_topn in itertools.product(
        gap_rules, amount_mins, atr_caps, pct_caps, keep_topns
    ):
        x = df[(df["open_gap"] >= gl) & (df["open_gap"] <= gh) & (df["amount"] >= amount_min)].copy()
        if atr_cap is not None:
            x = x[x["atr_pct"].notna() & (x["atr_pct"] <= atr_cap)]
        if pct_cap is not None:
            x = x[x["pct_chg"].abs() <= pct_cap]
        x = x.sort_values(["signal_date", "entry_score"], ascending=[True, False])
        x = x.groupby("signal_date", group_keys=False).head(keep_topn)
        if x.empty:
            continue
        daily = x.groupby("signal_date").agg(daily_oo=("ret_oo", "mean"), names=("stock_code", "count")).reset_index()
        coverage = len(daily) / sig["signal_date"].nunique()
        avg_names = float(daily["names"].mean())
        if coverage < 0.85 or avg_names < max(2.5, keep_topn * 0.65):
            continue
        mean = float(daily["daily_oo"].mean())
        rows.append(
            {
                "case_name": f"{gname}_amt{amount_min}_atr{atr_cap or 'none'}_pct{pct_cap or 'none'}_top{keep_topn}",
                "gap_low": gl,
                "gap_high": gh,
                "amount_min": amount_min,
                "atr_cap": atr_cap,
                "pct_cap": pct_cap,
                "keep_topn": keep_topn,
                "signal_days": len(daily),
                "coverage": coverage,
                "avg_names": avg_names,
                "annual_oo": _annualized(mean),
                "sharpe_oo": _sharpe(mean, float(daily["daily_oo"].std())),
                "mdd_oo": _mdd(daily["daily_oo"]),
                "recent60_annual_oo": _annualized(float(daily.tail(60)["daily_oo"].mean())),
                "rows": len(x),
            }
        )
    out = pd.DataFrame(rows).sort_values(["annual_oo", "sharpe_oo"], ascending=False)
    out.to_csv(REPORT_DIR / "postrank_open_filter_summary.csv", index=False, encoding="utf-8")
    print(out.head(30).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
