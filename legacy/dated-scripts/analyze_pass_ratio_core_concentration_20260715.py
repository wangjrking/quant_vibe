from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_pass_ratio_open_gap_sellfreq_20260715"
L2_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"

SIGNALS = {
    "rh_top1_s120_cap065": REPORT_DIR
    / "signals"
    / "recalc_hardgate_scale"
    / "rh_top1_s120_cap065_full.csv",
    "x125_drop_missing_gap": ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_four_year_l4_frequency_optimization_20260714"
    / "signals"
    / "core_buy_open_gap_variants"
    / "x125_drop_missing_gap.csv",
    "x125_gap_le_1p0": ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_four_year_l4_frequency_optimization_20260714"
    / "signals"
    / "core_buy_open_gap_variants"
    / "x125_gap_le_1p0.csv",
}

JUEJIN_RESULTS = [
    REPORT_DIR / "pass_ratio_recalc_hardgate_scale_juejin_20260715.csv",
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_four_year_l4_frequency_optimization_20260714"
    / "core_buy_open_gap_variants_juejin_results_20260714.csv",
]

OUT_TRADES = REPORT_DIR / "core_concentration_proxy_trades_20260715.csv"
OUT_SUMMARY = REPORT_DIR / "core_concentration_proxy_summary_20260715.csv"
OUT_JSON = REPORT_DIR / "core_concentration_proxy_summary_20260715.json"
OUT_MD = REPORT_DIR / "core_concentration_proxy_review_20260715.md"


def load_signal(name: str, path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    df["case"] = name
    for col in [
        "target_pct",
        "signal_pct_chg_raw",
        "buy_open_gap_raw_pct",
        "exec_open_gap_pct",
        "pred_1d",
        "pred_5d",
        "pred_10d",
        "rank",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def attach_next_open(frame: pd.DataFrame) -> pd.DataFrame:
    keys = frame[["case", "buy_date", "stock_code"]].drop_duplicates().copy()
    con = duckdb.connect(str(L2_DB), read_only=True)
    con.register("need_keys", keys)
    market = con.execute(
        """
        WITH cal AS (
            SELECT trade_date, lead(trade_date) OVER (ORDER BY trade_date) AS next_trade_date
            FROM (SELECT DISTINCT trade_date FROM STOCK_DAILY_DATA)
        )
        SELECT
            k.case,
            k.buy_date,
            k.stock_code,
            c.next_trade_date,
            b.name AS buy_name,
            b.open AS buy_open_raw,
            n.open AS next_open_raw,
            n.open / NULLIF(b.open, 0) - 1 AS next_open_ret
        FROM need_keys k
        LEFT JOIN cal c ON c.trade_date = k.buy_date
        LEFT JOIN STOCK_DAILY_DATA b ON b.trade_date = k.buy_date AND b.stock_code = k.stock_code
        LEFT JOIN STOCK_DAILY_DATA n ON n.trade_date = c.next_trade_date AND n.stock_code = k.stock_code
        """
    ).fetchdf()
    con.close()
    return frame.merge(market, on=["case", "buy_date", "stock_code"], how="left")


def max_share(series: pd.Series) -> float:
    total = series.abs().sum()
    if not total:
        return 0.0
    return float(series.abs().max() / total)


def summarize_case(df: pd.DataFrame) -> dict:
    work = df.copy()
    work["proxy_pnl"] = pd.to_numeric(work["target_pct"], errors="coerce").fillna(0.0) * pd.to_numeric(
        work["next_open_ret"], errors="coerce"
    ).fillna(0.0)
    work["month"] = work["buy_date"].astype(str).str.slice(0, 6)
    work["year"] = work["buy_date"].astype(str).str.slice(0, 4)
    buy_days = sorted(work["buy_date"].dropna().astype(str).unique())
    recent60 = set(buy_days[-60:])
    recent120 = set(buy_days[-120:])

    by_stock = work.groupby(["stock_code", "name"], dropna=False)["proxy_pnl"].sum().sort_values(ascending=False)
    by_month = work.groupby("month")["proxy_pnl"].sum().sort_values(ascending=False)
    by_year = work.groupby("year")["proxy_pnl"].sum().sort_values(ascending=False)
    total_abs = work["proxy_pnl"].abs().sum()
    total = work["proxy_pnl"].sum()
    best_stock = by_stock.index[0] if len(by_stock) else ("", "")
    worst_stock = by_stock.sort_values().index[0] if len(by_stock) else ("", "")

    removed_top_stock = work[work["stock_code"] != best_stock[0]]["proxy_pnl"].sum() if len(by_stock) else total
    best_month = by_month.index[0] if len(by_month) else ""
    removed_top_month = work[work["month"] != best_month]["proxy_pnl"].sum() if best_month else total

    return {
        "case": str(work["case"].iloc[0]),
        "rows": int(len(work)),
        "buy_days": int(work["buy_date"].nunique()),
        "stock_count": int(work["stock_code"].nunique()),
        "min_buy": str(work["buy_date"].min()),
        "max_buy": str(work["buy_date"].max()),
        "mean_target_pct": float(work["target_pct"].mean()),
        "proxy_total_pnl": float(total),
        "proxy_win_rate": float((work["proxy_pnl"] > 0).mean()),
        "recent60_rows": int(work[work["buy_date"].astype(str).isin(recent60)].shape[0]),
        "recent120_rows": int(work[work["buy_date"].astype(str).isin(recent120)].shape[0]),
        "top_stock": str(best_stock[0]),
        "top_stock_name": str(best_stock[1]),
        "top_stock_proxy_pnl": float(by_stock.iloc[0]) if len(by_stock) else 0.0,
        "top_stock_abs_share": max_share(by_stock),
        "worst_stock": str(worst_stock[0]),
        "worst_stock_name": str(worst_stock[1]),
        "worst_stock_proxy_pnl": float(by_stock.sort_values().iloc[0]) if len(by_stock) else 0.0,
        "top_month": str(best_month),
        "top_month_proxy_pnl": float(by_month.iloc[0]) if len(by_month) else 0.0,
        "top_month_abs_share": max_share(by_month),
        "top_year": str(by_year.index[0]) if len(by_year) else "",
        "top_year_proxy_pnl": float(by_year.iloc[0]) if len(by_year) else 0.0,
        "top_year_abs_share": max_share(by_year),
        "proxy_pnl_without_top_stock": float(removed_top_stock),
        "proxy_pnl_without_top_month": float(removed_top_month),
        "missing_next_open_rows": int(work["next_open_ret"].isna().sum()),
    }


def load_juejin() -> pd.DataFrame:
    rows = []
    for path in JUEJIN_RESULTS:
        if not path.exists():
            continue
        df = pd.read_csv(path, encoding="utf-8-sig")
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    frame = pd.concat(rows, ignore_index=True, sort=False)
    if "slice" in frame.columns:
        frame = frame[(frame["slice"].isna()) | (frame["slice"].astype(str).eq("full"))].copy()
    return frame


def fmt_pct(value: object) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value) * 100:.2f}%"


def main() -> None:
    signals = [load_signal(name, path) for name, path in SIGNALS.items()]
    merged = attach_next_open(pd.concat(signals, ignore_index=True, sort=False))
    merged.to_csv(OUT_TRADES, index=False, encoding="utf-8-sig")
    summary = pd.DataFrame([summarize_case(part) for _, part in merged.groupby("case", sort=False)])
    juejin = load_juejin()
    if not juejin.empty:
        keep_cols = [
            "case",
            "pnl_ratio_annual",
            "sharp_ratio",
            "max_drawdown",
            "open_count",
            "close_count",
        ]
        summary = summary.merge(juejin[[col for col in keep_cols if col in juejin.columns]], on="case", how="left")
    summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")
    OUT_JSON.write_text(json.dumps(summary.to_dict("records"), ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    lines = [
        "# 核心策略集中度与开盘缺口候选审查 20260715",
        "",
        "## 说明",
        "",
        "- 正式收益指标仍以掘金为准。",
        "- 本报告的 `proxy_pnl` 只用未复权买入日开盘到下一交易日开盘估算贡献集中度，不能替代掘金收益。",
        "- 本轮对比当前核心版和两个开盘缺口收紧候选。",
        "",
        "## 汇总",
        "",
        "| 版本 | 掘金年化 | Sharpe | 最大回撤 | 买入日 | 最新买入日 | 代理胜率 | Top股票贡献占比 | Top月份贡献占比 | 去Top股票代理PnL | 去Top月份代理PnL |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.to_dict("records"):
        lines.append(
            f"| `{row['case']}` | {fmt_pct(row.get('pnl_ratio_annual'))} | "
            f"{float(row.get('sharp_ratio') or 0):.4f} | {fmt_pct(row.get('max_drawdown'))} | "
            f"{int(row['buy_days'])} | {row['max_buy']} | {fmt_pct(row['proxy_win_rate'])} | "
            f"{fmt_pct(row['top_stock_abs_share'])} | {fmt_pct(row['top_month_abs_share'])} | "
            f"{row['proxy_pnl_without_top_stock']:.4f} | {row['proxy_pnl_without_top_month']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 证据路径",
            "",
            f"- 逐笔代理贡献：`{OUT_TRADES}`",
            f"- 汇总 CSV：`{OUT_SUMMARY}`",
            f"- 汇总 JSON：`{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_MD)


if __name__ == "__main__":
    main()
