from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
OUT_JSON = REPORT_DIR / "boundary_candidate_diagnostics_20260714.json"
OUT_MD = REPORT_DIR / "boundary_candidate_diagnostics_20260714.md"

CANDIDATES = [
    {
        "name": "annual_ge_500_best_sharpe",
        "case": "ss_mf_t2_sc1200_cap100_h3e98c102_sc096",
        "signal_file": REPORT_DIR
        / "signals"
        / "top3_sell_scale_refine"
        / "ss_mf_t2_sc1200_cap100_h3e98c102_sc096.csv",
        "juejin_annual": 5.0486279655,
        "juejin_sharpe": 3.9623013578,
        "juejin_mdd": 0.0375789655,
    },
    {
        "name": "sharpe_ge_4_best_annual",
        "case": "ss_mf_t2_sc1200_cap100_h3e98c102_sc092",
        "signal_file": REPORT_DIR
        / "signals"
        / "top3_sell_scale_refine"
        / "ss_mf_t2_sc1200_cap100_h3e98c102_sc092.csv",
        "juejin_annual": 4.4361263214,
        "juejin_sharpe": 4.0004090477,
        "juejin_mdd": 0.0360484920,
    },
    {
        "name": "best_sharpe",
        "case": "ss_hbb_t3_h78_m33_l18_h3e98c102_sc078",
        "signal_file": REPORT_DIR
        / "signals"
        / "top3_sell_scale_refine"
        / "ss_hbb_t3_h78_m33_l18_h3e98c102_sc078.csv",
        "juejin_annual": 2.9261175093,
        "juejin_sharpe": 4.0831014174,
        "juejin_mdd": 0.0333290432,
    },
]


MARKET_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
MARKET_TABLE = "STOCK_DAILY_DATA"


def _series_float(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _month_summary(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    work = df.copy()
    work["signal_month"] = work["signal_date"].astype(str).str.slice(0, 6)
    grouped = work.groupby("signal_month")
    return [
        {
            "month": str(month),
            "rows": int(len(part)),
            "buy_days": int(part["buy_date"].astype(str).nunique()),
            "target_sum": float(pd.to_numeric(part["target_pct"], errors="coerce").sum()),
            "avg_daily_target_sum": float(
                part.assign(target_pct_num=pd.to_numeric(part["target_pct"], errors="coerce"))
                .groupby("buy_date")["target_pct_num"]
                .sum()
                .mean()
            ),
        }
        for month, part in grouped
    ]


def _stock_summary(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    work = df.copy()
    work["target_pct_num"] = pd.to_numeric(work["target_pct"], errors="coerce").fillna(0.0)
    grouped = work.groupby(["stock_code", "name"], dropna=False)
    out = []
    for (stock_code, name), part in grouped:
        out.append(
            {
                "stock_code": str(stock_code),
                "name": str(name),
                "rows": int(len(part)),
                "buy_days": int(part["buy_date"].astype(str).nunique()),
                "target_sum": float(part["target_pct_num"].sum()),
            }
        )
    return sorted(out, key=lambda row: (row["rows"], row["target_sum"]), reverse=True)[:15]


def _join_next_open_return(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    keys = df[["buy_date", "stock_code"]].drop_duplicates().copy()
    dates = sorted(keys["buy_date"].astype(str).unique().tolist())
    if not dates:
        return df.copy()
    start, end = dates[0], dates[-1]
    with duckdb.connect(str(MARKET_DB), read_only=True) as con:
        market = con.execute(
            f"""
            WITH full_calendar AS (
                SELECT DISTINCT trade_date
                FROM {MARKET_TABLE}
            ),
            next_dates AS (
                SELECT trade_date, lead(trade_date) OVER (ORDER BY trade_date) AS next_trade_date
                FROM full_calendar
            )
            SELECT
                m.trade_date AS buy_date,
                m.stock_code,
                m.open AS buy_open_raw,
                m.pre_close AS buy_pre_close_raw,
                n.next_trade_date,
                m2.open AS next_open_raw
            FROM {MARKET_TABLE} m
            LEFT JOIN next_dates n ON n.trade_date = m.trade_date
            LEFT JOIN {MARKET_TABLE} m2
              ON m2.trade_date = n.next_trade_date AND m2.stock_code = m.stock_code
            WHERE m.trade_date BETWEEN ? AND ?
            """,
            [start, end],
        ).fetchdf()
    market["buy_date"] = market["buy_date"].astype(str)
    market["stock_code"] = market["stock_code"].astype(str)
    merged = df.merge(market, on=["buy_date", "stock_code"], how="left")
    merged["proxy_next_open_return"] = (
        pd.to_numeric(merged["next_open_raw"], errors="coerce")
        / pd.to_numeric(merged["buy_open_raw"], errors="coerce")
        - 1.0
    )
    merged["proxy_weighted_next_open_return"] = (
        pd.to_numeric(merged["target_pct"], errors="coerce").fillna(0.0)
        * merged["proxy_next_open_return"].fillna(0.0)
    )
    merged["buy_open_gap_raw_pct_proxy"] = (
        pd.to_numeric(merged["buy_open_raw"], errors="coerce")
        / pd.to_numeric(merged["buy_pre_close_raw"], errors="coerce")
        - 1.0
    ) * 100.0
    return merged


def diagnose(candidate: dict) -> dict:
    df = pd.read_csv(candidate["signal_file"], dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    target = _series_float(df, "target_pct").fillna(0.0)
    daily_target = target.groupby(df["buy_date"].astype(str)).sum()
    joined = _join_next_open_return(df)
    proxy_daily = joined.groupby(joined["buy_date"].astype(str))["proxy_weighted_next_open_return"].sum()
    proxy_month = joined.assign(month=joined["buy_date"].astype(str).str.slice(0, 6)).groupby("month")[
        "proxy_weighted_next_open_return"
    ].sum()
    return {
        "name": candidate["name"],
        "case": candidate["case"],
        "signal_file": str(candidate["signal_file"]),
        "juejin": {
            "annual": candidate["juejin_annual"],
            "sharpe": candidate["juejin_sharpe"],
            "max_drawdown": candidate["juejin_mdd"],
        },
        "coverage": {
            "rows": int(len(df)),
            "signal_days": int(df["signal_date"].astype(str).nunique()),
            "buy_days": int(df["buy_date"].astype(str).nunique()),
            "stock_count": int(df["stock_code"].astype(str).nunique()),
            "avg_names_per_buy_day": float(len(df) / max(1, df["buy_date"].astype(str).nunique())),
            "daily_target_sum_mean": float(daily_target.mean()) if not daily_target.empty else 0.0,
            "daily_target_sum_min": float(daily_target.min()) if not daily_target.empty else 0.0,
            "daily_target_sum_max": float(daily_target.max()) if not daily_target.empty else 0.0,
            "days_target_sum_below_50pct": int((daily_target < 0.5).sum()) if not daily_target.empty else 0,
            "days_target_sum_below_80pct": int((daily_target < 0.8).sum()) if not daily_target.empty else 0,
        },
        "feature_distribution": {
            "signal_pct_chg_raw_mean": float(_series_float(df, "signal_pct_chg_raw").mean()),
            "signal_pct_chg_raw_p10": float(_series_float(df, "signal_pct_chg_raw").quantile(0.1)),
            "signal_pct_chg_raw_p90": float(_series_float(df, "signal_pct_chg_raw").quantile(0.9)),
            "exec_open_gap_pct_mean": float(_series_float(df, "exec_open_gap_pct").mean()),
            "exec_open_gap_pct_p10": float(_series_float(df, "exec_open_gap_pct").quantile(0.1)),
            "exec_open_gap_pct_p90": float(_series_float(df, "exec_open_gap_pct").quantile(0.9)),
            "turnover_rate_mean": float(_series_float(df, "turnover_rate").mean()),
            "amount_median": float(_series_float(df, "amount").median()),
            "total_mv_median": float(_series_float(df, "total_mv").median()),
        },
        "month_coverage": _month_summary(df),
        "top_stock_concentration": _stock_summary(df),
        "proxy_next_open": {
            "available_rows": int(joined["proxy_next_open_return"].notna().sum()),
            "daily_proxy_mean": float(proxy_daily.mean()) if not proxy_daily.empty else 0.0,
            "daily_proxy_std": float(proxy_daily.std()) if len(proxy_daily) > 1 else 0.0,
            "positive_days": int((proxy_daily > 0).sum()) if not proxy_daily.empty else 0,
            "negative_days": int((proxy_daily < 0).sum()) if not proxy_daily.empty else 0,
            "best_month": str(proxy_month.idxmax()) if not proxy_month.empty else "",
            "best_month_proxy_return": float(proxy_month.max()) if not proxy_month.empty else 0.0,
            "worst_month": str(proxy_month.idxmin()) if not proxy_month.empty else "",
            "worst_month_proxy_return": float(proxy_month.min()) if not proxy_month.empty else 0.0,
        },
    }


def render_md(payload: dict) -> str:
    lines = [
        "# 边界候选诊断",
        "",
        "## 结论",
        "- 本报告只做本地代理诊断，不替代掘金回测结论。",
        "- 诊断对象是当前最接近 `年化>=500% 且 Sharpe>=4` 的边界候选。",
        "",
        "## 候选对比",
        "| 候选 | 掘金年化 | 掘金 Sharpe | 掘金回撤 | 行数 | 买入日 | 平均日仓位 | 低于80%仓位日 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in payload["candidates"]:
        c = item["coverage"]
        j = item["juejin"]
        lines.append(
            f"| `{item['case']}` | {j['annual']:.4f} | {j['sharpe']:.4f} | {j['max_drawdown']:.4f} | "
            f"{c['rows']} | {c['buy_days']} | {c['daily_target_sum_mean']:.4f} | {c['days_target_sum_below_80pct']} |"
        )
    lines.extend(["", "## 个股集中度 Top 10"])
    for item in payload["candidates"]:
        lines.append("")
        lines.append(f"### `{item['case']}`")
        lines.append("| 股票 | 名称 | 行数 | 买入日 | 目标仓位累计 |")
        lines.append("|---|---|---:|---:|---:|")
        for row in item["top_stock_concentration"][:10]:
            lines.append(
                f"| `{row['stock_code']}` | {row['name']} | {row['rows']} | {row['buy_days']} | {row['target_sum']:.4f} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"generated_at": pd.Timestamp.now().isoformat(), "candidates": [diagnose(c) for c in CANDIDATES]}
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"json": str(OUT_JSON), "md": str(OUT_MD)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
