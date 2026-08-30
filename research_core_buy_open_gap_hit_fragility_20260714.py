from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
BASE_SCRIPT = MAIN / "research_top1_three_tier_open_quality_20260714.py"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
L2_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
SOURCE_DIR = REPORT_DIR / "signals" / "core_buy_open_gap_variants"
SIGNAL_DIR = REPORT_DIR / "signals" / "core_buy_open_gap_hit_fragility"
LOG_DIR = REPORT_DIR / "logs" / "core_buy_open_gap_hit_fragility"
OUT_CSV = REPORT_DIR / "core_buy_open_gap_hit_fragility_juejin_results_20260714.csv"
OUT_JSON = REPORT_DIR / "core_buy_open_gap_hit_fragility_juejin_results_20260714.json"
PROXY_CSV = REPORT_DIR / "core_buy_open_gap_hit_proxy_contribution_20260714.csv"
SUMMARY_JSON = REPORT_DIR / "core_buy_open_gap_hit_fragility_summary_20260714.json"
REPORT_MD = REPORT_DIR / "核心买入日开盘缺口候选准入脆弱性复核_20260714.md"


spec = importlib.util.spec_from_file_location("base_three_tier", BASE_SCRIPT)
base_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base_mod)
base_mod.SIGNAL_DIR = SIGNAL_DIR
base_mod.LOG_DIR = LOG_DIR


SOURCES = {
    "x125_drop_missing_gap": SOURCE_DIR / "x125_drop_missing_gap.csv",
    "x125_gap_le_1p0": SOURCE_DIR / "x125_gap_le_1p0.csv",
    "x115_drop_missing_gap": SOURCE_DIR / "x115_drop_missing_gap.csv",
    "x115_gap_le_1p0": SOURCE_DIR / "x115_gap_le_1p0.csv",
}


def load_signal(name: str, path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    df["target_pct"] = pd.to_numeric(df["target_pct"], errors="coerce").fillna(0.0)
    df["source_case"] = name
    return df


def add_proxy_returns(all_df: pd.DataFrame) -> pd.DataFrame:
    con = duckdb.connect(str(L2_DB), read_only=True)
    try:
        prices = con.execute(
            """
            with cal as (
              select trade_date, lead(trade_date, 2) over(order by trade_date) as d2
              from (select distinct trade_date from STOCK_DAILY_DATA order by trade_date)
            )
            select b.stock_code, b.trade_date as buy_date, b.open as buy_open, e2.open as open_d2
            from STOCK_DAILY_DATA b
            left join cal c on c.trade_date=b.trade_date
            left join STOCK_DAILY_DATA e2 on e2.trade_date=c.d2 and e2.stock_code=b.stock_code
            where b.trade_date between '20220607' and '20260714'
            """
        ).fetchdf()
    finally:
        con.close()
    prices["buy_date"] = prices["buy_date"].astype(str)
    out = all_df.merge(prices, on=["stock_code", "buy_date"], how="left")
    out["ret_h2"] = out["open_d2"] / out["buy_open"] - 1.0
    out["weighted_ret_h2"] = out["target_pct"] * out["ret_h2"]
    out["month"] = out["buy_date"].str.slice(0, 6)
    out["year"] = out["buy_date"].str.slice(0, 4)
    out.to_csv(PROXY_CSV, index=False, encoding="utf-8-sig")
    return out


def cases_for_source(proxy: pd.DataFrame, source_case: str) -> list[dict]:
    part = proxy[proxy["source_case"] == source_case].copy()
    top_months = (
        part.groupby("month", as_index=False)["weighted_ret_h2"]
        .sum()
        .sort_values("weighted_ret_h2", ascending=False)
        .head(3)["month"]
        .astype(str)
        .tolist()
    )
    top_stocks = (
        part.groupby("stock_code", as_index=False)["weighted_ret_h2"]
        .sum()
        .sort_values("weighted_ret_h2", ascending=False)
        .head(3)["stock_code"]
        .astype(str)
        .tolist()
    )
    return [
        {"case": f"{source_case}_remove_top1_month", "source_case": source_case, "remove_months": top_months[:1], "remove_years": [], "remove_stocks": []},
        {"case": f"{source_case}_remove_top3_month", "source_case": source_case, "remove_months": top_months, "remove_years": [], "remove_stocks": []},
        {"case": f"{source_case}_remove_top1_stock", "source_case": source_case, "remove_months": [], "remove_years": [], "remove_stocks": top_stocks[:1]},
        {"case": f"{source_case}_remove_top3_stock", "source_case": source_case, "remove_months": [], "remove_years": [], "remove_stocks": top_stocks},
        {"case": f"{source_case}_remove_2024", "source_case": source_case, "remove_months": [], "remove_years": ["2024"], "remove_stocks": []},
    ]


def write_variant(source: pd.DataFrame, case: dict) -> dict:
    df = source.copy()
    if case.get("remove_months"):
        df = df[~df["buy_date"].str.slice(0, 6).isin(case["remove_months"])].copy()
    if case.get("remove_years"):
        df = df[~df["buy_date"].str.slice(0, 4).isin(case["remove_years"])].copy()
    if case.get("remove_stocks"):
        df = df[~df["stock_code"].isin(case["remove_stocks"])].copy()
    df["strategy_variant"] = case["case"]
    df["filter_name"] = case["case"]
    out = SIGNAL_DIR / f"{case['case']}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    daily_target = df.groupby("buy_date")["target_pct"].sum() if len(df) else pd.Series(dtype=float)
    return {
        "case": case["case"],
        "source_case": case["source_case"],
        "signal_file": str(out),
        "rows": int(len(df)),
        "buy_days": int(df["buy_date"].nunique()) if len(df) else 0,
        "stock_count": int(df["stock_code"].nunique()) if len(df) else 0,
        "mean_daily_target_sum": float(daily_target.mean()) if len(daily_target) else 0.0,
        "max_daily_target_sum": float(daily_target.max()) if len(daily_target) else 0.0,
        "remove_months": ",".join(case.get("remove_months", [])),
        "remove_years": ",".join(case.get("remove_years", [])),
        "remove_stocks": ",".join(case.get("remove_stocks", [])),
    }


def run_or_parse(row: dict) -> dict:
    log = LOG_DIR / f"{row['case']}.log"
    if log.exists() and log.stat().st_size > 0:
        ind = base_mod.extract_indicator(log.read_text(encoding="utf-8", errors="ignore"))
        if isinstance(ind, dict) and ind.get("pnl_ratio_annual") is not None:
            out = dict(row)
            out["returncode"] = 0
            out["log_file"] = str(log)
            out.update(ind)
            return out
    row = dict(row)
    row["max_positions"] = 1
    row["holding_days"] = 1
    row["max_holding_days"] = 3
    return base_mod.run_juejin(row)


def write_report(frame: pd.DataFrame) -> None:
    lines = [
        "# 核心买入日开盘涨幅候选准入脆弱性复核",
        "",
        "## 结论",
        "",
        "- 本报告只复核买入日开盘涨幅邻域中表面最强的四个候选。",
        "- 判断重点不是重新选最高收益，而是看去除关键月份、关键股票、早期年份后是否仍稳定满足目标。",
        "",
        "## 掘金复核结果",
        "",
        "| case | 年化 | Sharpe | 最大回撤 | 开仓 | 移除月份 | 移除年份 | 移除股票 |",
        "|---|---:|---:|---:|---:|---|---|---|",
    ]
    for _, row in frame.sort_values(["source_case", "case"]).iterrows():
        lines.append(
            f"| {row.get('case')} | {float(row.get('pnl_ratio_annual', 0))*100:.2f}% | "
            f"{float(row.get('sharp_ratio', 0)):.3f} | {float(row.get('max_drawdown', 0))*100:.2f}% | "
            f"{int(row.get('open_count', 0)) if pd.notna(row.get('open_count')) else ''} | "
            f"{row.get('remove_months', '')} | {row.get('remove_years', '')} | {row.get('remove_stocks', '')} |"
        )
    lines.extend(
        [
            "",
            "## 证据路径",
            "",
            f"- 汇总 CSV：`{OUT_CSV}`",
            f"- 汇总 JSON：`{OUT_JSON}`",
            f"- 贡献代理 CSV：`{PROXY_CSV}`",
            f"- 派生信号目录：`{SIGNAL_DIR}`",
            f"- 掘金日志目录：`{LOG_DIR}`",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    sources = {name: load_signal(name, path) for name, path in SOURCES.items()}
    proxy = add_proxy_returns(pd.concat(sources.values(), ignore_index=True))
    cases = []
    for source_case in sources:
        cases.extend(cases_for_source(proxy, source_case))
    manifests = [write_variant(sources[case["source_case"]], case) for case in cases]
    results = []
    for row in manifests:
        result = run_or_parse(row)
        results.append(result)
        frame = pd.DataFrame(results)
        frame.sort_values(["source_case", "case"], ascending=[True, True]).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
        OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(json.dumps({"case": result.get("case"), "annual": result.get("pnl_ratio_annual"), "sharpe": result.get("sharp_ratio"), "mdd": result.get("max_drawdown"), "open": result.get("open_count")}, ensure_ascii=False), flush=True)
    frame = pd.DataFrame(results)
    summary = {
        "sources": {k: str(v) for k, v in SOURCES.items()},
        "csv": str(OUT_CSV),
        "json": str(OUT_JSON),
        "proxy_csv": str(PROXY_CSV),
        "report": str(REPORT_MD),
        "results": results,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_report(frame)
    print(json.dumps(summary, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
