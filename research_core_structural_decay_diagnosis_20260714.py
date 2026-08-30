from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
L2_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
SOURCE = REPORT_DIR / "signals" / "core_buy_open_gap_variants" / "x125_drop_missing_gap.csv"
OUT_CSV = REPORT_DIR / "core_structural_decay_signal_diagnosis_20260714.csv"
GROUP_CSV = REPORT_DIR / "core_structural_decay_group_summary_20260714.csv"
REPORT_JSON = REPORT_DIR / "core_structural_decay_diagnosis_20260714.json"
REPORT_MD = REPORT_DIR / "核心信号结构衰减诊断_20260714.md"


FEATURES = [
    "pred_prob",
    "pred_1d",
    "pred_3d",
    "pred_5d",
    "pred_10d",
    "amount",
    "turnover_rate",
    "total_mv",
    "atr_qfq",
    "signal_pct_chg_raw",
    "buy_open_gap_raw_pct",
    "exec_open_gap_pct",
    "target_pct",
    "sort_score",
]


def _to_num(frame: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")


def load_signal() -> pd.DataFrame:
    df = pd.read_csv(SOURCE, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    _to_num(df, FEATURES)
    if "exec_open_gap_pct" not in df.columns:
        df["exec_open_gap_pct"] = df.get("buy_open_gap_raw_pct", df.get("buy_open_gap_pct"))
    else:
        df["exec_open_gap_pct"] = df["exec_open_gap_pct"].fillna(df.get("buy_open_gap_raw_pct")).fillna(df.get("buy_open_gap_pct"))
    df["buy_year"] = df["buy_date"].str.slice(0, 4)
    df["buy_month"] = df["buy_date"].str.slice(0, 6)
    return df


def add_forward_returns(df: pd.DataFrame) -> pd.DataFrame:
    con = duckdb.connect(str(L2_DB), read_only=True)
    try:
        prices = con.execute(
            """
            with cal as (
              select
                trade_date,
                lead(trade_date, 1) over(order by trade_date) as d1,
                lead(trade_date, 2) over(order by trade_date) as d2,
                lead(trade_date, 3) over(order by trade_date) as d3
              from (select distinct trade_date from STOCK_DAILY_DATA order by trade_date)
            )
            select
              b.stock_code,
              b.trade_date as buy_date,
              b.open as buy_open,
              d1.open as open_d1,
              d2.open as open_d2,
              d3.open as open_d3,
              b.pct_chg as buy_day_pct_chg,
              b.amount as buy_day_amount,
              b.turnover_rate as buy_day_turnover_rate
            from STOCK_DAILY_DATA b
            left join cal c on c.trade_date=b.trade_date
            left join STOCK_DAILY_DATA d1 on d1.trade_date=c.d1 and d1.stock_code=b.stock_code
            left join STOCK_DAILY_DATA d2 on d2.trade_date=c.d2 and d2.stock_code=b.stock_code
            left join STOCK_DAILY_DATA d3 on d3.trade_date=c.d3 and d3.stock_code=b.stock_code
            where b.trade_date between '20220607' and '20260714'
            """
        ).fetchdf()
    finally:
        con.close()
    prices["buy_date"] = prices["buy_date"].astype(str)
    out = df.merge(prices, on=["stock_code", "buy_date"], how="left")
    out["ret_o1"] = out["open_d1"] / out["buy_open"] - 1.0
    out["ret_o2"] = out["open_d2"] / out["buy_open"] - 1.0
    out["ret_o3"] = out["open_d3"] / out["buy_open"] - 1.0
    out["weighted_ret_o2"] = out["ret_o2"] * out["target_pct"]
    return out


def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    existing_market_cols = [
        "mkt_avg_pct",
        "mkt_median_pct",
        "mkt_up_ratio",
        "mkt_deep_down_ratio",
        "mkt_amount_sum",
        "mkt_amount_median",
    ]
    if "mkt_deep_down_ratio" not in df.columns and "mkt_down5_ratio" in df.columns:
        df["mkt_deep_down_ratio"] = df["mkt_down5_ratio"]
    if "mkt_up_ratio" in df.columns and "mkt_avg_pct" in df.columns:
        return df
    con = duckdb.connect(str(L2_DB), read_only=True)
    try:
        market = con.execute(
            """
            select
              trade_date as signal_date,
              avg(pct_chg) as mkt_avg_pct,
              median(pct_chg) as mkt_median_pct,
              avg(case when pct_chg > 0 then 1.0 else 0.0 end) as mkt_up_ratio,
              avg(case when pct_chg <= -5 then 1.0 else 0.0 end) as mkt_deep_down_ratio,
              sum(amount) as mkt_amount_sum,
              median(amount) as mkt_amount_median
            from STOCK_DAILY_DATA
            where trade_date between '20220606' and '20260714'
              and stock_code not like '%.BJ'
            group by trade_date
            """
        ).fetchdf()
    finally:
        con.close()
    market["signal_date"] = market["signal_date"].astype(str)
    merged = df.merge(market, on="signal_date", how="left", suffixes=("", "_l2"))
    for col in existing_market_cols:
        alt = f"{col}_l2"
        if col not in merged.columns and alt in merged.columns:
            merged[col] = merged[alt]
        elif col in merged.columns and alt in merged.columns:
            merged[col] = merged[col].fillna(merged[alt])
    return merged


def group_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = {
        "year": "buy_year",
        "signal_pct_bucket": pd.cut(
            df["signal_pct_chg_raw"],
            [-100, -10, -8, -5, -2, 0, 100],
            labels=["<=-10", "-10~-8", "-8~-5", "-5~-2", "-2~0", ">0"],
        ),
        "gap_bucket": pd.cut(
            df["exec_open_gap_pct"],
            [-100, -5, -2, 0, 1, 3, 100],
            labels=["<=-5", "-5~-2", "-2~0", "0~1", "1~3", ">3"],
        ),
        "pred10_bucket": pd.cut(
            df["pred_10d"],
            [0, 0.90, 0.95, 0.98, 0.995, 1.0],
            labels=["<.90", ".90-.95", ".95-.98", ".98-.995", ">=.995"],
        ),
        "mkt_up_bucket": pd.cut(
            df["mkt_up_ratio"],
            [0, 0.30, 0.45, 0.55, 0.70, 1.0],
            labels=["<=30%", "30-45%", "45-55%", "55-70%", ">70%"],
        ),
    }
    for name, key in groups.items():
        tmp = df.copy()
        tmp["_group"] = key if isinstance(key, pd.Series) else tmp[key]
        agg = tmp.groupby("_group", observed=False).agg(
            rows=("stock_code", "count"),
            buy_days=("buy_date", "nunique"),
            mean_target=("target_pct", "mean"),
            mean_ret_o2=("ret_o2", "mean"),
            median_ret_o2=("ret_o2", "median"),
            win_o2=("ret_o2", lambda x: float((x > 0).mean())),
            sum_weighted_ret_o2=("weighted_ret_o2", "sum"),
            mean_signal_pct=("signal_pct_chg_raw", "mean"),
            mean_gap=("exec_open_gap_pct", "mean"),
            mean_pred10=("pred_10d", "mean"),
            mean_mkt_up=("mkt_up_ratio", "mean"),
        ).reset_index()
        agg.insert(0, "group_type", name)
        agg = agg.rename(columns={"_group": "group"})
        rows.append(agg)
    return pd.concat(rows, ignore_index=True)


def feature_drift(df: pd.DataFrame) -> list[dict]:
    out = []
    base = df[df["buy_year"] == "2024"].copy()
    late = df[df["buy_year"].isin(["2025", "2026"])].copy()
    for col in [c for c in FEATURES + ["mkt_up_ratio", "mkt_deep_down_ratio", "mkt_avg_pct"] if c in df.columns]:
        b = pd.to_numeric(base[col], errors="coerce")
        l = pd.to_numeric(late[col], errors="coerce")
        out.append(
            {
                "feature": col,
                "mean_2024": float(b.mean()) if b.notna().any() else None,
                "mean_2025_2026": float(l.mean()) if l.notna().any() else None,
                "median_2024": float(b.median()) if b.notna().any() else None,
                "median_2025_2026": float(l.median()) if l.notna().any() else None,
                "diff_mean_late_minus_2024": float(l.mean() - b.mean()) if b.notna().any() and l.notna().any() else None,
            }
        )
    return out


def write_markdown(payload: dict, groups: pd.DataFrame) -> None:
    lines = [
        "# 核心信号结构衰减诊断",
        "",
        "## 结论",
        "",
        "- 本报告只做结构诊断，不发布新策略，不修改生产参数。",
        "- 诊断对象为当前表面最强核心候选 `x125_drop_missing_gap`。",
        "- 重点比较 2024 与 2025-2026 的信号收益、开盘缺口、信号日跌幅、模型分数和市场状态差异。",
        "",
        "## 年度代理收益",
        "",
        "| 年份 | 行数 | 买入日 | 平均 O+2 收益 | 胜率 | 加权收益和 | 平均信号日涨跌 | 平均买入日缺口 | 平均10D分数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    year = groups[groups["group_type"] == "year"].copy()
    for _, row in year.iterrows():
        lines.append(
            f"| {row['group']} | {int(row['rows'])} | {int(row['buy_days'])} | "
            f"{float(row['mean_ret_o2'])*100:.2f}% | {float(row['win_o2'])*100:.2f}% | "
            f"{float(row['sum_weighted_ret_o2']):.4f} | {float(row['mean_signal_pct']):.2f}% | "
            f"{float(row['mean_gap']):.2f}% | {float(row['mean_pred10']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## 关键漂移",
            "",
            "| 字段 | 2024均值 | 2025-2026均值 | 后期-2024 |",
            "|---|---:|---:|---:|",
        ]
    )
    for item in sorted(payload["feature_drift"], key=lambda x: abs(x["diff_mean_late_minus_2024"] or 0), reverse=True)[:12]:
        lines.append(
            f"| {item['feature']} | {item['mean_2024'] if item['mean_2024'] is not None else ''} | "
            f"{item['mean_2025_2026'] if item['mean_2025_2026'] is not None else ''} | "
            f"{item['diff_mean_late_minus_2024'] if item['diff_mean_late_minus_2024'] is not None else ''} |"
        )
    lines.extend(
        [
            "",
            "## 证据路径",
            "",
            f"- 明细 CSV：`{OUT_CSV}`",
            f"- 分组 CSV：`{GROUP_CSV}`",
            f"- JSON：`{REPORT_JSON}`",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    signal = load_signal()
    enriched = add_market_features(add_forward_returns(signal))
    groups = group_summary(enriched)
    payload = {
        "source": str(SOURCE),
        "l2_db": str(L2_DB),
        "rows": int(len(enriched)),
        "buy_days": int(enriched["buy_date"].nunique()),
        "min_buy_date": str(enriched["buy_date"].min()),
        "max_buy_date": str(enriched["buy_date"].max()),
        "feature_drift": feature_drift(enriched),
        "year_summary": groups[groups["group_type"] == "year"].to_dict("records"),
    }
    enriched.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    groups.to_csv(GROUP_CSV, index=False, encoding="utf-8-sig")
    REPORT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_markdown(payload, groups)
    print(json.dumps({"csv": str(OUT_CSV), "group_csv": str(GROUP_CSV), "json": str(REPORT_JSON), "report": str(REPORT_MD)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
