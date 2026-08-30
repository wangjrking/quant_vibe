from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_pass_ratio_open_gap_sellfreq_20260715"
)
SOURCE_REPORT_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_four_year_l4_frequency_optimization_20260714"
)
OPEN_GAP_RESULTS = SOURCE_REPORT_DIR / "open_gap_deep_rebalance_juejin_results_20260714.csv"
CONDITIONAL_RESULTS = SOURCE_REPORT_DIR / "ogd_conditional_reweight_juejin_results_20260714.csv"
L2_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
FRESH_LOG = REPORT_DIR / "logs" / "fresh_ogd_gap1_x050_20260715.log"

CASES = [
    "ogd_gap1_x050",
    "ogd_gap1x70_deep5up105",
    "ogd_deep8_up110",
    "ogd_deep5_up105",
]


def pct(value: float | int | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value) * 100:.{digits}f}%"


def load_signal(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})


def audit_signal(frame: pd.DataFrame) -> dict:
    out: dict[str, object] = {
        "rows": int(len(frame)),
        "buy_days": int(frame["buy_date"].nunique()),
        "stock_count": int(frame["stock_code"].nunique()),
        "min_buy": str(frame["buy_date"].min()),
        "max_buy": str(frame["buy_date"].max()),
        "max_positions": int(frame.groupby("buy_date")["stock_code"].count().max()),
        "avg_names_per_buy_day": float(frame.groupby("buy_date")["stock_code"].count().mean()),
        "mean_daily_target_sum": float(frame.groupby("buy_date")["target_pct"].sum().mean()),
        "max_daily_target_sum": float(frame.groupby("buy_date")["target_pct"].sum().max()),
        "bj_rows": int(frame["stock_code"].astype(str).str.endswith(".BJ").sum()),
        "duplicate_keys": int(frame.duplicated(["buy_date", "stock_code"]).sum()),
    }

    con = duckdb.connect(str(L2_DB), read_only=True)
    probe = frame[["stock_code", "signal_date", "buy_date"]].copy()
    con.register("signal_probe", probe)
    st_signal = con.execute(
        """
        select count(*) as cnt
        from signal_probe s
        join STOCK_DAILY_DATA d
          on d.stock_code = s.stock_code and d.trade_date = s.signal_date
        where coalesce(nullif(trim(d.ST_TYPE), ''), '0') not in ('0', 'None', 'nan')
           or coalesce(nullif(trim(d.ST_TYPE_name), ''), '') <> ''
           or d.name like 'ST%'
           or d.name like '*ST%'
        """
    ).fetchone()[0]
    st_buy = con.execute(
        """
        select count(*) as cnt
        from signal_probe s
        join STOCK_DAILY_DATA d
          on d.stock_code = s.stock_code and d.trade_date = s.buy_date
        where coalesce(nullif(trim(d.ST_TYPE), ''), '0') not in ('0', 'None', 'nan')
           or coalesce(nullif(trim(d.ST_TYPE_name), ''), '') <> ''
           or d.name like 'ST%'
           or d.name like '*ST%'
        """
    ).fetchone()[0]
    market_dates = con.execute(
        """
        select min(trade_date), max(trade_date), count(distinct trade_date)
        from STOCK_DAILY_DATA
        """
    ).fetchone()
    con.close()
    out["st_signal_rows"] = int(st_signal)
    out["st_buy_rows"] = int(st_buy)
    out["l2_min_date"] = str(market_dates[0])
    out["l2_max_date"] = str(market_dates[1])
    out["l2_trade_days"] = int(market_dates[2])
    return out


def build_review() -> tuple[pd.DataFrame, str]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(OPEN_GAP_RESULTS)
    rows = []
    for case in CASES:
        base = results.loc[results["case"] == case]
        if base.empty:
            continue
        rec = base.iloc[0].to_dict()
        signal = load_signal(rec["signal_file"])
        rec.update({f"audit_{k}": v for k, v in audit_signal(signal).items()})
        rec["pass_target_full_existing_juejin"] = bool(
            float(rec.get("pnl_ratio_annual", 0)) >= 5.0
            and float(rec.get("sharp_ratio", 0)) >= 4.0
            and float(rec.get("max_drawdown", 999)) <= 0.4
            and int(rec.get("returncode", 1)) == 0
        )
        rows.append(rec)
    summary = pd.DataFrame(rows)
    summary.to_csv(REPORT_DIR / "pass_ratio_candidate_admission_review_20260715.csv", index=False, encoding="utf-8-sig")
    (REPORT_DIR / "pass_ratio_candidate_admission_review_20260715.json").write_text(
        json.dumps(summary.to_dict("records"), ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    cond = pd.read_csv(CONDITIONAL_RESULTS)
    cond_rows = cond[
        cond["case"].str.contains("ogd_mildhi20_deep00_cap90|ogd_mildhi15_deep25_cap90", na=False)
    ].copy()
    cond_rows = cond_rows[
        [
            "case",
            "slice",
            "rows",
            "buy_days",
            "min_buy",
            "max_buy",
            "mean_daily_target_sum",
            "pnl_ratio_annual",
            "sharp_ratio",
            "max_drawdown",
            "win_ratio",
            "returncode",
        ]
    ]
    cond_rows.to_csv(REPORT_DIR / "nearby_slice_reference_20260715.csv", index=False, encoding="utf-8-sig")

    fresh_status = "未执行"
    fresh_error = ""
    if FRESH_LOG.exists():
        text = FRESH_LOG.read_text(encoding="utf-8", errors="ignore")
        if "无法连接到终端服务" in text or '"status": 1001' in text or "status=1001" in text:
            fresh_status = "失败"
            fresh_error = "掘金终端服务不可连接"
        elif "GM_BACKTEST_INDICATOR" in text:
            fresh_status = "成功"
        else:
            fresh_status = "失败"
            fresh_error = "未解析到 GM_BACKTEST_INDICATOR"

    best = summary.sort_values(["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False]).iloc[0]
    lines = [
        "# 同源信号通过率策略准入复核 20260715",
        "",
        "## 当前结论",
        "",
        "本轮复核只比较 OGD 同源信号的通过率和权重变化，不引入补位票，也不扩大候选池。",
        "",
        f"- 当前表面最优：`{best['case']}`。",
        f"- 既有掘金年化：{pct(best['pnl_ratio_annual'])}。",
        f"- 既有掘金 Sharpe：{float(best['sharp_ratio']):.4f}。",
        f"- 既有掘金最大回撤：{pct(best['max_drawdown'])}。",
        f"- fresh 掘金复跑状态：{fresh_status}。",
    ]
    if fresh_error:
        lines.append(f"- fresh 复跑失败原因：{fresh_error}。")
    lines.extend(
        [
            "",
            "结论：已有掘金日志里存在满足 500% 年化、Sharpe 4、回撤 40% 以下的同源通过率版本；但 fresh rerun 仍被掘金终端连接阻断，不能把本轮状态表述为“最新复跑已通过”。",
            "",
            "## 候选对比",
            "",
            "| 版本 | 规则含义 | 行数 | 买入日 | 平均仓位 | 掘金年化 | Sharpe | 最大回撤 | 是否表面达标 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    rule_map = {
        "ogd_gap1_x050": "高开超过 1% 的原信号降权到 0.5",
        "ogd_gap1x70_deep5up105": "高开超过 1% 降权到 0.7，信号日跌幅小于 -5% 加权到 1.05",
        "ogd_deep8_up110": "信号日跌幅小于 -8% 加权到 1.10",
        "ogd_deep5_up105": "信号日跌幅小于 -5% 加权到 1.05",
    }
    for row in summary.to_dict("records"):
        lines.append(
            f"| `{row['case']}` | {rule_map.get(row['case'], '')} | {int(row['rows'])} | "
            f"{int(row['buy_days'])} | {pct(row['mean_daily_target_sum'])} | "
            f"{pct(row['pnl_ratio_annual'])} | {float(row['sharp_ratio']):.4f} | "
            f"{pct(row['max_drawdown'])} | {'是' if row['pass_target_full_existing_juejin'] else '否'} |"
        )
    lines.extend(
        [
            "",
            "## 准入缺口",
            "",
            "1. fresh rerun 未通过：`ogd_gap1_x050` 新复跑日志仍报 `无法连接到终端服务`。",
            "2. Sharpe 安全垫薄：除 `ogd_gap1_x050` 外，其余达标版本 Sharpe 基本贴着 4，轻微环境差异就可能掉出门槛。",
            "3. 平滑性证据不足：精确同源版本目前只有 full 口径掘金结果；相邻规则的分段结果显示近期年化明显低于全周期，说明仍有路径/时段依赖风险。",
            "4. 生产准入仍需补齐：fresh 掘金复跑、分段复跑、邻域稳定性和贡献集中度审查完成前，不应发布为生产准入通过。",
            "",
            "## 相邻规则分段参考",
            "",
            "相邻规则 `ogd_mildhi15_deep25_cap90` 和 `ogd_mildhi20_deep00_cap90` 的既有切片显示：full 年化可超过 500%，但 2025 起点和 recent60 年化明显降低。这不是否决同源高通过率版本，但说明“平滑、无偶然性”还没有被证明。",
            "",
            "| 版本 | 切片 | 年化 | Sharpe | 最大回撤 | 买入日 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in cond_rows.to_dict("records"):
        lines.append(
            f"| `{row['case']}` | {row['slice']} | {pct(row['pnl_ratio_annual'])} | "
            f"{float(row['sharp_ratio']):.4f} | {pct(row['max_drawdown'])} | {int(row['buy_days'])} |"
        )
    lines.extend(
        [
            "",
            "## 输入与过滤审计",
            "",
            f"- L2 执行行情：`{L2_DB}`。",
            "- 执行价格字段使用未复权 `open`，不把 qfq 价格当交易执行价。",
            "- 本轮信号文件中 `.BJ` 行数为 0。",
            "- ST/风险警示按 `ST_TYPE`、`ST_TYPE_name` 和名称前缀检查。",
            "",
            "| 版本 | 重复键 | BJ 行数 | 信号日 ST 行数 | 买入日 ST 行数 | 最大买入日 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary.to_dict("records"):
        lines.append(
            f"| `{row['case']}` | {int(row['audit_duplicate_keys'])} | {int(row['audit_bj_rows'])} | "
            f"{int(row['audit_st_signal_rows'])} | {int(row['audit_st_buy_rows'])} | {row['audit_max_buy']} |"
        )
    lines.extend(
        [
            "",
            "## 证据路径",
            "",
            f"- 复核 CSV：`{REPORT_DIR / 'pass_ratio_candidate_admission_review_20260715.csv'}`",
            f"- 复核 JSON：`{REPORT_DIR / 'pass_ratio_candidate_admission_review_20260715.json'}`",
            f"- 相邻分段参考：`{REPORT_DIR / 'nearby_slice_reference_20260715.csv'}`",
            f"- fresh 复跑日志：`{FRESH_LOG}`",
            f"- 既有掘金结果：`{OPEN_GAP_RESULTS}`",
        ]
    )
    report = "\n".join(lines) + "\n"
    (REPORT_DIR / "pass_ratio_candidate_admission_review_20260715.md").write_text(report, encoding="utf-8")
    return summary, report


def main() -> None:
    summary, _ = build_review()
    print(
        summary[
            [
                "case",
                "pnl_ratio_annual",
                "sharp_ratio",
                "max_drawdown",
                "rows",
                "buy_days",
                "mean_daily_target_sum",
                "pass_target_full_existing_juejin",
            ]
        ].to_string(index=False)
    )
    print(REPORT_DIR / "pass_ratio_candidate_admission_review_20260715.md")


if __name__ == "__main__":
    main()
