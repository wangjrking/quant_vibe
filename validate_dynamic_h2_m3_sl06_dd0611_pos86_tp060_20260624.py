from __future__ import annotations

import json
from pathlib import Path

import validate_dynamic_h2_m3_sl06_dd0509_pos86_tp060_20260624 as val


CASE_NAME = "dynamic_h2_m3_c098_w78_5d12_3d10_pos86_e097_mh1_sl06_dd0611_tp060"
REPORT_DIR = (
    val.SOURCE_DIR
    / "formal_horizon_entry_confirmation_20260624"
    / "dynamic_h2_m3_sl06_dd0611_pos86_tp060_validation_20260624"
)
SIGNAL_FILE = (
    val.SOURCE_DIR
    / "formal_horizon_entry_confirmation_20260624"
    / "dynamic_h2_m3_pos86_tp060_dd_midpoint_20260624"
    / "signals"
    / "tp060_pos86_dd0611_s7050.csv"
)


def _f(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _rewrite_report() -> None:
    summary_path = REPORT_DIR / "validation_summary.json"
    if not summary_path.exists():
        return
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    time_rows = data["time_slices"]
    nearby_summary = data["nearby_summary"]
    stress_summary = data["stress_summary"]
    proxy_stats = data["proxy"]
    audit = data["audit"]

    lines = [
        "# sl06_dd0611_pos86_tp060 完整验证",
        "",
        "## 当前结论",
        "",
        f"验证对象：`{CASE_NAME}`。该版本固定 formal L4 多周期模型入场、Top1 单票、目标仓位 `86%`、动态持有 `h2_m3_c098`、分数退出 `0.97`、日内 `6%` 止损和通用 `6%` 止盈；账户回撤缩放为 `soft=6% / hard=11% / recover=3% / scale=0.70/0.50`。",
        "",
        "本报告用于判断回撤中间档是否比 `dd0509_s7050` 更适合作为强准入候选。收益指标只作为弱准入排序；持续性、低路径依赖、贡献压力、硬过滤和掘金复现是强准入依据。",
        "",
        "## 时间切片",
        "",
        "| 切片 | 年化 | Sharpe | 最大回撤 | 开仓数 | 平均仓位 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in time_rows:
        lines.append(
            f"| {row['slice']} | {_f(row['annual']):.2%} | {_f(row['sharpe']):.2f} | "
            f"{_f(row['max_drawdown']):.2%} | {row['open_count']} | {_f(row['avg_invested_pct']):.2%} |"
        )
    lines.extend(
        [
            "",
            "## 低路径依赖",
            "",
            "| 锚点 | 起点数 | 年化最小 | 年化中位 | 年化最大 | 最大回撤最大 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in nearby_summary:
        lines.append(
            f"| {row['anchor']} | {row['starts']} | {_f(row['annual_min']):.2%} | "
            f"{_f(row['annual_median']):.2%} | {_f(row['annual_max']):.2%} | "
            f"{_f(row['max_drawdown_max']):.2%} |"
        )
    lines.extend(
        [
            "",
            "## 贡献集中压力",
            "",
            f"- 最高代理贡献股票：`{proxy_stats['top_stock']}`，占正代理贡献 `{_f(proxy_stats['top_stock_share']):.2%}`。",
            f"- 最高代理贡献日：`{proxy_stats['top_day']}`，占正代理贡献 `{_f(proxy_stats['top_day_share']):.2%}`。",
            f"- 最高代理贡献月份：`{proxy_stats['top_month']}`，占正代理贡献 `{_f(proxy_stats['top_month_share']):.2%}`。",
            "",
            "| 压力变体 | 说明 | 年化 | Sharpe | 最大回撤 | 近期开仓最差年化 |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in stress_summary:
        lines.append(
            f"| {row['variant']} | {row['reason']} | {_f(row['full_annual']):.2%} | "
            f"{_f(row['full_sharpe']):.2f} | {_f(row['full_max_drawdown']):.2%} | "
            f"{_f(row['late_min_annual']):.2%} |"
        )
    lines.extend(
        [
            "",
            "## 硬过滤审计",
            "",
            f"- 审计文件数：`{audit['audited_files']}`",
            f"- 失败文件数：`{audit['failed_files']}`",
            f"- 买入日行情缺失：`{audit['buy_join_missing']}`",
            f"- 北交所命中：`{audit['bj_rows']}`",
            f"- 买入日 ST / 风险警示：`{audit['buy_st_rows']}`",
            f"- 买入日退市：`{audit['buy_delist_rows']}`",
            f"- 买入日开盘涨停：`{audit['open_limit_up_buy_rows']}`",
            "",
            "## 证据路径",
            "",
            f"- 时间切片：`{REPORT_DIR / 'time_slices.csv'}`",
            f"- 低路径依赖：`{REPORT_DIR / 'nearby_summary.csv'}`",
            f"- 贡献压力：`{REPORT_DIR / 'stress_summary.csv'}`",
            f"- 硬过滤审计：`{REPORT_DIR / 'hard_gate_audit.json'}`",
            f"- 掘金日志：`{REPORT_DIR / 'logs'}`",
        ]
    )
    (REPORT_DIR / "validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    old_values = {
        "CASE_NAME": val.CASE_NAME,
        "REPORT_DIR": val.REPORT_DIR,
        "SIGNAL_FILE": val.SIGNAL_FILE,
        "INTRADAY_STOP_LOSS": val.INTRADAY_STOP_LOSS,
        "TAKE_PROFIT": val.TAKE_PROFIT,
        "DD_SOFT": val.DD_SOFT,
        "DD_HARD": val.DD_HARD,
        "DD_RECOVER": val.DD_RECOVER,
        "DD_SOFT_SCALE": val.DD_SOFT_SCALE,
        "DD_HARD_SCALE": val.DD_HARD_SCALE,
    }
    try:
        val.CASE_NAME = CASE_NAME
        val.REPORT_DIR = REPORT_DIR
        val.SIGNAL_FILE = SIGNAL_FILE
        val.INTRADAY_STOP_LOSS = 0.06
        val.TAKE_PROFIT = 0.06
        val.DD_SOFT = 0.06
        val.DD_HARD = 0.11
        val.DD_RECOVER = 0.03
        val.DD_SOFT_SCALE = 0.70
        val.DD_HARD_SCALE = 0.50
        code = val.main()
        _rewrite_report()
        return code
    finally:
        for key, value in old_values.items():
            setattr(val, key, value)


if __name__ == "__main__":
    raise SystemExit(main())
