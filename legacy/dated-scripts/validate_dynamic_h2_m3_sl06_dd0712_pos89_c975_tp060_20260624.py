from __future__ import annotations

import json
from pathlib import Path

import validate_dynamic_h2_m3_sl06_dd0611_pos90_tp060_20260624 as val90


CASE_NAME = "dynamic_h2_m3_c0975_w78_5d12_3d10_pos89_e097_mh1_sl06_dd0712_tp060"
REPORT_DIR = (
    val90.val86.val.SOURCE_DIR
    / "formal_horizon_entry_confirmation_20260624"
    / "dynamic_h2_m3_sl06_dd0712_pos89_c975_tp060_validation_20260624"
)
SIGNAL_FILE = (
    val90.val86.val.SOURCE_DIR
    / "formal_horizon_entry_confirmation_20260624"
    / "dynamic_h2_m3_sl06_dd0712_pos89_rule_neighborhood_20260624"
    / "signals"
    / "continue_c975.csv"
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
    full = next((row for row in time_rows if row.get("slice") == "full"), None) or time_rows[0]
    late_min = min((_f(row.get("annual_min")) for row in nearby_summary), default=float("nan"))
    lines = [
        "# continue_c975 完整验证报告",
        "",
        "## 当前结论",
        "",
        f"- 验证对象：`{CASE_NAME}`。",
        "- 口径：沿用 `pos89/sl06/dd0712/tp060` 规则，只把连续持仓保留阈值从 `0.980` 调整为 `0.975`。",
        "- 该版本仍是研究验证入口，不修改生产策略、不更新 registry、不生成正式交易信号。",
        f"- 全周期年化：`{_f(full.get('annual')):.2%}`；Sharpe：`{_f(full.get('sharpe')):.2f}`；最大回撤：`{_f(full.get('max_drawdown')):.2%}`。",
        f"- 近期开仓最低年化：`{late_min:.2%}`。",
        "",
        "收益指标属于弱准入排序项；是否能进入生产候选仍取决于硬过滤、参数邻域、时间切片、低路径依赖、贡献压力、掘金复现和归档完整性。风格暴露和风格漂移只作为风险提示披露，不单独阻断准入。",
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
            "## 贡献压力",
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
    target = val90.val86.val
    old_values = {
        "CASE_NAME": target.CASE_NAME,
        "REPORT_DIR": target.REPORT_DIR,
        "SIGNAL_FILE": target.SIGNAL_FILE,
        "TARGET_PCT": target.TARGET_PCT,
        "INTRADAY_STOP_LOSS": target.INTRADAY_STOP_LOSS,
        "TAKE_PROFIT": target.TAKE_PROFIT,
        "DD_SOFT": target.DD_SOFT,
        "DD_HARD": target.DD_HARD,
        "DD_RECOVER": target.DD_RECOVER,
        "DD_SOFT_SCALE": target.DD_SOFT_SCALE,
        "DD_HARD_SCALE": target.DD_HARD_SCALE,
        "CONTINUE_RATIO": target.CONTINUE_RATIO,
    }
    try:
        target.CASE_NAME = CASE_NAME
        target.REPORT_DIR = REPORT_DIR
        target.SIGNAL_FILE = SIGNAL_FILE
        target.TARGET_PCT = 0.89
        target.INTRADAY_STOP_LOSS = 0.06
        target.TAKE_PROFIT = 0.06
        target.DD_SOFT = 0.07
        target.DD_HARD = 0.12
        target.DD_RECOVER = 0.03
        target.DD_SOFT_SCALE = 0.70
        target.DD_HARD_SCALE = 0.50
        target.CONTINUE_RATIO = 0.975
        code = target.main()
        _rewrite_report()
        return code
    finally:
        for key, value in old_values.items():
            setattr(target, key, value)


if __name__ == "__main__":
    raise SystemExit(main())
