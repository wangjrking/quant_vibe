from __future__ import annotations

import json
from pathlib import Path

import validate_dynamic_h2_m3_sl06_dd0509_pos86_tp060_20260624 as core


CASE_NAME = "dynamic_h2_m3_c0975_w78_5d12_3d10_pos90_e097_mh1_sl06_dd07115_scale7454"
REPORT_DIR = (
    core.SOURCE_DIR
    / "formal_horizon_entry_confirmation_20260624"
    / "dynamic_h2_m3_sl06_dd07115_pos90_scale7454_validation_20260625"
)
SIGNAL_FILE = (
    core.SOURCE_DIR
    / "formal_horizon_entry_confirmation_20260624"
    / "dynamic_h2_m3_sl06_dd0712_pos89_c975_high_pos_boundary_20260625"
    / "signals"
    / "pos90_scale7454.csv"
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
    audit = data["audit"]
    full = next((row for row in time_rows if row.get("slice") == "full"), None) or time_rows[0]
    late_min = min((_f(row.get("annual_min")) for row in nearby_summary), default=float("nan"))
    lines = [
        "# pos90_scale7454 + dh115 完整验证报告",
        "",
        "## 当前结论",
        "",
        f"- 验证对象：`{CASE_NAME}`",
        "- 信号文件沿用当前正式链路同一选股逻辑，只把仓位字段切到 `pos90_scale7454`，执行风控改为 `dd_soft=0.07 / dd_hard=0.115 / soft_scale=0.74 / hard_scale=0.54`。",
        "- 不使用行业限制、月份排除、日期排除、最新状态回填历史样本；不接入 1D 模型；仍保持北交所/ST/退市/涨停买入硬约束。",
        f"- 全周期年化：`{_f(full.get('annual')):.2%}`，Sharpe：`{_f(full.get('sharpe')):.2f}`，最大回撤：`{_f(full.get('max_drawdown')):.2%}`。",
        f"- 近端启动最低年化：`{late_min:.2%}`。",
        "",
        "## 时间切片",
        "",
        "| 切片 | 年化 | Sharpe | 最大回撤 | 开仓数 | 平均持仓率 |",
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
            "| 锚点 | 起点数 | 年化最低 | 年化中位 | 年化最高 | 最大回撤最高 |",
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
            "## 压力验证",
            "",
            "| 变体 | 原因 | 全周期年化 | Sharpe | 最大回撤 | 近端最低年化 |",
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
            "## 硬约束审计",
            "",
            f"- 审计文件数：`{audit['audited_files']}`",
            f"- 失败文件数：`{audit['failed_files']}`",
            f"- 买入日行情缺失：`{audit['buy_join_missing']}`",
            f"- 北交所命中：`{audit['bj_rows']}`",
            f"- 买入日 ST/风险警示：`{audit['buy_st_rows']}`",
            f"- 买入日退市：`{audit['buy_delist_rows']}`",
            f"- 买入日开盘涨停：`{audit['open_limit_up_buy_rows']}`",
        ]
    )
    (REPORT_DIR / "validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    target = core
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
        target.TARGET_PCT = 0.90
        target.INTRADAY_STOP_LOSS = 0.06
        target.TAKE_PROFIT = 0.07
        target.DD_SOFT = 0.07
        target.DD_HARD = 0.115
        target.DD_RECOVER = 0.03
        target.DD_SOFT_SCALE = 0.74
        target.DD_HARD_SCALE = 0.54
        target.CONTINUE_RATIO = 0.975
        code = target.main()
        _rewrite_report()
        return code
    finally:
        for key, value in old_values.items():
            setattr(target, key, value)


if __name__ == "__main__":
    raise SystemExit(main())
