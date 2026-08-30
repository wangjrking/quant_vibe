from __future__ import annotations

import csv
import json
from pathlib import Path

import validate_force_sell_pos56_candidate_20260624 as validator


CASE_NAME = "entry_confirm_w78_5d12_3d10_pos56_h3_e096_mh1"
REPORT_DIR = (
    validator.SOURCE_DIR
    / "formal_horizon_entry_confirmation_20260624"
    / "entry_confirm_w78_5d12_3d10_pos56_h3_validation_20260624"
)
SIGNAL_FILE = (
    validator.SOURCE_DIR
    / "formal_horizon_entry_confirmation_20260624"
    / "signals"
    / f"{CASE_NAME}.csv"
)


def _read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _f(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _write_clean_report() -> None:
    time_rows = _read_rows(REPORT_DIR / "time_slices.csv")
    nearby_rows = _read_rows(REPORT_DIR / "nearby_summary.csv")
    stress_rows = _read_rows(REPORT_DIR / "stress_summary.csv")
    audit = json.loads((REPORT_DIR / "hard_gate_audit.json").read_text(encoding="utf-8"))
    proxy = json.loads((REPORT_DIR / "proxy_summary.json").read_text(encoding="utf-8"))

    lines = [
        "# formal 3D 入场确认最佳候选验证报告",
        "",
        "## 当前结论",
        "",
        f"验证对象：`{CASE_NAME}`。",
        "",
        "该版本在不使用日期、月份、行业限制的前提下，将 3D formal 分数作为入场确认项，入场排序为 `0.78 * rank_10d + 0.12 * rank_5d + 0.10 * rank_3d`；卖出仍沿用当前 10D 核心 score table、`h3`、目标仓位 `56%`、强制市价卖出。",
        "",
        "它相对前一版 pos56 基准提升了全周期年化和 Sharpe，并改善了近期开仓最差年化。但该版本仍需要继续看 recent60、h2/h4 邻域和贡献压力，不能仅凭全周期收益直接判定生产准入通过。",
        "",
        "## 时间切片",
        "",
        "| 切片 | 年化 | Sharpe | 最大回撤 | 开仓数 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in time_rows:
        lines.append(
            f"| {row['slice']} | {_f(row['annual']):.2%} | {_f(row['sharpe']):.2f} | {_f(row['max_drawdown']):.2%} | {row['open_count']} |"
        )

    lines.extend([
        "",
        "## 低路径依赖",
        "",
        "| 锚点 | 起点数 | 年化最小 | 年化中位 | 年化最大 | 最大回撤最大 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in nearby_rows:
        lines.append(
            f"| {row['anchor']} | {row['starts']} | {_f(row['annual_min']):.2%} | {_f(row['annual_median']):.2%} | "
            f"{_f(row['annual_max']):.2%} | {_f(row['max_drawdown_max']):.2%} |"
        )

    lines.extend([
        "",
        "## 贡献集中压力",
        "",
        f"- 最高代理股票：`{proxy['top_stock']}`，占正代理贡献 `{_f(proxy['top_stock_share']):.2%}`。",
        f"- 最高代理日：`{proxy['top_day']}`，占正代理贡献 `{_f(proxy['top_day_share']):.2%}`。",
        f"- 最高代理月份：`{proxy['top_month']}`，占正代理贡献 `{_f(proxy['top_month_share']):.2%}`。",
        "",
        "| 压力变体 | 说明 | 年化 | Sharpe | 最大回撤 | 近期开仓最差年化 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ])
    for row in stress_rows:
        lines.append(
            f"| {row['variant']} | {row['reason']} | {_f(row['full_annual']):.2%} | {_f(row['full_sharpe']):.2f} | "
            f"{_f(row['full_max_drawdown']):.2%} | {_f(row['late_min_annual']):.2%} |"
        )

    lines.extend([
        "",
        "## 硬过滤审计",
        "",
        f"- 审计文件数：`{audit['audited_files']}`",
        f"- 失败文件数：`{audit['failed_files']}`",
        f"- 买入日行情缺失：`{audit['buy_join_missing']}`",
        f"- 北交所：`{audit['bj_rows']}`",
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
        f"- 信号文件：`{REPORT_DIR / 'signals'}`",
    ])
    (REPORT_DIR / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    validator.CASE_NAME = CASE_NAME
    validator.REPORT_DIR = REPORT_DIR
    validator.SIGNAL_FILE = SIGNAL_FILE
    result = validator.main()
    _write_clean_report()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
