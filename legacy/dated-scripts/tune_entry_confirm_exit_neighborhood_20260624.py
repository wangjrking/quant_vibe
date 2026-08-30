from __future__ import annotations

import json
from pathlib import Path

import tune_formal_horizon_entry_confirmation_20260624 as entry
import tune_strict_sync_liquidity_refill_20260624 as liq


REPORT_DIR = entry.SOURCE_DIR / "formal_horizon_entry_confirmation_20260624" / "entry_confirm_exit_neighborhood_20260624"

WEIGHT = {"weight_name": "w78_5d12_3d10", "w10d": 0.78, "w5d": 0.12, "w3d": 0.10, "w1d": 0.00}
EXIT_RATIOS = [0.94, 0.95, 0.96, 0.97, 0.98]
HOLDING_DAYS = [2, 3, 4]


def _case(exit_ratio: float, holding_days: int) -> dict:
    case = dict(WEIGHT)
    case["holding_days"] = holding_days
    case["exit_ratio_override"] = exit_ratio
    return case


def _f(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _write_report(summary: list[dict], audit: dict) -> None:
    ranked = sorted(summary, key=lambda row: (_f(row.get("late_min_annual")), _f(row.get("full_annual"))), reverse=True)
    annual_ranked = sorted(summary, key=lambda row: _f(row.get("full_annual")), reverse=True)
    lines = [
        "# entry_confirm 退出邻域验证",
        "",
        "## 当前结论",
        "",
        "本实验固定入场权重 `0.78 * rank_10d + 0.12 * rank_5d + 0.10 * rank_3d`，只测试 `h2/h3/h4` 与退出比例 `0.94-0.98` 的邻域。卖出仍沿用当前 10D 核心 score table，不使用日期、月份、行业过滤。",
        "",
        "## 按近期开仓最差年化排序",
        "",
        "| 候选 | h | 退出比例 | 年化 | Sharpe | 最大回撤 | 近期开仓最差年化 | 近期开仓中位年化 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ranked[:15]:
        lines.append(
            f"| {row['name']} | {row['holding_days']} | {float(row['exit_ratio']):.2f} | "
            f"{_f(row['full_annual']):.2%} | {_f(row['full_sharpe']):.2f} | {_f(row['full_max_drawdown']):.2%} | "
            f"{_f(row['late_min_annual']):.2%} | {_f(row['late_median_annual']):.2%} |"
        )
    lines.extend([
        "",
        "## 按全周期年化排序",
        "",
        "| 候选 | h | 退出比例 | 年化 | Sharpe | 最大回撤 | 近期开仓最差年化 | 近期开仓中位年化 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in annual_ranked[:15]:
        lines.append(
            f"| {row['name']} | {row['holding_days']} | {float(row['exit_ratio']):.2f} | "
            f"{_f(row['full_annual']):.2%} | {_f(row['full_sharpe']):.2f} | {_f(row['full_max_drawdown']):.2%} | "
            f"{_f(row['late_min_annual']):.2%} | {_f(row['late_median_annual']):.2%} |"
        )
    lines.extend([
        "",
        "## 硬过滤审计",
        "",
        f"- 审计文件数：`{audit.get('audited_files')}`",
        f"- 失败文件数：`{audit.get('failed_files')}`",
        f"- 买入日行情缺失：`{audit.get('buy_join_missing')}`",
        f"- 北交所：`{audit.get('bj_rows')}`",
        f"- ST / 风险警示：`{audit.get('buy_st_rows')}`",
        f"- 退市：`{audit.get('buy_delist_rows')}`",
        f"- 开盘涨停：`{audit.get('open_limit_up_buy_rows')}`",
        "",
        "## 证据路径",
        "",
        f"- 明细：`{REPORT_DIR / 'detail.csv'}`",
        f"- 汇总：`{REPORT_DIR / 'summary.csv'}`",
        f"- 硬过滤审计：`{REPORT_DIR / 'hard_gate_audit.json'}`",
        f"- 信号文件：`{REPORT_DIR / 'signals'}`",
        f"- 掘金日志：`{REPORT_DIR / 'logs'}`",
    ])
    (REPORT_DIR / "exit_neighborhood_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    old_report_dir = entry.REPORT_DIR
    old_exit_ratio = entry.EXIT_RATIO
    try:
        entry.REPORT_DIR = REPORT_DIR
        base = liq._load_base()
        market = liq._market_rows()
        next_date = liq._date_map(base)
        detail: list[dict] = []
        signal_files: list[Path] = []
        signal_stats: dict[str, dict] = {}
        for exit_ratio in EXIT_RATIOS:
            entry.EXIT_RATIO = exit_ratio
            for holding_days in HOLDING_DAYS:
                case = _case(exit_ratio, holding_days)
                signal_file, rows = entry._build_signal(base, market, next_date, case)
                signal_files.append(signal_file)
                name = entry._case_name(case)
                signal_stats[name] = {
                    "signal_rows": len(rows),
                    "signal_days": len({row["signal_date"] for row in rows}),
                    "latest_signal_date": max((row["signal_date"] for row in rows), default=None),
                }
                for start_name, start in entry.STARTS:
                    row = entry._run_case(case, signal_file, start_name, start)
                    row["exit_ratio"] = exit_ratio
                    detail.append(row)
                    print(f"{name} {start_name} annual={row['annual']} sharpe={row['sharpe']}", flush=True)
        summary = entry._summarize(detail, signal_stats)
        for row in summary:
            parts = row["name"].split("_e")
            row["exit_ratio"] = float(parts[-1].split("_", 1)[0]) / 100 if len(parts) > 1 else None
        entry._write_rows(REPORT_DIR / "detail.csv", detail)
        entry._write_rows(REPORT_DIR / "summary.csv", summary)
        entry._write_rows(REPORT_DIR / "summary_by_late_min.csv", sorted(summary, key=lambda row: _f(row["late_min_annual"]), reverse=True))
        entry._write_rows(REPORT_DIR / "summary_by_annual.csv", sorted(summary, key=lambda row: _f(row["full_annual"]), reverse=True))
        audit = liq._audit_signals(signal_files)
        (REPORT_DIR / "hard_gate_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_report(summary, audit)
        return 0
    finally:
        entry.REPORT_DIR = old_report_dir
        entry.EXIT_RATIO = old_exit_ratio


if __name__ == "__main__":
    raise SystemExit(main())
