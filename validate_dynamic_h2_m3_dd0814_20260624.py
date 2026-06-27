from __future__ import annotations

import json

import validate_dynamic_h2_m3_c098_20260624 as base
import validate_force_sell_pos56_candidate_20260624 as validator


CASE_NAME = "dynamic_h2_m3_c098_w78_5d12_3d10_pos56_e097_mh1_dd0814_s8565"
REPORT_DIR = (
    validator.SOURCE_DIR
    / "formal_horizon_entry_confirmation_20260624"
    / "dynamic_h2_m3_c098_dd0814_validation_20260624"
)
RISK_ENV = {
    "GM_EQUITY_DD_RISK_MODE": "1",
    "GM_EQUITY_DD_SOFT_TRIGGER": "0.08",
    "GM_EQUITY_DD_HARD_TRIGGER": "0.14",
    "GM_EQUITY_DD_RECOVER_TRIGGER": "0.04",
    "GM_EQUITY_DD_SOFT_SCALE": "0.85",
    "GM_EQUITY_DD_HARD_SCALE": "0.65",
    "GM_EQUITY_DD_RESIZE_EXISTING": "0",
}


def _run_with_risk(signal_file, tag: str, start: str, end: str = base.liq.BACKTEST_END) -> dict:
    original_run = base._run

    def patched_run(signal_file_inner, tag_inner: str, start_inner: str, end_inner: str = base.liq.BACKTEST_END) -> dict:
        import ast
        import datetime as datetime_module
        import os
        import subprocess

        log_file = REPORT_DIR / "logs" / f"{tag_inner}.log"
        indicator = base._extract_indicator(log_file)
        returncode = 0
        if indicator is None:
            env = os.environ.copy()
            env.update(base.liq.BASE_ENV)
            env.update(RISK_ENV)
            env["GM_FORCE_SELL_MARKET_ORDER"] = "1"
            env["GM_SCORE_EXIT_ENTRY_RATIO"] = str(base.EXIT_RATIO)
            env["GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT"] = str(base.MIN_HOLD)
            env["GM_SCORE_CONTINUE_ENTRY_RATIO"] = str(base.CONTINUE_RATIO)
            command = [
                str(base.liq.JUEJIN_PYTHON),
                str(base.liq.MAIN / "run_juejin_signal_backtest.py"),
                "--strategy-dir",
                str(base.liq.STRATEGY_DIR),
                "--signal-file",
                str(signal_file_inner),
                "--log-file",
                str(log_file),
                "--max-positions",
                "1",
                "--holding-days",
                str(base.HOLDING_DAYS),
                "--max-holding-days",
                str(base.MAX_HOLDING_DAYS),
                "--target-position-pct",
                str(base.TARGET_PCT),
                "--score-db",
                str(base.liq.SCORE_DB),
                "--score-table",
                base.liq.SCORE_TABLE,
                "--market-db",
                str(base.liq.MARKET_DB),
                "--backtest-start",
                start_inner,
                "--backtest-end",
                end_inner,
                "--backtest-adjust",
                "none",
                "--backtest-initial-cash",
                "600000",
                "--backtest-slippage-ratio",
                "0.0015",
            ]
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with log_file.open("w", encoding="utf-8") as log:
                proc = subprocess.run(command, cwd=str(base.liq.MAIN), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
            returncode = proc.returncode
            indicator = base._extract_indicator(log_file)
        return {
            "tag": tag_inner,
            "start": start_inner,
            "end": end_inner,
            "returncode": returncode,
            "annual": indicator.get("pnl_ratio_annual") if indicator else None,
            "pnl_ratio": indicator.get("pnl_ratio") if indicator else None,
            "sharpe": indicator.get("sharp_ratio") if indicator else None,
            "max_drawdown": indicator.get("max_drawdown") if indicator else None,
            "win_ratio": indicator.get("win_ratio") if indicator else None,
            "open_count": indicator.get("open_count") if indicator else None,
            "close_count": indicator.get("close_count") if indicator else None,
            "signal_file": str(signal_file_inner),
            "log_file": str(log_file),
            **base._exposure_stats(log_file),
        }

    try:
        base._run = patched_run
        return patched_run(signal_file, tag, start, end)
    finally:
        base._run = original_run


def _f(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _write_clean_report() -> None:
    time_rows = base._read_rows(REPORT_DIR / "time_slices.csv")
    nearby_rows = base._read_rows(REPORT_DIR / "nearby_summary.csv")
    stress_rows = base._read_rows(REPORT_DIR / "stress_summary.csv")
    audit = json.loads((REPORT_DIR / "hard_gate_audit.json").read_text(encoding="utf-8"))
    proxy = json.loads((REPORT_DIR / "proxy_summary.json").read_text(encoding="utf-8"))
    lines = [
        "# dynamic h2_m3_c098 加账户回撤缩放完整验证",
        "",
        "## 当前结论",
        "",
        f"验证对象：`{CASE_NAME}`。该版本使用动态持有 `h2_m3_c098`，并启用账户回撤缩放 `dd_08_14_s85_65`。",
        "",
        "它牺牲部分收益，将最大回撤降到约 31%。相对固定 h3，它更能解释持仓日敏感性；相对无风控动态 h2，它降低回撤但削弱近期开仓最差表现。",
        "",
        "## 时间切片",
        "",
        "| 切片 | 年化 | Sharpe | 最大回撤 | 开仓数 | 平均仓位 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in time_rows:
        lines.append(
            f"| {row['slice']} | {_f(row['annual']):.2%} | {_f(row['sharpe']):.2f} | {_f(row['max_drawdown']):.2%} | {row['open_count']} | {_f(row['avg_invested_pct']):.2%} |"
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
            f"| {row['anchor']} | {row['starts']} | {_f(row['annual_min']):.2%} | {_f(row['annual_median']):.2%} | {_f(row['annual_max']):.2%} | {_f(row['max_drawdown_max']):.2%} |"
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
            f"| {row['variant']} | {row['reason']} | {_f(row['full_annual']):.2%} | {_f(row['full_sharpe']):.2f} | {_f(row['full_max_drawdown']):.2%} | {_f(row['late_min_annual']):.2%} |"
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
    ])
    (REPORT_DIR / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    old_case = base.CASE_NAME
    old_report = base.REPORT_DIR
    try:
        base.CASE_NAME = CASE_NAME
        base.REPORT_DIR = REPORT_DIR
        base._run = _run_with_risk
        result = base.main()
        _write_clean_report()
        return result
    finally:
        base.CASE_NAME = old_case
        base.REPORT_DIR = old_report


if __name__ == "__main__":
    raise SystemExit(main())
