from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import os
import re
import subprocess
from pathlib import Path

import pandas as pd

import tune_strict_sync_liquidity_refill_20260624 as liq
import validate_force_sell_pos56_candidate_20260624 as validator


CASE_NAME = "dynamic_h2_m3_c098_w78_5d12_3d10_pos76_e097_mh1_sl06_dd0612"
SOURCE_DIR = liq.SOURCE_DIR / "strict_sync_liquidity_neighborhood_20260624"
REPORT_DIR = SOURCE_DIR / "formal_horizon_entry_confirmation_20260624" / "dynamic_h2_m3_sl06_dd0612_pos76_validation_20260624"
SIGNAL_FILE = (
    SOURCE_DIR
    / "formal_horizon_entry_confirmation_20260624"
    / "dynamic_h2_m3_sl06_dd0612_position_20260624"
    / "signals"
    / "sl06_dd0612_pos76.csv"
)

TARGET_PCT = 0.76
HOLDING_DAYS = 2
MAX_HOLDING_DAYS = 3
EXIT_RATIO = 0.97
MIN_HOLD = 1
CONTINUE_RATIO = 0.98


BASE_SIGNAL_FILE = (
    SOURCE_DIR
    / "formal_horizon_entry_confirmation_20260624"
    / "dynamic_hold_20260624"
    / "signals"
    / "dynamic_h2_m3_c098_w78_5d12_3d10_pos56_e097_mh1.csv"
)
INTRADAY_STOP_LOSS = 0.06
DD_SOFT = 0.06
DD_HARD = 0.12
DD_RECOVER = 0.03
DD_SOFT_SCALE = 0.80
DD_HARD_SCALE = 0.60


def _read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _write_rows(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fields} for row in rows])


def _f(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _extract_indicator(log_file: Path) -> dict | None:
    if not log_file.exists():
        return None
    for line in reversed(log_file.read_text(encoding="utf-8", errors="ignore").splitlines()):
        if "GM_BACKTEST_INDICATOR:" not in line:
            continue
        payload = line.split("GM_BACKTEST_INDICATOR:", 1)[1].strip()
        try:
            return ast.literal_eval(payload)
        except Exception:
            return eval(payload, {"__builtins__": {}}, {"datetime": datetime_module})
    return None


def _exposure_stats(log_file: Path) -> dict:
    values = []
    active = []
    pattern = re.compile(r"EXPOSURE .* invested_pct=([0-9.]+).*active_positions=(\d+)")
    if not log_file.exists():
        return {"avg_invested_pct": None, "ge80_ratio": None, "max_active_positions": None, "exposure_points": 0}
    for match in pattern.finditer(log_file.read_text(encoding="utf-8", errors="ignore")):
        values.append(float(match.group(1)))
        active.append(int(match.group(2)))
    if not values:
        return {"avg_invested_pct": None, "ge80_ratio": None, "max_active_positions": None, "exposure_points": 0}
    return {
        "avg_invested_pct": sum(values) / len(values),
        "ge80_ratio": sum(1 for value in values if value >= 0.80) / len(values),
        "max_active_positions": max(active) if active else None,
        "exposure_points": len(values),
    }


def _run(signal_file: Path, tag: str, start: str, end: str = liq.BACKTEST_END) -> dict:
    log_file = REPORT_DIR / "logs" / f"{tag}.log"
    indicator = _extract_indicator(log_file)
    returncode = 0
    if indicator is None:
        env = os.environ.copy()
        env.update(liq.BASE_ENV)
        env["GM_FORCE_SELL_MARKET_ORDER"] = "1"
        env["GM_SCORE_EXIT_ENTRY_RATIO"] = str(EXIT_RATIO)
        env["GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT"] = str(MIN_HOLD)
        env["GM_SCORE_CONTINUE_ENTRY_RATIO"] = str(CONTINUE_RATIO)
        env["GM_INTRADAY_RISK_MODE"] = "1"
        env["GM_INTRADAY_REPLACE_BUY"] = "0"
        env["GM_INTRADAY_RISK_TIMES"] = "10:00:00,11:00:00,14:30:00"
        env["GM_EQUITY_DD_RISK_MODE"] = "1"
        env["GM_EQUITY_DD_SOFT_TRIGGER"] = str(DD_SOFT)
        env["GM_EQUITY_DD_HARD_TRIGGER"] = str(DD_HARD)
        env["GM_EQUITY_DD_RECOVER_TRIGGER"] = str(DD_RECOVER)
        env["GM_EQUITY_DD_SOFT_SCALE"] = str(DD_SOFT_SCALE)
        env["GM_EQUITY_DD_HARD_SCALE"] = str(DD_HARD_SCALE)
        env["GM_EQUITY_DD_RESIZE_EXISTING"] = "0"
        command = [
            str(liq.JUEJIN_PYTHON),
            str(liq.MAIN / "run_juejin_signal_backtest.py"),
            "--strategy-dir",
            str(liq.STRATEGY_DIR),
            "--signal-file",
            str(signal_file),
            "--log-file",
            str(log_file),
            "--max-positions",
            "1",
            "--holding-days",
            str(HOLDING_DAYS),
            "--max-holding-days",
            str(MAX_HOLDING_DAYS),
            "--target-position-pct",
            str(TARGET_PCT),
            "--score-db",
            str(liq.SCORE_DB),
            "--score-table",
            liq.SCORE_TABLE,
            "--market-db",
            str(liq.MARKET_DB),
            "--backtest-start",
            start,
            "--backtest-end",
            end,
            "--backtest-adjust",
            "none",
            "--backtest-initial-cash",
            "600000",
            "--backtest-slippage-ratio",
            "0.0015",
            "--stop-loss-pct",
            str(INTRADAY_STOP_LOSS),
        ]
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w", encoding="utf-8") as log:
            proc = subprocess.run(command, cwd=str(liq.MAIN), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        returncode = proc.returncode
        indicator = _extract_indicator(log_file)
    return {
        "tag": tag,
        "start": start,
        "end": end,
        "returncode": returncode,
        "annual": indicator.get("pnl_ratio_annual") if indicator else None,
        "pnl_ratio": indicator.get("pnl_ratio") if indicator else None,
        "sharpe": indicator.get("sharp_ratio") if indicator else None,
        "max_drawdown": indicator.get("max_drawdown") if indicator else None,
        "win_ratio": indicator.get("win_ratio") if indicator else None,
        "open_count": indicator.get("open_count") if indicator else None,
        "close_count": indicator.get("close_count") if indicator else None,
        "signal_file": str(signal_file),
        "log_file": str(log_file),
        **_exposure_stats(log_file),
    }


def _summarize_nearby(rows: list[dict]) -> list[dict]:
    out = []
    for anchor in validator.ANCHORS:
        items = [row for row in rows if row["anchor"] == anchor and row.get("annual") is not None]
        annuals = [_f(row["annual"]) for row in items]
        sharpes = [_f(row["sharpe"]) for row in items if row.get("sharpe") is not None]
        drawdowns = [_f(row["max_drawdown"]) for row in items if row.get("max_drawdown") is not None]
        out.append(
            {
                "anchor": anchor,
                "starts": len(items),
                "annual_min": min(annuals) if annuals else None,
                "annual_median": float(pd.Series(annuals).median()) if annuals else None,
                "annual_max": max(annuals) if annuals else None,
                "sharpe_min": min(sharpes) if sharpes else None,
                "sharpe_median": float(pd.Series(sharpes).median()) if sharpes else None,
                "max_drawdown_max": max(drawdowns) if drawdowns else None,
            }
        )
    return out


def _stress_summary(rows: list[dict], stats: dict) -> list[dict]:
    out = []
    for variant in sorted({row["variant"] for row in rows}):
        items = [row for row in rows if row["variant"] == variant]
        full = next((row for row in items if row["start_name"] == "full_20240605"), None)
        late = [_f(row["annual"]) for row in items if row["start_name"].startswith("late_") and row.get("annual") is not None]
        out.append(
            {
                "variant": variant,
                "reason": stats["variant_reasons"].get(variant),
                "full_annual": _f(full.get("annual")) if full else None,
                "full_sharpe": _f(full.get("sharpe")) if full else None,
                "full_max_drawdown": _f(full.get("max_drawdown")) if full else None,
                "late_min_annual": min(late) if late else None,
            }
        )
    return out


def _write_report(time_rows: list[dict], nearby_summary: list[dict], stress_summary: list[dict], proxy_stats: dict, audit: dict) -> None:
    lines = [
        "# dynamic h2_m3_c098 + sl06_dd0612 + pos76 完整验证",
        "",
        "## 当前结论",
        "",
        f"验证对象：`{CASE_NAME}`。该版本在 dynamic h2_m3_c098 基础上增加日内 `6%` 止损、账户回撤缩放 `soft=6% / hard=12% / scale=0.80/0.60`，并将单票目标仓位提高到 `76%`。",
        "",
        "该规则不改变入场信号，只验证可实时执行的退出风控是否改善回撤、recent60 和低路径依赖。",
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
    for row in nearby_summary:
        lines.append(
            f"| {row['anchor']} | {row['starts']} | {_f(row['annual_min']):.2%} | {_f(row['annual_median']):.2%} | "
            f"{_f(row['annual_max']):.2%} | {_f(row['max_drawdown_max']):.2%} |"
        )
    lines.extend([
        "",
        "## 贡献集中压力",
        "",
        f"- 最高代理股票：`{proxy_stats['top_stock']}`，占正代理贡献 `{_f(proxy_stats['top_stock_share']):.2%}`。",
        f"- 最高代理日：`{proxy_stats['top_day']}`，占正代理贡献 `{_f(proxy_stats['top_day_share']):.2%}`。",
        f"- 最高代理月份：`{proxy_stats['top_month']}`，占正代理贡献 `{_f(proxy_stats['top_month_share']):.2%}`。",
        "",
        "| 压力变体 | 说明 | 年化 | Sharpe | 最大回撤 | 近期开仓最差年化 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ])
    for row in stress_summary:
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
    ])
    (REPORT_DIR / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    if not SIGNAL_FILE.exists():
        if not BASE_SIGNAL_FILE.exists():
            raise FileNotFoundError(BASE_SIGNAL_FILE)
        rows = _read_rows(BASE_SIGNAL_FILE)
        for row in rows:
            row["signal_stop_loss_pct"] = f"{INTRADAY_STOP_LOSS:.5f}"
            row["strategy_variant"] = CASE_NAME
        _write_rows(SIGNAL_FILE, rows)
    old_report = validator.REPORT_DIR
    old_holding = validator.HOLDING_DAYS
    old_target = validator.TARGET_PCT
    try:
        validator.REPORT_DIR = REPORT_DIR
        validator.HOLDING_DAYS = MAX_HOLDING_DAYS
        validator.TARGET_PCT = TARGET_PCT

        signal_rows = _read_rows(SIGNAL_FILE)
        time_rows = []
        for name, start, end in validator.TIME_SLICES:
            row = _run(SIGNAL_FILE, f"time_{name}", start, end)
            row["slice"] = name
            time_rows.append(row)
            print(f"time {name} annual={row['annual']} sharpe={row['sharpe']} maxdd={row['max_drawdown']}", flush=True)
        _write_rows(REPORT_DIR / "time_slices.csv", time_rows)

        nearby_rows = []
        for item in validator._nearby_starts():
            tag = f"nearby_{item['anchor']}_{item['offset']}_{item['start_date']}".replace("-", "m")
            row = _run(SIGNAL_FILE, tag, item["start_time"], liq.BACKTEST_END)
            row.update(item)
            nearby_rows.append(row)
            print(f"nearby {item['anchor']} {item['offset']} annual={row['annual']} sharpe={row['sharpe']}", flush=True)
        nearby_summary = _summarize_nearby(nearby_rows)
        _write_rows(REPORT_DIR / "nearby_cold_starts.csv", nearby_rows)
        _write_rows(REPORT_DIR / "nearby_summary.csv", nearby_summary)

        variants, proxy_stats = validator._build_stress_variants(signal_rows)
        stress_rows = []
        signal_files = [SIGNAL_FILE]
        for variant in variants:
            path = REPORT_DIR / "signals" / f"{variant['name']}.csv"
            _write_rows(path, variant["rows"])
            signal_files.append(path)
            for start_name, start in validator.MAIN_STARTS:
                row = _run(path, f"stress_{variant['name']}_{start_name}", start, liq.BACKTEST_END)
                row["variant"] = variant["name"]
                row["start_name"] = start_name
                stress_rows.append(row)
                print(f"stress {variant['name']} {start_name} annual={row['annual']} sharpe={row['sharpe']}", flush=True)
        stress_sum = _stress_summary(stress_rows, proxy_stats)
        _write_rows(REPORT_DIR / "stress_cases.csv", stress_rows)
        _write_rows(REPORT_DIR / "stress_summary.csv", stress_sum)
        (REPORT_DIR / "proxy_summary.json").write_text(json.dumps(proxy_stats, ensure_ascii=False, indent=2), encoding="utf-8")

        audit = liq._audit_signals(signal_files)
        (REPORT_DIR / "hard_gate_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        (REPORT_DIR / "validation_summary.json").write_text(
            json.dumps(
                {"time_slices": time_rows, "nearby_summary": nearby_summary, "stress_summary": stress_sum, "proxy": proxy_stats, "audit": audit},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        _write_report(time_rows, nearby_summary, stress_sum, proxy_stats, audit)
        return 0
    finally:
        validator.REPORT_DIR = old_report
        validator.HOLDING_DAYS = old_holding
        validator.TARGET_PCT = old_target


if __name__ == "__main__":
    raise SystemExit(main())
