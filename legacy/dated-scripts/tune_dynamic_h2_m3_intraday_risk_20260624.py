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


SOURCE_DIR = liq.SOURCE_DIR / "strict_sync_liquidity_neighborhood_20260624"
BASE_DIR = SOURCE_DIR / "formal_horizon_entry_confirmation_20260624"
REPORT_DIR = BASE_DIR / "dynamic_h2_m3_intraday_risk_20260624"
SIGNAL_FILE = BASE_DIR / "dynamic_hold_20260624" / "signals" / "dynamic_h2_m3_c098_w78_5d12_3d10_pos56_e097_mh1.csv"

TARGET_PCT = 0.56
HOLDING_DAYS = 2
MAX_HOLDING_DAYS = 3
EXIT_RATIO = 0.97
MIN_HOLD = 1
CONTINUE_RATIO = 0.98

STARTS = [
    ("full_20240605", "2024-06-05 09:00:00", liq.BACKTEST_END),
    ("recent60_20260324", "2026-03-24 09:00:00", liq.BACKTEST_END),
    ("late_20250701", "2025-07-01 09:00:00", liq.BACKTEST_END),
    ("late_20251009", "2025-10-09 09:00:00", liq.BACKTEST_END),
    ("late_20260105", "2026-01-05 09:00:00", liq.BACKTEST_END),
]

CASES = [
    {"name": "base_no_intraday", "intraday": False, "stop_loss": None, "take_profit": None, "risk_times": None},
    {"name": "idrisk_sl04", "intraday": True, "stop_loss": 0.04, "take_profit": None, "risk_times": "10:00:00,11:00:00,14:30:00"},
    {"name": "idrisk_sl06", "intraday": True, "stop_loss": 0.06, "take_profit": None, "risk_times": "10:00:00,11:00:00,14:30:00"},
    {"name": "idrisk_sl08", "intraday": True, "stop_loss": 0.08, "take_profit": None, "risk_times": "10:00:00,11:00:00,14:30:00"},
    {"name": "idrisk_sl06_tp08", "intraday": True, "stop_loss": 0.06, "take_profit": 0.08, "risk_times": "10:00:00,11:00:00,14:30:00"},
    {"name": "idrisk_sl06_tp10", "intraday": True, "stop_loss": 0.06, "take_profit": 0.10, "risk_times": "10:00:00,11:00:00,14:30:00"},
    {"name": "idrisk_sl08_tp12", "intraday": True, "stop_loss": 0.08, "take_profit": 0.12, "risk_times": "10:00:00,11:00:00,14:30:00"},
    {"name": "idrisk_sl06_once1430", "intraday": True, "stop_loss": 0.06, "take_profit": None, "risk_times": "14:30:00"},
    {
        "name": "idrisk_sl06_dd0814",
        "intraday": True,
        "stop_loss": 0.06,
        "take_profit": None,
        "risk_times": "10:00:00,11:00:00,14:30:00",
        "equity_dd": {"soft": 0.08, "hard": 0.14, "recover": 0.04, "soft_scale": 0.85, "hard_scale": 0.65},
    },
    {
        "name": "idrisk_sl06_dd0612",
        "intraday": True,
        "stop_loss": 0.06,
        "take_profit": None,
        "risk_times": "10:00:00,11:00:00,14:30:00",
        "equity_dd": {"soft": 0.06, "hard": 0.12, "recover": 0.03, "soft_scale": 0.80, "hard_scale": 0.60},
    },
    {
        "name": "idrisk_sl06_dd0510",
        "intraday": True,
        "stop_loss": 0.06,
        "take_profit": None,
        "risk_times": "10:00:00,11:00:00,14:30:00",
        "equity_dd": {"soft": 0.05, "hard": 0.10, "recover": 0.03, "soft_scale": 0.80, "hard_scale": 0.60},
    },
    {
        "name": "idrisk_sl06_dd0713",
        "intraday": True,
        "stop_loss": 0.06,
        "take_profit": None,
        "risk_times": "10:00:00,11:00:00,14:30:00",
        "equity_dd": {"soft": 0.07, "hard": 0.13, "recover": 0.04, "soft_scale": 0.85, "hard_scale": 0.65},
    },
]


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


def _case_args(case: dict) -> list[str]:
    args: list[str] = []
    if case.get("stop_loss") is not None:
        args.extend(["--stop-loss-pct", str(case["stop_loss"])])
    if case.get("take_profit") is not None:
        args.extend(["--take-profit-pct", str(case["take_profit"])])
    return args


def _run(case: dict, start_name: str, start: str, end: str) -> dict:
    log_file = REPORT_DIR / "logs" / f"{case['name']}_{start_name}.log"
    indicator = _extract_indicator(log_file)
    returncode = 0
    if indicator is None:
        env = os.environ.copy()
        env.update(liq.BASE_ENV)
        env["GM_FORCE_SELL_MARKET_ORDER"] = "1"
        env["GM_SCORE_EXIT_ENTRY_RATIO"] = str(EXIT_RATIO)
        env["GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT"] = str(MIN_HOLD)
        env["GM_SCORE_CONTINUE_ENTRY_RATIO"] = str(CONTINUE_RATIO)
        env["GM_INTRADAY_RISK_MODE"] = "1" if case.get("intraday") else "0"
        env["GM_INTRADAY_REPLACE_BUY"] = "0"
        if case.get("risk_times"):
            env["GM_INTRADAY_RISK_TIMES"] = str(case["risk_times"])
        if case.get("equity_dd"):
            equity_dd = case["equity_dd"]
            env["GM_EQUITY_DD_RISK_MODE"] = "1"
            env["GM_EQUITY_DD_SOFT_TRIGGER"] = str(equity_dd["soft"])
            env["GM_EQUITY_DD_HARD_TRIGGER"] = str(equity_dd["hard"])
            env["GM_EQUITY_DD_RECOVER_TRIGGER"] = str(equity_dd["recover"])
            env["GM_EQUITY_DD_SOFT_SCALE"] = str(equity_dd["soft_scale"])
            env["GM_EQUITY_DD_HARD_SCALE"] = str(equity_dd["hard_scale"])
            env["GM_EQUITY_DD_RESIZE_EXISTING"] = "0"
        command = [
            str(liq.JUEJIN_PYTHON),
            str(liq.MAIN / "run_juejin_signal_backtest.py"),
            "--strategy-dir",
            str(liq.STRATEGY_DIR),
            "--signal-file",
            str(SIGNAL_FILE),
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
            *_case_args(case),
        ]
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w", encoding="utf-8") as log:
            proc = subprocess.run(command, cwd=str(liq.MAIN), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        returncode = proc.returncode
        indicator = _extract_indicator(log_file)
    return {
        "name": case["name"],
        "intraday": case.get("intraday"),
        "stop_loss": case.get("stop_loss"),
        "take_profit": case.get("take_profit"),
        "risk_times": case.get("risk_times"),
        "equity_dd": json.dumps(case.get("equity_dd"), ensure_ascii=False, sort_keys=True) if case.get("equity_dd") else "",
        "start_name": start_name,
        "start": start,
        "returncode": returncode,
        "annual": indicator.get("pnl_ratio_annual") if indicator else None,
        "pnl_ratio": indicator.get("pnl_ratio") if indicator else None,
        "sharpe": indicator.get("sharp_ratio") if indicator else None,
        "max_drawdown": indicator.get("max_drawdown") if indicator else None,
        "win_ratio": indicator.get("win_ratio") if indicator else None,
        "open_count": indicator.get("open_count") if indicator else None,
        "close_count": indicator.get("close_count") if indicator else None,
        "log_file": str(log_file),
        **_exposure_stats(log_file),
    }


def _summarize(detail: list[dict]) -> list[dict]:
    out = []
    for name in sorted({row["name"] for row in detail}):
        rows = [row for row in detail if row["name"] == name]
        by_start = {row["start_name"]: row for row in rows}
        late = [_f(by_start[key]["annual"]) for key in ["late_20250701", "late_20251009", "late_20260105"] if key in by_start]
        late = [value for value in late if value == value]
        first = rows[0]
        out.append(
            {
                "name": name,
                "intraday": first.get("intraday"),
                "stop_loss": first.get("stop_loss"),
                "take_profit": first.get("take_profit"),
                "risk_times": first.get("risk_times"),
                "equity_dd": first.get("equity_dd"),
                "full_annual": by_start.get("full_20240605", {}).get("annual"),
                "full_sharpe": by_start.get("full_20240605", {}).get("sharpe"),
                "full_max_drawdown": by_start.get("full_20240605", {}).get("max_drawdown"),
                "recent60_annual": by_start.get("recent60_20260324", {}).get("annual"),
                "recent60_sharpe": by_start.get("recent60_20260324", {}).get("sharpe"),
                "recent60_max_drawdown": by_start.get("recent60_20260324", {}).get("max_drawdown"),
                "late_min_annual": min(late) if late else None,
                "late_median_annual": float(pd.Series(late).median()) if late else None,
            }
        )
    return out


def _write_report(summary: list[dict], audit: dict) -> None:
    ranked = sorted(
        summary,
        key=lambda row: (_f(row["recent60_annual"]), _f(row["full_sharpe"]), -_f(row["full_max_drawdown"])),
        reverse=True,
    )
    lines = [
        "# 动态持有日内风控实验",
        "",
        "## 当前结论",
        "",
        "本实验固定 dynamic h2_m3_c098 信号和入场规则，只测试掘金可执行的日内浮亏/浮盈退出参数。该规则使用持仓期间实时可见的账户持仓收益，不使用未来样本筛选。",
        "",
        "| 候选 | 止损 | 止盈 | 回撤缩放 | 全周期年化 | Sharpe | 最大回撤 | recent60 年化 | recent60 Sharpe | recent60 最大回撤 | 近期开仓最差年化 |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ranked:
        stop_loss_text = "" if row.get("stop_loss") in (None, "") else f"{float(row['stop_loss']):.2%}"
        take_profit_text = "" if row.get("take_profit") in (None, "") else f"{float(row['take_profit']):.2%}"
        lines.append(
            f"| {row['name']} | {stop_loss_text} | "
            f"{take_profit_text} | {row.get('equity_dd') or ''} | "
            f"{_f(row['full_annual']):.2%} | {_f(row['full_sharpe']):.2f} | {_f(row['full_max_drawdown']):.2%} | "
            f"{_f(row['recent60_annual']):.2%} | {_f(row['recent60_sharpe']):.2f} | {_f(row['recent60_max_drawdown']):.2%} | {_f(row['late_min_annual']):.2%} |"
        )
    lines.extend(
        [
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
            f"- 掘金日志：`{REPORT_DIR / 'logs'}`",
        ]
    )
    (REPORT_DIR / "intraday_risk_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    if not SIGNAL_FILE.exists():
        raise FileNotFoundError(SIGNAL_FILE)
    detail: list[dict] = []
    for case in CASES:
        for start_name, start, end in STARTS:
            row = _run(case, start_name, start, end)
            detail.append(row)
            print(f"{case['name']} {start_name} annual={row['annual']} sharpe={row['sharpe']} maxdd={row['max_drawdown']}", flush=True)
    summary = _summarize(detail)
    _write_rows(REPORT_DIR / "detail.csv", detail)
    _write_rows(REPORT_DIR / "summary.csv", summary)
    _write_rows(
        REPORT_DIR / "summary_by_recent60.csv",
        sorted(summary, key=lambda row: (_f(row["recent60_annual"]), _f(row["full_sharpe"]), -_f(row["full_max_drawdown"])), reverse=True),
    )
    audit = liq._audit_signals([SIGNAL_FILE])
    (REPORT_DIR / "hard_gate_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(summary, audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
