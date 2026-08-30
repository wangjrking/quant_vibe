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
REPORT_DIR = BASE_DIR / "dynamic_h2_m3_dd0714_tp060_high_position_20260624"
BASE_SIGNAL_FILE = BASE_DIR / "dynamic_h2_m3_tp060_pos84_dd_neighborhood_20260624" / "signals" / "tp060_pos84_dd0714_s8060.csv"

HOLDING_DAYS = 2
MAX_HOLDING_DAYS = 3
EXIT_RATIO = 0.97
MIN_HOLD = 1
CONTINUE_RATIO = 0.98
STOP_LOSS = 0.06
TAKE_PROFIT = 0.06

DD_SOFT = 0.07
DD_HARD = 0.14
DD_RECOVER = 0.04
DD_SOFT_SCALE = 0.80
DD_HARD_SCALE = 0.60

TARGET_PCTS = [0.82, 0.84, 0.85, 0.86, 0.88, 0.90]

STARTS = [
    ("full_20240605", "2024-06-05 09:00:00", liq.BACKTEST_END),
    ("recent60_20260324", "2026-03-24 09:00:00", liq.BACKTEST_END),
    ("late_20250701", "2025-07-01 09:00:00", liq.BACKTEST_END),
    ("late_20251009", "2025-10-09 09:00:00", liq.BACKTEST_END),
    ("late_20260105", "2026-01-05 09:00:00", liq.BACKTEST_END),
]


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


def _case_name(target_pct: float) -> str:
    return f"dd0714_tp060_pos{int(round(target_pct * 100)):02d}"


def _signal_for_case(target_pct: float) -> Path:
    if not BASE_SIGNAL_FILE.exists():
        raise FileNotFoundError(BASE_SIGNAL_FILE)
    path = REPORT_DIR / "signals" / f"{_case_name(target_pct)}.csv"
    if path.exists():
        return path
    rows = _read_rows(BASE_SIGNAL_FILE)
    for row in rows:
        row["target_pct"] = f"{target_pct:.5f}"
        row["signal_stop_loss_pct"] = f"{STOP_LOSS:.5f}"
        row["signal_take_profit_pct"] = f"{TAKE_PROFIT:.5f}"
        row["strategy_variant"] = f"dynamic_h2_m3_dd0714_tp060_pos{target_pct:.2f}"
    _write_rows(path, rows)
    return path


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


def _run(target_pct: float, signal_file: Path, start_name: str, start: str, end: str) -> dict:
    case_name = _case_name(target_pct)
    log_file = REPORT_DIR / "logs" / f"{case_name}_{start_name}.log"
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
            str(target_pct),
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
            str(STOP_LOSS),
            "--take-profit-pct",
            str(TAKE_PROFIT),
        ]
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w", encoding="utf-8") as log:
            proc = subprocess.run(command, cwd=str(liq.MAIN), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        returncode = proc.returncode
        indicator = _extract_indicator(log_file)
    return {
        "case_name": case_name,
        "target_pct": target_pct,
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


def _summarize(detail: list[dict], signal_stats: dict[str, dict]) -> list[dict]:
    out = []
    for case_name in sorted({row["case_name"] for row in detail}):
        rows = [row for row in detail if row["case_name"] == case_name]
        by_start = {row["start_name"]: row for row in rows}
        late = [_f(by_start[key]["annual"]) for key in ["late_20250701", "late_20251009", "late_20260105"] if key in by_start]
        late = [value for value in late if value == value]
        full = by_start.get("full_20240605", {})
        recent60 = by_start.get("recent60_20260324", {})
        out.append(
            {
                "case_name": case_name,
                "target_pct": rows[0]["target_pct"],
                "full_annual": full.get("annual"),
                "full_sharpe": full.get("sharpe"),
                "full_max_drawdown": full.get("max_drawdown"),
                "full_avg_invested_pct": full.get("avg_invested_pct"),
                "recent60_annual": recent60.get("annual"),
                "recent60_sharpe": recent60.get("sharpe"),
                "recent60_max_drawdown": recent60.get("max_drawdown"),
                "late_min_annual": min(late) if late else None,
                "late_median_annual": float(pd.Series(late).median()) if late else None,
                **signal_stats.get(case_name, {}),
            }
        )
    return out


def _write_report(summary: list[dict], audit: dict) -> None:
    ranked = sorted(summary, key=lambda row: (_f(row["full_annual"]), _f(row["full_sharpe"])), reverse=True)
    lines = [
        "# dd0714_tp060 高仓位邻域验证",
        "",
        "## 当前结论",
        "",
        "本轮固定当前高收益候选的入场、动态持有、6% 止盈、6% 止损和 dd0714 账户回撤缩放，只向上测试目标仓位。目标是确认 `84%` 是否局部最优，以及更高仓位是否突破强准入风险边界。",
        "",
        "| 候选 | 目标仓位 | 全周期年化 | Sharpe | 最大回撤 | recent60 年化 | recent60 Sharpe | recent60 最大回撤 | 近期开仓最差年化 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ranked:
        lines.append(
            f"| {row['case_name']} | {_f(row['target_pct']):.0%} | {_f(row['full_annual']):.2%} | {_f(row['full_sharpe']):.2f} | "
            f"{_f(row['full_max_drawdown']):.2%} | {_f(row['recent60_annual']):.2%} | {_f(row['recent60_sharpe']):.2f} | "
            f"{_f(row['recent60_max_drawdown']):.2%} | {_f(row['late_min_annual']):.2%} |"
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
            f"- 信号文件：`{REPORT_DIR / 'signals'}`",
            f"- 掘金日志：`{REPORT_DIR / 'logs'}`",
        ]
    )
    (REPORT_DIR / "high_position_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    detail: list[dict] = []
    signal_files: list[Path] = []
    signal_stats: dict[str, dict] = {}
    for target_pct in TARGET_PCTS:
        signal_file = _signal_for_case(target_pct)
        signal_files.append(signal_file)
        rows = _read_rows(signal_file)
        case_name = _case_name(target_pct)
        signal_stats[case_name] = {
            "signal_rows": len(rows),
            "signal_days": len({row["signal_date"] for row in rows}),
            "latest_signal_date": max((row["signal_date"] for row in rows), default=None),
            "latest_buy_date": max((row["buy_date"] for row in rows), default=None),
        }
        for start_name, start, end in STARTS:
            row = _run(target_pct, signal_file, start_name, start, end)
            detail.append(row)
            print(f"{case_name} {start_name} annual={row['annual']} sharpe={row['sharpe']} maxdd={row['max_drawdown']}", flush=True)
    summary = _summarize(detail, signal_stats)
    _write_rows(REPORT_DIR / "detail.csv", detail)
    _write_rows(REPORT_DIR / "summary.csv", summary)
    _write_rows(REPORT_DIR / "summary_by_annual.csv", sorted(summary, key=lambda row: _f(row["full_annual"]), reverse=True))
    audit = liq._audit_signals(signal_files)
    (REPORT_DIR / "hard_gate_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(summary, audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
