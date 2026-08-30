from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import os
import re
import subprocess
from pathlib import Path

import tune_strict_sync_liquidity_refill_20260624 as liq


CASE_NAME = "liq_amt10w_mv20w_w90_5d10_pos75_h3_e096_mh1"
SOURCE_DIR = liq.SOURCE_DIR / "strict_sync_liquidity_neighborhood_20260624"
REPORT_DIR = SOURCE_DIR / "time_slice_check_20260624"
SIGNAL_FILE = SOURCE_DIR / "signals" / f"{CASE_NAME}.csv"

SLICES = [
    ("full", "2024-06-05 09:00:00", "2026-06-23 15:30:00"),
    ("slice_2024h2", "2024-06-05 09:00:00", "2024-12-31 15:30:00"),
    ("slice_2025h1", "2025-01-02 09:00:00", "2025-06-30 15:30:00"),
    ("slice_2025h2", "2025-07-01 09:00:00", "2025-12-31 15:30:00"),
    ("slice_2026ytd", "2026-01-05 09:00:00", "2026-06-23 15:30:00"),
    ("slice_recent120", "2025-12-24 09:00:00", "2026-06-23 15:30:00"),
    ("slice_recent60", "2026-03-24 09:00:00", "2026-06-23 15:30:00"),
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


def _run_slice(name: str, start: str, end: str) -> dict:
    log_file = REPORT_DIR / "logs" / f"{CASE_NAME}_{name}.log"
    indicator = _extract_indicator(log_file)
    returncode = 0
    if indicator is None:
        env = os.environ.copy()
        env.update(liq.BASE_ENV)
        env["GM_SCORE_EXIT_ENTRY_RATIO"] = "0.96"
        env["GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT"] = "1"
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
            "3",
            "--max-holding-days",
            "3",
            "--target-position-pct",
            "0.75",
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
        ]
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w", encoding="utf-8") as log:
            proc = subprocess.run(command, cwd=str(liq.MAIN), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        returncode = proc.returncode
        indicator = _extract_indicator(log_file)
    return {
        "slice": name,
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
        "log_file": str(log_file),
        **_exposure_stats(log_file),
    }


def _write_report(rows: list[dict], audit: dict) -> None:
    lines = [
        "# 流动性候选时间切片检查",
        "",
        "## 当前结论",
        "",
        f"检查对象：`{CASE_NAME}`。本轮只改变掘金回测起止区间，不改变信号、模型、过滤或交易参数。",
        "",
        "## 掘金时间切片",
        "",
        "| 切片 | 起点 | 终点 | 年化 | 累计 | 夏普 | 最大回撤 | 开仓 | 平均持仓率 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['slice']} | {row['start']} | {row['end']} | "
            f"{float(row['annual']):.4f} | {float(row['pnl_ratio']):.4f} | "
            f"{float(row['sharpe']):.4f} | {float(row['max_drawdown']):.4f} | "
            f"{row['open_count']} | {float(row['avg_invested_pct']):.4f} |"
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
        f"- 明细：`{REPORT_DIR / 'time_slice_results.csv'}`",
        f"- 掘金日志：`{REPORT_DIR / 'logs'}`",
        f"- 硬过滤审计：`{REPORT_DIR / 'time_slice_hard_gate_audit.json'}`",
        f"- 信号文件：`{SIGNAL_FILE}`",
    ])
    (REPORT_DIR / "time_slice_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    if not SIGNAL_FILE.exists():
        raise FileNotFoundError(SIGNAL_FILE)
    rows: list[dict] = []
    rows_path = REPORT_DIR / "time_slice_results.csv"
    if rows_path.exists():
        with rows_path.open("r", encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
    existing = {row["slice"] for row in rows}
    for index, (name, start, end) in enumerate(SLICES, start=1):
        if name in existing:
            print(f"[{index}/{len(SLICES)}] reuse {name}", flush=True)
            continue
        row = _run_slice(name, start, end)
        rows.append(row)
        _write_rows(rows_path, rows)
        print(
            f"[{index}/{len(SLICES)}] {name} annual={row['annual']} sharpe={row['sharpe']} maxdd={row['max_drawdown']}",
            flush=True,
        )
    audit = liq._audit_signals([SIGNAL_FILE])
    (REPORT_DIR / "time_slice_hard_gate_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT_DIR / "time_slice_summary.json").write_text(json.dumps({"rows": rows, "audit": audit}, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(rows, audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
