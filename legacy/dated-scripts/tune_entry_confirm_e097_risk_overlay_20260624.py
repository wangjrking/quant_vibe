from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import os
import re
import subprocess
from pathlib import Path

import pandas as pd

import tune_strict_sync_liquidity_refill_20260624 as liq


SOURCE_DIR = liq.SOURCE_DIR / "strict_sync_liquidity_neighborhood_20260624"
REPORT_DIR = SOURCE_DIR / "formal_horizon_entry_confirmation_20260624" / "entry_confirm_e097_risk_overlay_20260624"
SIGNAL_FILE = (
    SOURCE_DIR
    / "formal_horizon_entry_confirmation_20260624"
    / "entry_confirm_exit_neighborhood_20260624"
    / "signals"
    / "entry_confirm_w78_5d12_3d10_pos56_h3_e097_mh1.csv"
)

TARGET_PCT = 0.56
HOLDING_DAYS = 3
EXIT_RATIO = 0.97
MIN_HOLD = 1

STARTS = [
    ("full_20240605", "2024-06-05 09:00:00"),
    ("late_20250701", "2025-07-01 09:00:00"),
    ("late_20251009", "2025-10-09 09:00:00"),
    ("late_20260105", "2026-01-05 09:00:00"),
]

RISK_CASES = [
    {"name": "no_dd_overlay", "enabled": False},
    {"name": "dd_08_14_s85_65", "enabled": True, "soft": 0.08, "hard": 0.14, "recover": 0.04, "soft_scale": 0.85, "hard_scale": 0.65, "resize_existing": False},
    {"name": "dd_10_18_s85_65", "enabled": True, "soft": 0.10, "hard": 0.18, "recover": 0.05, "soft_scale": 0.85, "hard_scale": 0.65, "resize_existing": False},
    {"name": "dd_10_18_s75_55", "enabled": True, "soft": 0.10, "hard": 0.18, "recover": 0.05, "soft_scale": 0.75, "hard_scale": 0.55, "resize_existing": False},
    {"name": "dd_12_20_s80_60", "enabled": True, "soft": 0.12, "hard": 0.20, "recover": 0.06, "soft_scale": 0.80, "hard_scale": 0.60, "resize_existing": False},
    {"name": "dd_10_18_s85_65_resize", "enabled": True, "soft": 0.10, "hard": 0.18, "recover": 0.05, "soft_scale": 0.85, "hard_scale": 0.65, "resize_existing": True},
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


def _apply_risk_env(env: dict[str, str], case: dict) -> None:
    if not case.get("enabled"):
        env["GM_EQUITY_DD_RISK_MODE"] = "0"
        return
    env["GM_EQUITY_DD_RISK_MODE"] = "1"
    env["GM_EQUITY_DD_SOFT_TRIGGER"] = str(case["soft"])
    env["GM_EQUITY_DD_HARD_TRIGGER"] = str(case["hard"])
    env["GM_EQUITY_DD_RECOVER_TRIGGER"] = str(case["recover"])
    env["GM_EQUITY_DD_SOFT_SCALE"] = str(case["soft_scale"])
    env["GM_EQUITY_DD_HARD_SCALE"] = str(case["hard_scale"])
    env["GM_EQUITY_DD_RESIZE_EXISTING"] = "1" if case.get("resize_existing") else "0"


def _run(case: dict, start_name: str, start: str) -> dict:
    log_file = REPORT_DIR / "logs" / f"{case['name']}_{start_name}.log"
    indicator = _extract_indicator(log_file)
    returncode = 0
    if indicator is None:
        env = os.environ.copy()
        env.update(liq.BASE_ENV)
        env["GM_FORCE_SELL_MARKET_ORDER"] = "1"
        env["GM_SCORE_EXIT_ENTRY_RATIO"] = str(EXIT_RATIO)
        env["GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT"] = str(MIN_HOLD)
        _apply_risk_env(env, case)
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
            str(HOLDING_DAYS),
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
            liq.BACKTEST_END,
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
        "risk_name": case["name"],
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


def _summarize(rows: list[dict]) -> list[dict]:
    out = []
    for risk_name in sorted({row["risk_name"] for row in rows}):
        items = [row for row in rows if row["risk_name"] == risk_name]
        full = next((row for row in items if row["start_name"] == "full_20240605"), None)
        late_annuals = [_f(row["annual"]) for row in items if row["start_name"].startswith("late_")]
        late_annuals = [value for value in late_annuals if value == value]
        out.append(
            {
                "risk_name": risk_name,
                "full_annual": full.get("annual") if full else None,
                "full_sharpe": full.get("sharpe") if full else None,
                "full_max_drawdown": full.get("max_drawdown") if full else None,
                "late_min_annual": min(late_annuals) if late_annuals else None,
                "late_median_annual": float(pd.Series(late_annuals).median()) if late_annuals else None,
                "avg_invested_pct": full.get("avg_invested_pct") if full else None,
                "open_count": full.get("open_count") if full else None,
            }
        )
    return out


def _write_report(summary: list[dict]) -> None:
    rows = sorted(summary, key=lambda row: (_f(row["full_max_drawdown"]) * -1, _f(row["full_annual"])), reverse=True)
    lines = [
        "# entry_confirm e097 账户回撤缩放验证",
        "",
        "## 当前结论",
        "",
        "本实验只测试通用账户回撤缩放，不改变选股池、不使用日期、月份、行业过滤。若风险缩放不能明显降低回撤或导致收益大幅下降，则不采纳为生产候选规则。",
        "",
        "| 风控版本 | 年化 | Sharpe | 最大回撤 | 近期开仓最差年化 | 近期开仓中位年化 | 平均仓位 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['risk_name']} | {_f(row['full_annual']):.2%} | {_f(row['full_sharpe']):.2f} | {_f(row['full_max_drawdown']):.2%} | "
            f"{_f(row['late_min_annual']):.2%} | {_f(row['late_median_annual']):.2%} | {_f(row['avg_invested_pct']):.2%} |"
        )
    lines.extend([
        "",
        "## 证据路径",
        "",
        f"- 明细：`{REPORT_DIR / 'detail.csv'}`",
        f"- 汇总：`{REPORT_DIR / 'summary.csv'}`",
        f"- 掘金日志：`{REPORT_DIR / 'logs'}`",
    ])
    (REPORT_DIR / "risk_overlay_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    if not SIGNAL_FILE.exists():
        raise FileNotFoundError(SIGNAL_FILE)
    rows = []
    for case in RISK_CASES:
        for start_name, start in STARTS:
            row = _run(case, start_name, start)
            rows.append(row)
            print(f"{case['name']} {start_name} annual={row['annual']} sharpe={row['sharpe']} maxdd={row['max_drawdown']}", flush=True)
    summary = _summarize(rows)
    _write_rows(REPORT_DIR / "detail.csv", rows)
    _write_rows(REPORT_DIR / "summary.csv", summary)
    _write_rows(REPORT_DIR / "summary_by_drawdown.csv", sorted(summary, key=lambda row: _f(row["full_max_drawdown"])))
    _write_report(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
