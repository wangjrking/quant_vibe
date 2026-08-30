from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import median

import tune_strict_sync_liquidity_refill_20260624 as liq
import validate_dynamic_h2_m3_sl06_dd0509_pos86_tp060_20260624 as core


BASE_DIR = core.SOURCE_DIR / "formal_horizon_entry_confirmation_20260624"
REPORT_DIR = BASE_DIR / "dynamic_h2_m3_sl06_dd0712_pos89_c975_high_pos_boundary_20260625"
SOURCE_SIGNAL = (
    BASE_DIR
    / "dynamic_h2_m3_sl06_dd0712_pos89_c975_fine_neighborhood_20260624"
    / "signals"
    / "c975_tp070.csv"
)

SCREEN_STARTS = [
    ("full_20240605", "2024-06-05 09:00:00", liq.BACKTEST_END),
    ("recent60_20260324", "2026-03-24 09:00:00", liq.BACKTEST_END),
    ("late_20250701", "2025-07-01 09:00:00", liq.BACKTEST_END),
    ("late_20251009", "2025-10-09 09:00:00", liq.BACKTEST_END),
    ("late_20260105", "2026-01-05 09:00:00", liq.BACKTEST_END),
]

BASE_CASE = {
    "case_name": "base_pos89_scale7050",
    "target_pct": 0.89,
    "exit_ratio": 0.970,
    "continue_ratio": 0.975,
    "holding_days": 2,
    "max_holding_days": 3,
    "stop_loss": 0.060,
    "take_profit": 0.070,
    "dd_soft": 0.070,
    "dd_hard": 0.120,
    "dd_recover": 0.030,
    "dd_soft_scale": 0.70,
    "dd_hard_scale": 0.50,
}

CASES = [
    BASE_CASE,
    {**BASE_CASE, "case_name": "pos895_scale7353", "target_pct": 0.895, "dd_soft_scale": 0.73, "dd_hard_scale": 0.53},
    {**BASE_CASE, "case_name": "pos895_scale7454", "target_pct": 0.895, "dd_soft_scale": 0.74, "dd_hard_scale": 0.54},
    {**BASE_CASE, "case_name": "pos895_scale7555", "target_pct": 0.895, "dd_soft_scale": 0.75, "dd_hard_scale": 0.55},
    {**BASE_CASE, "case_name": "pos8975_scale7353", "target_pct": 0.8975, "dd_soft_scale": 0.73, "dd_hard_scale": 0.53},
    {**BASE_CASE, "case_name": "pos8975_scale7454", "target_pct": 0.8975, "dd_soft_scale": 0.74, "dd_hard_scale": 0.54},
    {**BASE_CASE, "case_name": "pos8975_scale7555", "target_pct": 0.8975, "dd_soft_scale": 0.75, "dd_hard_scale": 0.55},
    {**BASE_CASE, "case_name": "pos90_scale7252", "target_pct": 0.90, "dd_soft_scale": 0.72, "dd_hard_scale": 0.52},
    {**BASE_CASE, "case_name": "pos90_scale7353", "target_pct": 0.90, "dd_soft_scale": 0.73, "dd_hard_scale": 0.53},
    {**BASE_CASE, "case_name": "pos90_scale7454", "target_pct": 0.90, "dd_soft_scale": 0.74, "dd_hard_scale": 0.54},
    {**BASE_CASE, "case_name": "pos90_scale7555", "target_pct": 0.90, "dd_soft_scale": 0.75, "dd_hard_scale": 0.55},
    {**BASE_CASE, "case_name": "pos9025_scale7252", "target_pct": 0.9025, "dd_soft_scale": 0.72, "dd_hard_scale": 0.52},
    {**BASE_CASE, "case_name": "pos9025_scale7353", "target_pct": 0.9025, "dd_soft_scale": 0.73, "dd_hard_scale": 0.53},
    {**BASE_CASE, "case_name": "pos905_scale7252", "target_pct": 0.905, "dd_soft_scale": 0.72, "dd_hard_scale": 0.52},
    {**BASE_CASE, "case_name": "pos90_dd065115_scale7555", "target_pct": 0.90, "dd_soft": 0.065, "dd_hard": 0.115, "dd_soft_scale": 0.75, "dd_hard_scale": 0.55},
    {**BASE_CASE, "case_name": "pos8975_dd065115_scale7555", "target_pct": 0.8975, "dd_soft": 0.065, "dd_hard": 0.115, "dd_soft_scale": 0.75, "dd_hard_scale": 0.55},
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


def _signal_for_case(case: dict) -> Path:
    path = REPORT_DIR / "signals" / f"{case['case_name']}.csv"
    if path.exists():
        return path
    rows = _read_rows(SOURCE_SIGNAL)
    for row in rows:
        row["target_pct"] = f"{case['target_pct']:.5f}"
        row["holding_days"] = str(case["holding_days"])
        row["max_holding_days"] = str(case["max_holding_days"])
        row["score_exit_entry_ratio"] = f"{case['exit_ratio']:.5f}"
        row["score_continue_entry_ratio"] = f"{case['continue_ratio']:.5f}"
        row["signal_stop_loss_pct"] = f"{case['stop_loss']:.5f}"
        row["signal_take_profit_pct"] = f"{case['take_profit']:.5f}"
        row["strategy_variant"] = case["case_name"]
    _write_rows(path, rows)
    return path


def _run_case(case: dict, signal_file: Path, start_name: str, start: str, end: str) -> dict:
    old_values = {
        "REPORT_DIR": core.REPORT_DIR,
        "TARGET_PCT": core.TARGET_PCT,
        "HOLDING_DAYS": core.HOLDING_DAYS,
        "MAX_HOLDING_DAYS": core.MAX_HOLDING_DAYS,
        "EXIT_RATIO": core.EXIT_RATIO,
        "MIN_HOLD": core.MIN_HOLD,
        "CONTINUE_RATIO": core.CONTINUE_RATIO,
        "INTRADAY_STOP_LOSS": core.INTRADAY_STOP_LOSS,
        "TAKE_PROFIT": core.TAKE_PROFIT,
        "DD_SOFT": core.DD_SOFT,
        "DD_HARD": core.DD_HARD,
        "DD_RECOVER": core.DD_RECOVER,
        "DD_SOFT_SCALE": core.DD_SOFT_SCALE,
        "DD_HARD_SCALE": core.DD_HARD_SCALE,
    }
    try:
        core.REPORT_DIR = REPORT_DIR
        core.TARGET_PCT = float(case["target_pct"])
        core.HOLDING_DAYS = int(case["holding_days"])
        core.MAX_HOLDING_DAYS = int(case["max_holding_days"])
        core.EXIT_RATIO = float(case["exit_ratio"])
        core.MIN_HOLD = 1
        core.CONTINUE_RATIO = float(case["continue_ratio"])
        core.INTRADAY_STOP_LOSS = float(case["stop_loss"])
        core.TAKE_PROFIT = float(case["take_profit"])
        core.DD_SOFT = float(case["dd_soft"])
        core.DD_HARD = float(case["dd_hard"])
        core.DD_RECOVER = float(case["dd_recover"])
        core.DD_SOFT_SCALE = float(case["dd_soft_scale"])
        core.DD_HARD_SCALE = float(case["dd_hard_scale"])
        row = core._run(signal_file, f"{case['case_name']}_{start_name}", start, end)
        row["case_name"] = case["case_name"]
        row["start_name"] = start_name
        for key, value in case.items():
            if key != "case_name":
                row[key] = value
        return row
    finally:
        for key, value in old_values.items():
            setattr(core, key, value)


def _summarize(detail: list[dict], signal_stats: dict[str, dict]) -> list[dict]:
    out: list[dict] = []
    for case in CASES:
        rows = [row for row in detail if row["case_name"] == case["case_name"]]
        by_start = {row["start_name"]: row for row in rows}
        late = [
            _f(by_start[key]["annual"])
            for key in ["late_20250701", "late_20251009", "late_20260105"]
            if key in by_start
        ]
        late = [value for value in late if value == value]
        full = by_start.get("full_20240605", {})
        recent60 = by_start.get("recent60_20260324", {})
        out.append(
            {
                **case,
                "full_annual": full.get("annual"),
                "full_sharpe": full.get("sharpe"),
                "full_max_drawdown": full.get("max_drawdown"),
                "full_avg_invested_pct": full.get("avg_invested_pct"),
                "recent60_annual": recent60.get("annual"),
                "recent60_sharpe": recent60.get("sharpe"),
                "recent60_max_drawdown": recent60.get("max_drawdown"),
                "late_min_annual": min(late) if late else None,
                "late_median_annual": median(late) if late else None,
                "late_max_drawdown_max": max(
                    [_f(row.get("max_drawdown")) for row in rows if row["start_name"].startswith("late_")],
                    default=None,
                ),
                **signal_stats.get(case["case_name"], {}),
            }
        )
    return out


def _ranking_key(row: dict) -> tuple:
    return (
        _f(row["full_max_drawdown"]) <= 0.40,
        _f(row["late_min_annual"]) >= 1.50,
        _f(row["recent60_annual"]) >= 0.40,
        _f(row["full_annual"]),
        _f(row["full_sharpe"]),
    )


def _write_report(summary: list[dict], audit: dict) -> None:
    ranked = sorted(summary, key=_ranking_key, reverse=True)
    lines = [
        "# 高仓位边界精调验证",
        "",
        "## 当前结论",
        "",
        "本轮固定 formal L4 输入、Top1 股票池、ST/BJ/退市/涨停硬过滤和非行业非月份口径，只在 `pos90/scale7555` 附近微调目标仓位与账户回撤缩放。风格暴露和风格漂移仅作为提示，不作为单独强准入。",
        "",
        "| 候选 | 年化 | Sharpe | 最大回撤 | recent60 年化 | 近期开仓最差年化 | 平均仓位 | 参数摘要 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in ranked:
        params = (
            f"pos={_f(row['target_pct']):.2%}, "
            f"dd={_f(row['dd_soft']):.1%}/{_f(row['dd_hard']):.1%}, "
            f"scale={_f(row['dd_soft_scale']):.0%}/{_f(row['dd_hard_scale']):.0%}"
        )
        lines.append(
            f"| `{row['case_name']}` | {_f(row['full_annual']):.2%} | {_f(row['full_sharpe']):.2f} | "
            f"{_f(row['full_max_drawdown']):.2%} | {_f(row['recent60_annual']):.2%} | "
            f"{_f(row['late_min_annual']):.2%} | {_f(row['full_avg_invested_pct']):.2%} | {params} |"
        )
    lines.extend(
        [
            "",
            "## 硬过滤审计",
            "",
            f"- 审计文件数：`{audit.get('audited_files')}`",
            f"- 失败文件数：`{audit.get('failed_files')}`",
            f"- 买入日行情缺失：`{audit.get('buy_join_missing')}`",
            f"- 北交所命中：`{audit.get('bj_rows')}`",
            f"- ST / 风险警示命中：`{audit.get('buy_st_rows')}`",
            f"- 退市命中：`{audit.get('buy_delist_rows')}`",
            f"- 开盘涨停买入命中：`{audit.get('open_limit_up_buy_rows')}`",
            "",
            "## 证据路径",
            "",
            f"- 明细：`{REPORT_DIR / 'detail.csv'}`",
            f"- 汇总：`{REPORT_DIR / 'summary.csv'}`",
            f"- 准入排序：`{REPORT_DIR / 'summary_by_admission.csv'}`",
            f"- 硬过滤：`{REPORT_DIR / 'hard_gate_audit.json'}`",
            f"- 掘金日志：`{REPORT_DIR / 'logs'}`",
        ]
    )
    (REPORT_DIR / "high_pos_boundary_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    if not SOURCE_SIGNAL.exists():
        raise FileNotFoundError(SOURCE_SIGNAL)
    detail: list[dict] = []
    signal_files: list[Path] = []
    signal_stats: dict[str, dict] = {}
    total = len(CASES) * len(SCREEN_STARTS)
    done = 0
    for case in CASES:
        signal_file = _signal_for_case(case)
        signal_files.append(signal_file)
        rows = _read_rows(signal_file)
        signal_stats[case["case_name"]] = {
            "signal_rows": len(rows),
            "signal_days": len({row["signal_date"] for row in rows}),
            "latest_signal_date": max((row["signal_date"] for row in rows), default=None),
            "latest_buy_date": max((row["buy_date"] for row in rows), default=None),
        }
        for start_name, start, end in SCREEN_STARTS:
            done += 1
            print(f"[{done}/{total}] {case['case_name']} {start_name}", flush=True)
            row = _run_case(case, signal_file, start_name, start, end)
            detail.append(row)
    summary = _summarize(detail, signal_stats)
    _write_rows(REPORT_DIR / "detail.csv", detail)
    _write_rows(REPORT_DIR / "summary.csv", summary)
    _write_rows(REPORT_DIR / "summary_by_admission.csv", sorted(summary, key=_ranking_key, reverse=True))
    audit = liq._audit_signals(signal_files)
    (REPORT_DIR / "hard_gate_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(summary, audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
