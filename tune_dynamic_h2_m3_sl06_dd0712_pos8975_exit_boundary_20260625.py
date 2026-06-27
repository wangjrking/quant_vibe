from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import median

import tune_strict_sync_liquidity_refill_20260624 as liq
import validate_dynamic_h2_m3_sl06_dd0509_pos86_tp060_20260624 as core


BASE_DIR = core.SOURCE_DIR / "formal_horizon_entry_confirmation_20260624"
REPORT_DIR = BASE_DIR / "dynamic_h2_m3_sl06_dd0712_pos8975_exit_boundary_20260625"
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
    "case_name": "base_pos8975_scale7555_tp070_e970_c975",
    "target_pct": 0.8975,
    "exit_ratio": 0.970,
    "continue_ratio": 0.975,
    "holding_days": 2,
    "max_holding_days": 3,
    "stop_loss": 0.060,
    "take_profit": 0.070,
    "dd_soft": 0.070,
    "dd_hard": 0.120,
    "dd_recover": 0.030,
    "dd_soft_scale": 0.75,
    "dd_hard_scale": 0.55,
}


CASES = [
    BASE_CASE,
    {**BASE_CASE, "case_name": "pos8975_tp065", "take_profit": 0.065},
    {**BASE_CASE, "case_name": "pos8975_tp075", "take_profit": 0.075},
    {**BASE_CASE, "case_name": "pos8975_tp080", "take_profit": 0.080},
    {**BASE_CASE, "case_name": "pos8975_e965", "exit_ratio": 0.965},
    {**BASE_CASE, "case_name": "pos8975_e975", "exit_ratio": 0.975},
    {**BASE_CASE, "case_name": "pos8975_e980", "exit_ratio": 0.980},
    {**BASE_CASE, "case_name": "pos8975_c970", "continue_ratio": 0.970},
    {**BASE_CASE, "case_name": "pos8975_c9725", "continue_ratio": 0.9725},
    {**BASE_CASE, "case_name": "pos8975_c9775", "continue_ratio": 0.9775},
    {**BASE_CASE, "case_name": "pos8975_c980", "continue_ratio": 0.980},
    {**BASE_CASE, "case_name": "pos8975_tp075_e965", "take_profit": 0.075, "exit_ratio": 0.965},
    {**BASE_CASE, "case_name": "pos8975_tp075_e975", "take_profit": 0.075, "exit_ratio": 0.975},
    {**BASE_CASE, "case_name": "pos90s7454_tp070", "target_pct": 0.9000, "dd_soft_scale": 0.74, "dd_hard_scale": 0.54},
    {**BASE_CASE, "case_name": "pos90s7454_tp075", "target_pct": 0.9000, "dd_soft_scale": 0.74, "dd_hard_scale": 0.54, "take_profit": 0.075},
    {**BASE_CASE, "case_name": "pos895s7353_tp070", "target_pct": 0.8950, "dd_soft_scale": 0.73, "dd_hard_scale": 0.53},
    {**BASE_CASE, "case_name": "pos895s7353_tp075", "target_pct": 0.8950, "dd_soft_scale": 0.73, "dd_hard_scale": 0.53, "take_profit": 0.075},
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
        _f(row["recent60_annual"]) >= 0.45,
        _f(row["full_annual"]),
        _f(row["recent60_annual"]),
        _f(row["full_sharpe"]),
    )


def _write_report(summary: list[dict], audit: dict) -> None:
    ranked = sorted(summary, key=_ranking_key, reverse=True)
    lines = [
        "# pos8975 退出参数邻域验证",
        "",
        "## 当前结论",
        "",
        "本轮固定 formal L4 输入、Top1 股票池、10D/5D/3D 入场权重、ST/BJ/退市/涨停硬过滤和非行业非月份口径，只微调止盈、分数退出阈值、继续持有阈值及两个已验证的仓位缩放邻近点。风格暴露和风格漂移仅作为提示项。",
        "",
        "| 候选 | 年化 | Sharpe | 最大回撤 | recent60 年化 | 近期开仓最差年化 | 平均仓位 | 参数摘要 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in ranked:
        params = (
            f"pos={_f(row['target_pct']):.2%}, tp={_f(row['take_profit']):.2%}, "
            f"exit={_f(row['exit_ratio']):.3f}, cont={_f(row['continue_ratio']):.4f}, "
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
    (REPORT_DIR / "exit_boundary_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
            detail.append(_run_case(case, signal_file, start_name, start, end))
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
