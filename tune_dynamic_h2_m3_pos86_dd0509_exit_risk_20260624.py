from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

import tune_dynamic_h2_m3_pos94_dd0509_exit_risk_20260624 as risk


BASE_DIR = risk.BASE_DIR
REPORT_DIR = BASE_DIR / "dynamic_h2_m3_pos86_dd0509_exit_risk_20260624"
BASE_SIGNAL_FILE = BASE_DIR / "dynamic_h2_m3_pos86_tp060_dd_neighborhood_20260624" / "signals" / "tp060_pos86_dd0509_s7050.csv"

TARGET_PCT = 0.86
RISK_CASES = [
    {"case_name": f"pos86_sl{sl:02d}_tp{tp:02d}", "stop_loss": sl / 100, "take_profit": tp / 100}
    for sl in [5, 6, 7]
    for tp in [5, 6, 7, 8]
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
    rows = _read_rows(BASE_SIGNAL_FILE)
    for row in rows:
        row["target_pct"] = f"{TARGET_PCT:.5f}"
        row["signal_stop_loss_pct"] = f"{case['stop_loss']:.5f}"
        row["signal_take_profit_pct"] = f"{case['take_profit']:.5f}"
        row["strategy_variant"] = (
            f"dynamic_h2_m3_dd0509_pos86_sl{case['stop_loss']:.2f}_tp{case['take_profit']:.2f}"
        )
    _write_rows(path, rows)
    return path


def _run(case: dict, signal_file: Path, start_name: str, start: str, end: str) -> dict:
    old_report = risk.REPORT_DIR
    old_target = risk.TARGET_PCT
    try:
        risk.REPORT_DIR = REPORT_DIR
        risk.TARGET_PCT = TARGET_PCT
        return risk._run(case, signal_file, start_name, start, end)
    finally:
        risk.REPORT_DIR = old_report
        risk.TARGET_PCT = old_target


def _summarize(detail: list[dict], signal_stats: dict[str, dict]) -> list[dict]:
    out: list[dict] = []
    for case_name in sorted({row["case_name"] for row in detail}):
        rows = [row for row in detail if row["case_name"] == case_name]
        by_start = {row["start_name"]: row for row in rows}
        late = [
            _f(by_start[key]["annual"])
            for key in ["late_20250701", "late_20251009", "late_20260105"]
            if key in by_start
        ]
        late = [value for value in late if value == value]
        full = by_start.get("full_20240605", {})
        recent60 = by_start.get("recent60_20260324", {})
        first = rows[0]
        out.append(
            {
                "case_name": case_name,
                "target_pct": TARGET_PCT,
                "stop_loss": first["stop_loss"],
                "take_profit": first["take_profit"],
                "dd_soft": risk.DD_SOFT,
                "dd_hard": risk.DD_HARD,
                "dd_recover": risk.DD_RECOVER,
                "dd_soft_scale": risk.DD_SOFT_SCALE,
                "dd_hard_scale": risk.DD_HARD_SCALE,
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
    ranked = sorted(
        summary,
        key=lambda row: (
            _f(row["full_max_drawdown"]) <= 0.40,
            _f(row["recent60_annual"]),
            _f(row["full_annual"]),
        ),
        reverse=True,
    )
    lines = [
        "# pos86 dd0509 止损止盈邻域验证",
        "",
        "## 当前结论",
        "",
        "本轮固定 formal L4 融合信号、Top1、目标仓位 86%、账户回撤缩放 dd0509 和动态持有 h2_m3，只测试通用止损/止盈阈值。收益指标只作为弱准入排序；最大回撤、recent60、近期开仓和硬过滤作为强准入判断。",
        "",
        "| 候选 | 止损 | 止盈 | 年化 | Sharpe | 最大回撤 | recent60 年化 | 近期开仓最差年化 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ranked:
        lines.append(
            f"| `{row['case_name']}` | {_f(row['stop_loss']):.0%} | {_f(row['take_profit']):.0%} | "
            f"{_f(row['full_annual']):.2%} | {_f(row['full_sharpe']):.2f} | "
            f"{_f(row['full_max_drawdown']):.2%} | {_f(row['recent60_annual']):.2%} | "
            f"{_f(row['late_min_annual']):.2%} |"
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
        ]
    )
    (REPORT_DIR / "exit_risk_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    detail: list[dict] = []
    signal_files: list[Path] = []
    signal_stats: dict[str, dict] = {}
    for case in RISK_CASES:
        signal_file = _signal_for_case(case)
        signal_files.append(signal_file)
        rows = _read_rows(signal_file)
        case_name = case["case_name"]
        signal_stats[case_name] = {
            "signal_rows": len(rows),
            "signal_days": len({row["signal_date"] for row in rows}),
            "latest_signal_date": max((row["signal_date"] for row in rows), default=None),
            "latest_buy_date": max((row["buy_date"] for row in rows), default=None),
        }
        for start_name, start, end in risk.STARTS:
            row = _run(case, signal_file, start_name, start, end)
            detail.append(row)
            print(
                f"{case_name} {start_name} annual={row['annual']} sharpe={row['sharpe']} maxdd={row['max_drawdown']}",
                flush=True,
            )
    summary = _summarize(detail, signal_stats)
    _write_rows(REPORT_DIR / "detail.csv", detail)
    _write_rows(REPORT_DIR / "summary.csv", summary)
    _write_rows(REPORT_DIR / "summary_by_recent60.csv", sorted(summary, key=lambda row: _f(row["recent60_annual"]), reverse=True))
    _write_rows(REPORT_DIR / "summary_by_annual.csv", sorted(summary, key=lambda row: _f(row["full_annual"]), reverse=True))
    audit = risk.liq._audit_signals(signal_files)
    (REPORT_DIR / "hard_gate_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(summary, audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
