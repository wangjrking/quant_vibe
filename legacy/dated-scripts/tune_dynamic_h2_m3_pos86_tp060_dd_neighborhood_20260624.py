from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

import tune_dynamic_h2_m3_tp060_pos84_dd_neighborhood_20260624 as dd


BASE_DIR = dd.BASE_DIR
REPORT_DIR = BASE_DIR / "dynamic_h2_m3_pos86_tp060_dd_neighborhood_20260624"
BASE_SIGNAL_FILE = BASE_DIR / "dynamic_h2_m3_dd0714_tp060_high_position_20260624" / "signals" / "dd0714_tp060_pos86.csv"

TARGET_PCT = 0.86
DD_CASES = [
    {
        "case_name": "tp060_pos86_dd0509_s7050",
        "risk_mode": 1,
        "soft": 0.05,
        "hard": 0.09,
        "recover": 0.03,
        "soft_scale": 0.70,
        "hard_scale": 0.50,
    },
    {
        "case_name": "tp060_pos86_dd0510_s8060",
        "risk_mode": 1,
        "soft": 0.05,
        "hard": 0.10,
        "recover": 0.03,
        "soft_scale": 0.80,
        "hard_scale": 0.60,
    },
    {
        "case_name": "tp060_pos86_dd0612_s8060",
        "risk_mode": 1,
        "soft": 0.06,
        "hard": 0.12,
        "recover": 0.03,
        "soft_scale": 0.80,
        "hard_scale": 0.60,
    },
    {
        "case_name": "tp060_pos86_dd0612_s8565",
        "risk_mode": 1,
        "soft": 0.06,
        "hard": 0.12,
        "recover": 0.03,
        "soft_scale": 0.85,
        "hard_scale": 0.65,
    },
    {
        "case_name": "tp060_pos86_dd0714_s8060",
        "risk_mode": 1,
        "soft": 0.07,
        "hard": 0.14,
        "recover": 0.04,
        "soft_scale": 0.80,
        "hard_scale": 0.60,
    },
    {
        "case_name": "tp060_pos86_dd0814_s8565",
        "risk_mode": 1,
        "soft": 0.08,
        "hard": 0.14,
        "recover": 0.04,
        "soft_scale": 0.85,
        "hard_scale": 0.65,
    },
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
        row["signal_stop_loss_pct"] = f"{dd.STOP_LOSS:.5f}"
        row["signal_take_profit_pct"] = f"{dd.TAKE_PROFIT:.5f}"
        row["strategy_variant"] = f"dynamic_h2_m3_pos86_{case['case_name']}"
    _write_rows(path, rows)
    return path


def _summarize(detail: list[dict], signal_stats: dict[str, dict]) -> list[dict]:
    out = []
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
                "dd_soft": first["dd_soft"],
                "dd_hard": first["dd_hard"],
                "dd_recover": first["dd_recover"],
                "dd_soft_scale": first["dd_soft_scale"],
                "dd_hard_scale": first["dd_hard_scale"],
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
    ranked = sorted(summary, key=lambda row: (_f(row["full_max_drawdown"]) <= 0.40, _f(row["full_annual"])), reverse=True)
    lines = [
        "# pos86 回撤缩放邻域验证",
        "",
        "## 当前结论",
        "",
        "本轮固定 pos86 的选股、止损、止盈、持有和退出规则，只调整账户回撤缩放参数。目的不是追求最高年化，而是检查当前 39.94% 最大回撤是否能从强准入红线附近拉开。",
        "",
        "| 候选 | 回撤参数 | 年化 | Sharpe | 最大回撤 | recent60 年化 | 近期开仓最差年化 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ranked:
        risk_text = (
            f"soft={_f(row['dd_soft']):.0%}, hard={_f(row['dd_hard']):.0%}, "
            f"scale={_f(row['dd_soft_scale']):.0%}/{_f(row['dd_hard_scale']):.0%}"
        )
        lines.append(
            f"| {row['case_name']} | {risk_text} | {_f(row['full_annual']):.2%} | "
            f"{_f(row['full_sharpe']):.2f} | {_f(row['full_max_drawdown']):.2%} | "
            f"{_f(row['recent60_annual']):.2%} | {_f(row['late_min_annual']):.2%} |"
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
        ]
    )
    (REPORT_DIR / "dd_neighborhood_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    detail: list[dict] = []
    signal_files: list[Path] = []
    signal_stats: dict[str, dict] = {}
    dd.REPORT_DIR = REPORT_DIR
    dd.BASE_SIGNAL_FILE = BASE_SIGNAL_FILE
    dd.TARGET_PCT = TARGET_PCT
    for case in DD_CASES:
        signal_file = _signal_for_case(case)
        signal_files.append(signal_file)
        rows = _read_rows(signal_file)
        signal_stats[case["case_name"]] = {
            "signal_rows": len(rows),
            "signal_days": len({row["signal_date"] for row in rows}),
            "latest_signal_date": max((row["signal_date"] for row in rows), default=None),
            "latest_buy_date": max((row["buy_date"] for row in rows), default=None),
        }
        for start_name, start, end in dd.STARTS:
            row = dd._run(case, signal_file, start_name, start, end)
            detail.append(row)
            print(f"{case['case_name']} {start_name} annual={row['annual']} sharpe={row['sharpe']} maxdd={row['max_drawdown']}", flush=True)
    summary = _summarize(detail, signal_stats)
    _write_rows(REPORT_DIR / "detail.csv", detail)
    _write_rows(REPORT_DIR / "summary.csv", summary)
    _write_rows(REPORT_DIR / "summary_by_annual.csv", sorted(summary, key=lambda row: _f(row["full_annual"]), reverse=True))
    audit = dd.liq._audit_signals(signal_files)
    (REPORT_DIR / "hard_gate_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(summary, audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
