from __future__ import annotations

import csv
import json
from pathlib import Path

import tune_strict_sync_exit_reentry_20260624 as base


base.REPORT_DIR = base.SOURCE_DIR / "strict_sync_dd_overlay_20260624"

BASE_CASES = [
    {
        "source": "div_top1_w90_5d10_h5_e099",
        "target_pct": 0.99,
        "holding_days": 3,
        "exit_ratio": 0.97,
        "min_hold": 1,
    },
    {
        "source": "div_top1_w90_5d10_h5_e099",
        "target_pct": 0.90,
        "holding_days": 3,
        "exit_ratio": 0.97,
        "min_hold": 1,
    },
    {
        "source": "div_top1_w90_5d10_h5_e099",
        "target_pct": 0.75,
        "holding_days": 3,
        "exit_ratio": 0.97,
        "min_hold": 1,
    },
]

DD_CONFIGS = [
    {
        "tag": "dd06_10_s75_50_noresize",
        "soft_trigger": 0.06,
        "hard_trigger": 0.10,
        "recover_trigger": 0.03,
        "soft_scale": 0.75,
        "hard_scale": 0.50,
        "resize_existing": 0,
    },
    {
        "tag": "dd06_10_s75_50_resize",
        "soft_trigger": 0.06,
        "hard_trigger": 0.10,
        "recover_trigger": 0.03,
        "soft_scale": 0.75,
        "hard_scale": 0.50,
        "resize_existing": 1,
    },
    {
        "tag": "dd08_14_s85_65_resize",
        "soft_trigger": 0.08,
        "hard_trigger": 0.14,
        "recover_trigger": 0.04,
        "soft_scale": 0.85,
        "hard_scale": 0.65,
        "resize_existing": 1,
    },
]


def _case_name(case: dict, dd: dict) -> str:
    pct = int(round(case["target_pct"] * 100))
    er = int(round(case["exit_ratio"] * 100))
    return f"dd_{case['source']}_pos{pct:02d}_h{case['holding_days']}_e{er:03d}_{dd['tag']}"


def _make_case(case: dict, dd: dict) -> dict:
    out = dict(case)
    out["name"] = _case_name(case, dd)
    out["dd_config"] = dd
    return out


def _run_case_with_dd(case: dict, signal_file: Path, start_name: str, start: str) -> dict:
    old_env = dict(base.BASE_ENV)
    dd = case["dd_config"]
    base.BASE_ENV.update(
        {
            "GM_EQUITY_DD_RISK_MODE": "1",
            "GM_EQUITY_DD_SOFT_TRIGGER": str(dd["soft_trigger"]),
            "GM_EQUITY_DD_HARD_TRIGGER": str(dd["hard_trigger"]),
            "GM_EQUITY_DD_RECOVER_TRIGGER": str(dd["recover_trigger"]),
            "GM_EQUITY_DD_SOFT_SCALE": str(dd["soft_scale"]),
            "GM_EQUITY_DD_HARD_SCALE": str(dd["hard_scale"]),
            "GM_EQUITY_DD_RESIZE_EXISTING": str(dd["resize_existing"]),
        }
    )
    try:
        row = base._run_case(case, signal_file, start_name, start)
    finally:
        base.BASE_ENV.clear()
        base.BASE_ENV.update(old_env)
    row.update(
        {
            "dd_tag": dd["tag"],
            "dd_soft_trigger": dd["soft_trigger"],
            "dd_hard_trigger": dd["hard_trigger"],
            "dd_recover_trigger": dd["recover_trigger"],
            "dd_soft_scale": dd["soft_scale"],
            "dd_hard_scale": dd["hard_scale"],
            "dd_resize_existing": dd["resize_existing"],
        }
    )
    return row


def main() -> int:
    rows = []
    cases_path = base.REPORT_DIR / "cases.csv"
    if cases_path.exists():
        with cases_path.open("r", encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
    existing = {(row["name"], row["start_name"]) for row in rows}
    cases = [_make_case(case, dd) for case in BASE_CASES for dd in DD_CONFIGS]
    total = len(cases) * len(base.STARTS)
    done = 0
    for case in cases:
        signal_file = base._write_signal(case)
        for start_name, start in base.STARTS:
            done += 1
            if (case["name"], start_name) in existing:
                print(f"[{done}/{total}] reuse {case['name']} {start_name}", flush=True)
                continue
            row = _run_case_with_dd(case, signal_file, start_name, start)
            rows.append(row)
            base._write_rows(cases_path, rows)
            print(
                f"[{done}/{total}] {case['name']} {start_name} "
                f"annual={row['annual']} sharpe={row['sharpe']} maxdd={row['max_drawdown']} active={row['max_active_positions']}",
                flush=True,
            )
    summary = base._summarize(rows)
    by_name = {row["name"]: row for row in rows if row.get("start_name") == "full_20240605"}
    for item in summary:
        full = by_name.get(item["name"]) or {}
        for key in [
            "dd_tag",
            "dd_soft_trigger",
            "dd_hard_trigger",
            "dd_recover_trigger",
            "dd_soft_scale",
            "dd_hard_scale",
            "dd_resize_existing",
        ]:
            item[key] = full.get(key, "")
    base._write_rows(base.REPORT_DIR / "summary.csv", summary)
    base._write_rows(base.REPORT_DIR / "summary_by_annual.csv", sorted(summary, key=lambda row: base._f(row["full_annual"]), reverse=True))
    (base.REPORT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
