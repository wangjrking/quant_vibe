from __future__ import annotations

import csv
import json

import tune_strict_sync_dd_overlay_20260624 as dd_base


base = dd_base.base
base.REPORT_DIR = base.SOURCE_DIR / "strict_sync_dd_overlay_soft_20260624"

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
]

DD_CONFIGS = [
    {
        "tag": "dd10_18_s90_75_resize",
        "soft_trigger": 0.10,
        "hard_trigger": 0.18,
        "recover_trigger": 0.05,
        "soft_scale": 0.90,
        "hard_scale": 0.75,
        "resize_existing": 1,
    },
    {
        "tag": "dd12_20_s90_70_resize",
        "soft_trigger": 0.12,
        "hard_trigger": 0.20,
        "recover_trigger": 0.06,
        "soft_scale": 0.90,
        "hard_scale": 0.70,
        "resize_existing": 1,
    },
    {
        "tag": "dd10_16_s85_65_resize",
        "soft_trigger": 0.10,
        "hard_trigger": 0.16,
        "recover_trigger": 0.05,
        "soft_scale": 0.85,
        "hard_scale": 0.65,
        "resize_existing": 1,
    },
]


def main() -> int:
    rows = []
    cases_path = base.REPORT_DIR / "cases.csv"
    if cases_path.exists():
        with cases_path.open("r", encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
    existing = {(row["name"], row["start_name"]) for row in rows}
    cases = [dd_base._make_case(case, dd) for case in BASE_CASES for dd in DD_CONFIGS]
    total = len(cases) * len(base.STARTS)
    done = 0
    for case in cases:
        signal_file = base._write_signal(case)
        for start_name, start in base.STARTS:
            done += 1
            if (case["name"], start_name) in existing:
                print(f"[{done}/{total}] reuse {case['name']} {start_name}", flush=True)
                continue
            row = dd_base._run_case_with_dd(case, signal_file, start_name, start)
            rows.append(row)
            base._write_rows(cases_path, rows)
            print(
                f"[{done}/{total}] {case['name']} {start_name} "
                f"annual={row['annual']} sharpe={row['sharpe']} maxdd={row['max_drawdown']} active={row['max_active_positions']}",
                flush=True,
            )
    summary = base._summarize(rows)
    full_rows = {row["name"]: row for row in rows if row.get("start_name") == "full_20240605"}
    for item in summary:
        full = full_rows.get(item["name"]) or {}
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
