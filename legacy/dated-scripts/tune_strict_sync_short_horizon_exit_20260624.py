from __future__ import annotations

import csv
import json

import tune_strict_sync_exit_reentry_20260624 as base


base.REPORT_DIR = base.SOURCE_DIR / "strict_sync_short_horizon_exit_20260624"

SOURCE = "div_top1_w90_5d10_h5_e099"

GRID = []
for target_pct in [0.90, 0.99]:
    for holding_days in [2, 4]:
        for exit_ratio in [0.95, 0.96, 0.97, 0.98]:
            GRID.append(
                {
                    "name": base._case_name(SOURCE, target_pct, holding_days, exit_ratio, 1),
                    "source": SOURCE,
                    "target_pct": target_pct,
                    "holding_days": holding_days,
                    "exit_ratio": exit_ratio,
                    "min_hold": 1,
                }
            )


def main() -> int:
    rows = []
    cases_path = base.REPORT_DIR / "cases.csv"
    if cases_path.exists():
        with cases_path.open("r", encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
    existing = {(row["name"], row["start_name"]) for row in rows}
    total = len(GRID) * len(base.STARTS)
    done = 0
    for case in GRID:
        signal_file = base._write_signal(case)
        for start_name, start in base.STARTS:
            done += 1
            if (case["name"], start_name) in existing:
                print(f"[{done}/{total}] reuse {case['name']} {start_name}", flush=True)
                continue
            row = base._run_case(case, signal_file, start_name, start)
            rows.append(row)
            base._write_rows(cases_path, rows)
            print(
                f"[{done}/{total}] {case['name']} {start_name} "
                f"annual={row['annual']} sharpe={row['sharpe']} maxdd={row['max_drawdown']} active={row['max_active_positions']}",
                flush=True,
            )
    summary = base._summarize(rows)
    base._write_rows(base.REPORT_DIR / "summary.csv", summary)
    base._write_rows(base.REPORT_DIR / "summary_by_annual.csv", sorted(summary, key=lambda row: base._f(row["full_annual"]), reverse=True))
    (base.REPORT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
