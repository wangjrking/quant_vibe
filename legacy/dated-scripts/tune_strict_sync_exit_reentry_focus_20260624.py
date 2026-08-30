from __future__ import annotations

import json

import tune_strict_sync_exit_reentry_20260624 as base


FOCUS_CASES = []
for source in [
    "div_top1_w90_5d10_h5_e099",
    "div_top1_w90_5d10_h5_e098",
    "div_top1_w89_5d11_h5_e099",
]:
    for target_pct in [0.90, 0.99]:
        for holding_days in [3, 5]:
            for exit_ratio in [0.97, 0.99]:
                FOCUS_CASES.append(
                    {
                        "name": base._case_name(source, target_pct, holding_days, exit_ratio, 1),
                        "source": source,
                        "target_pct": target_pct,
                        "holding_days": holding_days,
                        "exit_ratio": exit_ratio,
                        "min_hold": 1,
                    }
                )


def main() -> int:
    rows = []
    if (base.REPORT_DIR / "cases.csv").exists():
        import csv

        with (base.REPORT_DIR / "cases.csv").open("r", encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
    existing = {(row["name"], row["start_name"]) for row in rows}
    total = len(FOCUS_CASES) * len(base.STARTS)
    done = 0
    for case in FOCUS_CASES:
        signal_file = base._write_signal(case)
        for start_name, start in base.STARTS:
            done += 1
            if (case["name"], start_name) in existing:
                print(f"[{done}/{total}] reuse {case['name']} {start_name}", flush=True)
                continue
            row = base._run_case(case, signal_file, start_name, start)
            rows.append(row)
            base._write_rows(base.REPORT_DIR / "cases.csv", rows)
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
