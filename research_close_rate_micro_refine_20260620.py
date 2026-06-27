from __future__ import annotations

import csv
import json
from pathlib import Path

from research_confirm_mv_boundary_refine_20260620 import (
    _filtered_rows,
    _result_row,
    _run_backtest,
    _write_signal,
)
from research_list_age_current_best_probe_20260620 import _load_rows, _safe


REPORT_DIR = (
    Path(__file__).resolve().parents[2]
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_model_application_20260620"
    / "latest_10d_close_rate_micro_refine_20260620"
)


def main() -> None:
    REPORT_DIR.joinpath("signals").mkdir(parents=True, exist_ok=True)
    REPORT_DIR.joinpath("logs").mkdir(parents=True, exist_ok=True)
    rows = _load_rows()
    grid = [
        {
            "confirm_rank_min": 0.50,
            "min_close_rate": min_close_rate,
            "max_close_rate": max_close_rate,
            "max_total_mv": 200000.0,
            "top_k": 5,
            "holding_days": 5,
            "target_total_pct": 0.98,
            "open_daily_score_exit": 0,
            "max_daily_sells": 1,
        }
        for min_close_rate in [0.9625, 0.9650, 0.9675, 0.9700]
        for max_close_rate in [1.0850, 1.0900, 1.0950, 1.1000]
    ]
    results = []
    for params in grid:
        label = f"cr{_safe(params['min_close_rate'])}_{_safe(params['max_close_rate'])}"
        signal_file = REPORT_DIR / "signals" / f"{label}.csv"
        log_file = REPORT_DIR / "logs" / f"{label}.log"
        if not log_file.exists():
            filtered = _filtered_rows(
                rows,
                confirm_rank_min=float(params["confirm_rank_min"]),
                min_close_rate=float(params["min_close_rate"]),
                max_close_rate=float(params["max_close_rate"]),
            )
            _write_signal(filtered, signal_file, max_total_mv=float(params["max_total_mv"]), top_k=5)
            returncode = _run_backtest(signal_file, log_file)
        else:
            returncode = 0
        row = _result_row(params, signal_file, log_file, returncode)
        results.append(row)
        print(json.dumps(row, ensure_ascii=False, default=str))

    fieldnames = list(results[0].keys())
    outputs = {
        "summary.csv": results,
        "summary_by_sharpe.csv": sorted(results, key=lambda row: float(row.get("sharpe") or -999), reverse=True),
        "summary_by_annual.csv": sorted(results, key=lambda row: float(row.get("annual") or -999), reverse=True),
        "summary_qualified_annual2_avg80_by_sharpe.csv": [
            row
            for row in sorted(results, key=lambda item: float(item.get("sharpe") or -999), reverse=True)
            if float(row.get("annual") or -999) >= 2.0 and float(row.get("avg_invested_pct") or -999) >= 0.80
        ],
    }
    for name, data in outputs.items():
        with (REPORT_DIR / name).open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)


if __name__ == "__main__":
    main()
