from __future__ import annotations

import csv
import json
from pathlib import Path

from research_confirm_mv_boundary_refine_20260620 import (
    _result_row,
    _run_backtest,
    _write_signal,
)
from research_list_age_current_best_probe_20260620 import _load_rows, _safe, _to_float


REPORT_DIR = (
    Path(__file__).resolve().parents[2]
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_model_application_20260620"
    / "latest_10d_liquidity_filter_probe_20260620"
)


def _filtered_rows(
    rows: list[dict],
    *,
    min_amount: float | None,
    max_turnover_rate: float | None,
    max_volume_ratio: float | None,
) -> list[dict]:
    selected = []
    for row in rows:
        rank5d = _to_float(row.get("rank5d"))
        if rank5d is None or rank5d < 0.50:
            continue
        close_rate = _to_float(row.get("close_rate"))
        if close_rate is None or close_rate < 0.965 or close_rate > 1.085:
            continue
        amount = _to_float(row.get("amount"))
        if min_amount is not None and (amount is None or amount < min_amount):
            continue
        turnover_rate = _to_float(row.get("turnover_rate"))
        if max_turnover_rate is not None and (turnover_rate is None or turnover_rate > max_turnover_rate):
            continue
        volume_ratio = _to_float(row.get("volume_ratio"))
        if max_volume_ratio is not None and (volume_ratio is None or volume_ratio > max_volume_ratio):
            continue
        selected.append(row)
    return selected


def _stats_for_filter(rows: list[dict]) -> dict:
    return {
        "prefilter_rows": len(rows),
        "prefilter_trade_days": len({row.get("trade_date") for row in rows if row.get("trade_date")}),
    }


def main() -> None:
    REPORT_DIR.joinpath("signals").mkdir(parents=True, exist_ok=True)
    REPORT_DIR.joinpath("logs").mkdir(parents=True, exist_ok=True)
    rows = _load_rows()
    grid = [
        {"min_amount": None, "max_turnover_rate": None, "max_volume_ratio": 1.8},
        {"min_amount": None, "max_turnover_rate": None, "max_volume_ratio": 2.2},
        {"min_amount": None, "max_turnover_rate": None, "max_volume_ratio": 2.8},
        {"min_amount": None, "max_turnover_rate": 6.0, "max_volume_ratio": None},
        {"min_amount": None, "max_turnover_rate": 8.0, "max_volume_ratio": None},
        {"min_amount": None, "max_turnover_rate": 10.0, "max_volume_ratio": None},
        {"min_amount": 5000.0, "max_turnover_rate": None, "max_volume_ratio": None},
        {"min_amount": 10000.0, "max_turnover_rate": None, "max_volume_ratio": None},
        {"min_amount": 20000.0, "max_turnover_rate": None, "max_volume_ratio": None},
        {"min_amount": 5000.0, "max_turnover_rate": None, "max_volume_ratio": 2.2},
        {"min_amount": 10000.0, "max_turnover_rate": None, "max_volume_ratio": 2.2},
        {"min_amount": 5000.0, "max_turnover_rate": 8.0, "max_volume_ratio": None},
        {"min_amount": 10000.0, "max_turnover_rate": 8.0, "max_volume_ratio": None},
        {"min_amount": None, "max_turnover_rate": 8.0, "max_volume_ratio": 2.2},
        {"min_amount": 10000.0, "max_turnover_rate": 8.0, "max_volume_ratio": 2.2},
    ]
    results = []
    for params in grid:
        label = (
            f"amt{_safe(params['min_amount'])}"
            f"_turn{_safe(params['max_turnover_rate'])}"
            f"_vol{_safe(params['max_volume_ratio'])}"
        )
        signal_file = REPORT_DIR / "signals" / f"{label}.csv"
        log_file = REPORT_DIR / "logs" / f"{label}.log"
        filtered = _filtered_rows(rows, **params)
        if not log_file.exists():
            _write_signal(filtered, signal_file, max_total_mv=200000.0, top_k=5)
            returncode = _run_backtest(signal_file, log_file)
        else:
            returncode = 0
        row = {
            **params,
            "confirm_rank_min": 0.50,
            "min_close_rate": 0.965,
            "max_close_rate": 1.085,
            "max_total_mv": 200000.0,
            "top_k": 5,
            "holding_days": 5,
            "target_total_pct": 0.98,
            "open_daily_score_exit": 0,
            "max_daily_sells": 1,
            **_stats_for_filter(filtered),
            **_result_row(params, signal_file, log_file, returncode),
        }
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
