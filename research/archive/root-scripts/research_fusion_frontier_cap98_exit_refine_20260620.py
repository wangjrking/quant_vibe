from __future__ import annotations

import csv
from pathlib import Path

from research_fusion_frontier_runtime_refine_20260620 import (
    REPORT_DIR as BASE_RUNTIME_REPORT_DIR,
    SIGNALS,
    _extract_indicator,
    _exposure_stats,
    _run,
    _safe,
)


REPORT_DIR = BASE_RUNTIME_REPORT_DIR.parent / "fusion_frontier_cap98_exit_refine_20260620"

CONFIGS = []
for signal_name, signal_file in SIGNALS:
    for ratio in (0.85, 0.88, 0.90, 0.92, 0.95, 0.98, 1.00, 1.02):
        for min_hold in (1, 2, 3):
            CONFIGS.append(
                {
                    "signal_name": signal_name,
                    "signal_file": signal_file,
                    "target_cap": 0.98,
                    "open_daily_score_exit": 1,
                    "score_exit_entry_ratio": ratio,
                    "min_hold": min_hold,
                }
            )
    CONFIGS.append(
        {
            "signal_name": signal_name,
            "signal_file": signal_file,
            "target_cap": 0.98,
            "open_daily_score_exit": 0,
            "score_exit_entry_ratio": None,
            "min_hold": 2,
        }
    )


def main() -> None:
    REPORT_DIR.joinpath("logs").mkdir(parents=True, exist_ok=True)
    rows = []
    for config in CONFIGS:
        name = (
            f"{config['signal_name']}_cap0p98_"
            f"exit{config['open_daily_score_exit']}_ratio{_safe(config['score_exit_entry_ratio'])}_"
            f"mh{config['min_hold']}"
        )
        log_file = REPORT_DIR / "logs" / f"{name}.log"
        returncode = 0 if log_file.exists() and log_file.stat().st_size > 0 else _run(config, log_file)
        indicator = _extract_indicator(log_file)
        row = {
            "name": name,
            "returncode": returncode,
            "signal_name": config["signal_name"],
            "signal_file": str(Path(config["signal_file"])),
            "target_cap": config["target_cap"],
            "open_daily_score_exit": config["open_daily_score_exit"],
            "score_exit_entry_ratio": config["score_exit_entry_ratio"],
            "min_hold": config["min_hold"],
            "log_file": str(log_file),
            **_exposure_stats(log_file),
        }
        if indicator:
            row.update(
                {
                    "annual": indicator.get("pnl_ratio_annual"),
                    "sharpe": indicator.get("sharp_ratio"),
                    "max_drawdown": indicator.get("max_drawdown"),
                    "open_count": indicator.get("open_count"),
                    "close_count": indicator.get("close_count"),
                    "win_ratio": indicator.get("win_ratio"),
                    "calmar_ratio": indicator.get("calmar_ratio"),
                }
            )
        rows.append(row)
        print(row)

    fieldnames = list(rows[0].keys())
    with (REPORT_DIR / "summary.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    qualified = [
        row
        for row in rows
        if row.get("annual") is not None
        and float(row["annual"]) > 2.0
        and row.get("avg_invested_pct") is not None
        and float(row["avg_invested_pct"]) >= 0.80
    ]
    with (REPORT_DIR / "summary_qualified_annual2_avg80_by_sharpe.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(qualified, key=lambda row: float(row["sharpe"]), reverse=True))


if __name__ == "__main__":
    main()
