from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import os
import re
import subprocess
from pathlib import Path

from research_list_age_current_best_probe_20260620 import (
    JUEJIN_PYTHON,
    MAIN,
    MARKET_DB,
    PRED_DB,
    STRATEGY_DIR,
    TABLE_10D,
    _safe,
)


ROOT = Path(__file__).resolve().parents[2]
BASE_REPORT_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_model_application_20260620"
)
SIGNAL_FILE = (
    BASE_REPORT_DIR
    / "latest_10d_confirm_mv_boundary_refine_20260620"
    / "signals"
    / "c0p5_mv200_cr0p965_1p085.csv"
)
REPORT_DIR = BASE_REPORT_DIR / "latest_10d_exit_continue_probe_20260620"


def _extract_indicator(log_file: Path) -> dict | None:
    marker = "GM_BACKTEST_INDICATOR:"
    if not log_file.exists():
        return None
    for line in reversed(log_file.read_text(encoding="utf-8", errors="ignore").splitlines()):
        if marker not in line:
            continue
        payload = line.split(marker, 1)[1].strip()
        try:
            return ast.literal_eval(payload)
        except Exception:
            return eval(payload, {"__builtins__": {}}, {"datetime": datetime_module})
    return None


def _exposure_stats(log_file: Path) -> dict:
    values = []
    active = []
    pattern = re.compile(r"invested_pct=([0-9.]+).*active_positions=(\d+)")
    for line in log_file.read_text(encoding="utf-8", errors="ignore").splitlines() if log_file.exists() else []:
        match = pattern.search(line)
        if not match:
            continue
        values.append(float(match.group(1)))
        active.append(int(match.group(2)))
    if not values:
        return {"avg_invested_pct": None, "ge80_ratio": None, "max_active_positions": None, "exposure_points": 0}
    return {
        "avg_invested_pct": sum(values) / len(values),
        "ge80_ratio": sum(1 for value in values if value >= 0.80) / len(values),
        "max_active_positions": max(active) if active else None,
        "exposure_points": len(values),
    }


def _signal_stats() -> dict:
    with SIGNAL_FILE.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    return {
        "signal_count": len(rows),
        "buy_days": len({row.get("buy_date") for row in rows if row.get("buy_date")}),
    }


def _run(params: dict, log_file: Path) -> int:
    env = os.environ.copy()
    env.update(
        {
            "GM_OPEN_DAILY_SCORE_EXIT": str(params["open_daily_score_exit"]),
            "GM_MAX_DAILY_SELLS": "1",
            "GM_LOG_EXPOSURE": "1",
            "GM_SKIP_OPEN_LIMIT_UP_BUY": "1",
            "GM_FORCE_MARKET_ORDER": "0",
            "GM_FORCE_BUY_MARKET_ORDER": "0",
            "GM_SYNC_POSITIONS": "0",
            "GM_CASH_BUFFER": "0.995",
            "GM_VERBOSE_TRADES": "0",
        }
    )
    command = [
        str(JUEJIN_PYTHON),
        str(MAIN / "run_juejin_signal_backtest.py"),
        "--strategy-dir",
        str(STRATEGY_DIR),
        "--signal-file",
        str(SIGNAL_FILE),
        "--log-file",
        str(log_file),
        "--max-positions",
        "5",
        "--holding-days",
        "5",
        "--max-holding-days",
        str(params["max_holding_days"]),
        "--target-position-pct",
        "0.98",
        "--score-db",
        str(PRED_DB),
        "--score-table",
        TABLE_10D,
        "--market-db",
        str(MARKET_DB),
        "--score-continue-entry-ratio",
        str(params["score_continue_entry_ratio"]),
        "--backtest-adjust",
        "none",
        "--backtest-initial-cash",
        "600000",
        "--backtest-slippage-ratio",
        "0.0015",
    ]
    if int(params["open_daily_score_exit"]):
        command.extend(
            [
                "--score-exit-entry-ratio",
                str(params["score_exit_entry_ratio"]),
                "--min-holding-days-before-score-exit",
                str(params["min_holding_days_before_score_exit"]),
            ]
        )
    proc = subprocess.run(command, cwd=str(MAIN), env=env, capture_output=True, text=True)
    if proc.stdout.strip():
        try:
            payload = json.loads(proc.stdout.strip().splitlines()[-1])
            return int(payload.get("returncode", proc.returncode))
        except Exception:
            pass
    return proc.returncode


def _row(params: dict, log_file: Path, returncode: int) -> dict:
    indicator = _extract_indicator(log_file)
    row = {
        **params,
        "returncode": returncode,
        "signal_file": str(SIGNAL_FILE),
        "log_file": str(log_file),
        **_signal_stats(),
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
    return row


def main() -> None:
    REPORT_DIR.joinpath("logs").mkdir(parents=True, exist_ok=True)
    grid = [
        {
            "open_daily_score_exit": 0,
            "score_exit_entry_ratio": None,
            "min_holding_days_before_score_exit": None,
            "score_continue_entry_ratio": cont,
            "max_holding_days": max_hold,
        }
        for max_hold in [5, 7, 10]
        for cont in [0.90, 0.95, 1.00, 1.05]
    ]
    grid.extend(
        {
            "open_daily_score_exit": 1,
            "score_exit_entry_ratio": ratio,
            "min_holding_days_before_score_exit": min_hold,
            "score_continue_entry_ratio": 1.00,
            "max_holding_days": 5,
        }
        for min_hold in [1, 2, 3]
        for ratio in [0.85, 0.90, 0.95]
    )
    results = []
    for params in grid:
        label = (
            f"ose{params['open_daily_score_exit']}"
            f"_mh{params['max_holding_days']}"
            f"_cont{_safe(params['score_continue_entry_ratio'])}"
            f"_er{_safe(params['score_exit_entry_ratio'])}"
            f"_emh{_safe(params['min_holding_days_before_score_exit'])}"
        )
        log_file = REPORT_DIR / "logs" / f"{label}.log"
        returncode = 0 if log_file.exists() else _run(params, log_file)
        row = _row(params, log_file, returncode)
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
