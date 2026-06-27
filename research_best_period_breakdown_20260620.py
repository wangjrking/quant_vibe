from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import os
import re
import subprocess
from pathlib import Path

from research_list_age_current_best_probe_20260620 import JUEJIN_PYTHON, MAIN, MARKET_DB, PRED_DB, STRATEGY_DIR


ROOT = Path(__file__).resolve().parents[2]
BASE_REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_model_application_20260620"
REPORT_DIR = BASE_REPORT_DIR / "latest_10d_best_period_breakdown_20260620"
SIGNAL_FILE = (
    BASE_REPORT_DIR
    / "latest_10d_confirm5d_close_rate_refine_20260620"
    / "signals"
    / "tk5_h5_mv200_5d50_cr0p965_1p085.csv"
)
TABLE_10D = "stock_predict_data_model_agent_toprank_latestfactor_20260618_executable_10d_open_return_score_20240604_20260618"


PERIODS = [
    ("2024H2", "2024-06-05 09:00:00", "2024-12-31 15:30:00"),
    ("2025H1", "2025-01-01 09:00:00", "2025-06-30 15:30:00"),
    ("2025H2", "2025-07-01 09:00:00", "2025-12-31 15:30:00"),
    ("2026H1", "2026-01-01 09:00:00", "2026-06-25 15:30:00"),
]


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
    pattern = re.compile(r"invested_pct=([0-9.]+).*active_positions=(\d+)")
    if not log_file.exists():
        return {"avg_invested_pct": None, "ge80_ratio": None, "exposure_points": 0}
    for line in log_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if match:
            values.append(float(match.group(1)))
    if not values:
        return {"avg_invested_pct": None, "ge80_ratio": None, "exposure_points": 0}
    return {
        "avg_invested_pct": sum(values) / len(values),
        "ge80_ratio": sum(1 for value in values if value >= 0.80) / len(values),
        "exposure_points": len(values),
    }


def _run(period: str, start: str, end: str, log_file: Path) -> int:
    env = os.environ.copy()
    env.update(
        {
            "GM_OPEN_DAILY_SCORE_EXIT": "0",
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
        "5",
        "--target-position-pct",
        "0.98",
        "--score-db",
        str(PRED_DB),
        "--score-table",
        TABLE_10D,
        "--market-db",
        str(MARKET_DB),
        "--backtest-start",
        start,
        "--backtest-end",
        end,
        "--backtest-adjust",
        "none",
        "--backtest-initial-cash",
        "600000",
        "--backtest-slippage-ratio",
        "0.0015",
    ]
    proc = subprocess.run(command, cwd=str(MAIN), env=env, capture_output=True, text=True)
    if proc.stdout.strip():
        try:
            payload = json.loads(proc.stdout.strip().splitlines()[-1])
            return int(payload.get("returncode", proc.returncode))
        except Exception:
            pass
    return proc.returncode


def main() -> None:
    REPORT_DIR.joinpath("logs").mkdir(parents=True, exist_ok=True)
    results = []
    for period, start, end in PERIODS:
        log_file = REPORT_DIR / "logs" / f"{period}.log"
        returncode = 0 if log_file.exists() else _run(period, start, end, log_file)
        indicator = _extract_indicator(log_file)
        row = {
            "period": period,
            "start": start,
            "end": end,
            "returncode": returncode,
            "signal_file": str(SIGNAL_FILE),
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
        results.append(row)
        print(json.dumps(row, ensure_ascii=False, default=str))
    fieldnames = list(results[0].keys())
    with (REPORT_DIR / "summary.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
