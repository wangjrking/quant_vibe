from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import os
import re
import subprocess
import sys
from pathlib import Path

MAIN = Path(__file__).resolve().parents[3]
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))

from research_list_age_current_best_probe_20260620 import JUEJIN_PYTHON, MAIN, MARKET_DB, PRED_DB, STRATEGY_DIR, TABLE_10D


ROOT = Path(__file__).resolve().parents[5]
REPORT_ROOT_20260620 = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_model_application_20260620"
REPORT_ROOT_20260621 = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_model_application_20260621"
REPORT_DIR = REPORT_ROOT_20260621 / "robustness_5d10d_matrix_20260621"

CANDIDATES = [
    {
        "name": "base_10d_5d50_cr965_mv200",
        "signal_file": REPORT_ROOT_20260620
        / "latest_10d_confirm_mv_boundary_refine_20260620"
        / "signals"
        / "c0p5_mv200_cr0p965_1p085.csv",
    },
    {
        "name": "base_10d_5d45_cr965_mv200",
        "signal_file": REPORT_ROOT_20260620
        / "latest_10d_confirm_mv_boundary_refine_20260620"
        / "signals"
        / "c0p45_mv200_cr0p965_1p085.csv",
    },
    {
        "name": "base_10d_5d55_cr965_mv200",
        "signal_file": REPORT_ROOT_20260620
        / "latest_10d_confirm_mv_boundary_refine_20260620"
        / "signals"
        / "c0p55_mv200_cr0p965_1p085.csv",
    },
    {
        "name": "base_10d_5d60_cr965_mv200",
        "signal_file": REPORT_ROOT_20260620
        / "latest_10d_confirm_mv_boundary_refine_20260620"
        / "signals"
        / "c0p6_mv200_cr0p965_1p085.csv",
    },
    {
        "name": "drop_highscore_gt0p5_no_refill",
        "signal_file": REPORT_ROOT_20260621
        / "core10d_drop_highscore_target_scale_20260621"
        / "signals"
        / "target0p196.csv",
    },
    {
        "name": "cap0p4_refill_10d_5d50",
        "signal_file": REPORT_ROOT_20260621
        / "core10d_highscore_cap_refill_20260621"
        / "signals"
        / "cap0p4_minnone_c0p5_cr0p965_1p085.csv",
    },
    {
        "name": "cap0p5_refill_cr955_10d_5d50",
        "signal_file": REPORT_ROOT_20260621
        / "core10d_highscore_cap_refill_20260621"
        / "signals"
        / "cap0p5_minnone_c0p5_cr0p955_1p085.csv",
    },
]

PERIODS = [
    ("2024H2", "2024-06-05 09:00:00", "2024-12-31 15:30:00"),
    ("2025H1", "2025-01-01 09:00:00", "2025-06-30 15:30:00"),
    ("2025H2", "2025-07-01 09:00:00", "2025-12-31 15:30:00"),
    ("2026H1", "2026-01-01 09:00:00", "2026-06-25 15:30:00"),
]


def _extract_indicator(log_file: Path) -> dict | None:
    if not log_file.exists():
        return None
    for line in reversed(log_file.read_text(encoding="utf-8", errors="ignore").splitlines()):
        if "GM_BACKTEST_INDICATOR:" not in line:
            continue
        payload = line.split("GM_BACKTEST_INDICATOR:", 1)[1].strip()
        try:
            return ast.literal_eval(payload)
        except Exception:
            return eval(payload, {"__builtins__": {}}, {"datetime": datetime_module})
    return None


def _exposure_stats(log_file: Path) -> dict:
    values = []
    active = []
    pattern = re.compile(r"EXPOSURE .* invested_pct=([0-9.]+).*active_positions=(\d+)")
    if not log_file.exists():
        return {"avg_invested_pct": None, "ge80_ratio": None, "max_active_positions": None, "exposure_points": 0}
    for match in pattern.finditer(log_file.read_text(encoding="utf-8", errors="ignore")):
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


def _run_backtest(candidate: dict, period: tuple[str, str, str]) -> tuple[int, Path]:
    period_name, start, end = period
    log_file = REPORT_DIR / "logs" / candidate["name"] / f"{period_name}.log"
    if log_file.exists() and _extract_indicator(log_file):
        return 0, log_file
    log_file.parent.mkdir(parents=True, exist_ok=True)
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
        str(candidate["signal_file"]),
        "--log-file",
        str(log_file),
        "--max-positions",
        "5",
        "--holding-days",
        "5",
        "--max-holding-days",
        "5",
        "--target-position-pct",
        "0.196",
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
    with log_file.open("w", encoding="utf-8") as log:
        proc = subprocess.run(command, cwd=str(MAIN), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    return proc.returncode, log_file


def _signal_count(signal_file: Path) -> int:
    with signal_file.open("r", encoding="utf-8-sig", newline="") as file:
        return sum(1 for _ in csv.DictReader(file))


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for candidate in CANDIDATES:
        signal_file = Path(candidate["signal_file"])
        if not signal_file.exists():
            print(json.dumps({"missing_signal": candidate["name"], "path": str(signal_file)}, ensure_ascii=False))
            continue
        signal_count = _signal_count(signal_file)
        for period in PERIODS:
            period_name, start, end = period
            returncode, log_file = _run_backtest(candidate, period)
            indicator = _extract_indicator(log_file) or {}
            row = {
                "candidate": candidate["name"],
                "period": period_name,
                "backtest_start": start,
                "backtest_end": end,
                "returncode": returncode,
                "signal_file": str(signal_file),
                "signal_count": signal_count,
                "log_file": str(log_file),
                "annual": indicator.get("pnl_ratio_annual"),
                "sharpe": indicator.get("sharp_ratio"),
                "max_drawdown": indicator.get("max_drawdown"),
                "win_ratio": indicator.get("win_ratio"),
                "open_count": indicator.get("open_count"),
                "close_count": indicator.get("close_count"),
                **_exposure_stats(log_file),
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False, default=str), flush=True)
    if not rows:
        raise SystemExit("No robustness rows produced.")
    with (REPORT_DIR / "period_summary.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["candidate"]), []).append(row)
    aggregate = []
    for candidate, items in grouped.items():
        sharpes = [float(row["sharpe"]) for row in items if row.get("sharpe") is not None]
        annuals = [float(row["annual"]) for row in items if row.get("annual") is not None]
        invested = [float(row["avg_invested_pct"]) for row in items if row.get("avg_invested_pct") is not None]
        aggregate.append(
            {
                "candidate": candidate,
                "period_count": len(items),
                "min_period_annual": min(annuals) if annuals else None,
                "median_period_annual": sorted(annuals)[len(annuals) // 2] if annuals else None,
                "min_period_sharpe": min(sharpes) if sharpes else None,
                "median_period_sharpe": sorted(sharpes)[len(sharpes) // 2] if sharpes else None,
                "min_avg_invested_pct": min(invested) if invested else None,
                "median_avg_invested_pct": sorted(invested)[len(invested) // 2] if invested else None,
            }
        )
    with (REPORT_DIR / "robustness_summary.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(aggregate[0].keys()))
        writer.writeheader()
        writer.writerows(sorted(aggregate, key=lambda row: float(row.get("min_period_sharpe") or -999), reverse=True))


if __name__ == "__main__":
    main()
