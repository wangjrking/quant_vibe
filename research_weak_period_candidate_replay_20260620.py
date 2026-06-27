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
REPORT_DIR = BASE_REPORT_DIR / "latest_10d_weak_period_candidate_replay_20260620"
TABLE_10D = "stock_predict_data_model_agent_toprank_latestfactor_20260618_executable_10d_open_return_score_20240604_20260618"

PERIODS = [
    ("2025H1", "2025-01-01 09:00:00", "2025-06-30 15:30:00"),
    ("2026H1", "2026-01-01 09:00:00", "2026-06-25 15:30:00"),
]

CANDIDATES = [
    (
        "baseline_cr0p965_1p085",
        BASE_REPORT_DIR
        / "latest_10d_confirm5d_close_rate_refine_20260620"
        / "signals"
        / "tk5_h5_mv200_5d50_cr0p965_1p085.csv",
    ),
    (
        "baseline_cr0p965_1p09",
        BASE_REPORT_DIR
        / "latest_10d_confirm5d_close_rate_refine_20260620"
        / "signals"
        / "tk5_h5_mv200_5d50_cr0p965_1p09.csv",
    ),
    (
        "cr0p9625_1p085",
        BASE_REPORT_DIR / "latest_10d_close_rate_micro_refine_20260620" / "signals" / "cr0p9625_1p085.csv",
    ),
    (
        "cr0p9675_1p085",
        BASE_REPORT_DIR / "latest_10d_close_rate_micro_refine_20260620" / "signals" / "cr0p9675_1p085.csv",
    ),
    (
        "cr0p97_1p085",
        BASE_REPORT_DIR / "latest_10d_close_rate_micro_refine_20260620" / "signals" / "cr0p97_1p085.csv",
    ),
    (
        "confirm45_mv200",
        BASE_REPORT_DIR / "latest_10d_confirm_mv_boundary_refine_20260620" / "signals" / "c0p45_mv200_cr0p965_1p085.csv",
    ),
    (
        "confirm55_mv200",
        BASE_REPORT_DIR / "latest_10d_confirm_mv_boundary_refine_20260620" / "signals" / "c0p55_mv200_cr0p965_1p085.csv",
    ),
    (
        "confirm50_mv180",
        BASE_REPORT_DIR / "latest_10d_confirm_mv_boundary_refine_20260620" / "signals" / "c0p5_mv180_cr0p965_1p085.csv",
    ),
    (
        "confirm50_mv250",
        BASE_REPORT_DIR / "latest_10d_confirm_mv_boundary_refine_20260620" / "signals" / "c0p5_mv250_cr0p965_1p085.csv",
    ),
    (
        "volume_ratio_2p8",
        BASE_REPORT_DIR / "latest_10d_liquidity_filter_probe_20260620" / "signals" / "amtnone_turnnone_vol2p8.csv",
    ),
    (
        "amount_10000",
        BASE_REPORT_DIR / "latest_10d_liquidity_filter_probe_20260620" / "signals" / "amt10000p0_turnnone_volnone.csv",
    ),
    (
        "amount_20000",
        BASE_REPORT_DIR / "latest_10d_liquidity_filter_probe_20260620" / "signals" / "amt20000p0_turnnone_volnone.csv",
    ),
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
    active_positions = []
    pattern = re.compile(r"invested_pct=([0-9.]+).*active_positions=(\d+)")
    if not log_file.exists():
        return {
            "avg_invested_pct": None,
            "ge80_ratio": None,
            "max_active_positions": None,
            "exposure_points": 0,
        }
    for line in log_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        values.append(float(match.group(1)))
        active_positions.append(int(match.group(2)))
    if not values:
        return {
            "avg_invested_pct": None,
            "ge80_ratio": None,
            "max_active_positions": None,
            "exposure_points": 0,
        }
    return {
        "avg_invested_pct": sum(values) / len(values),
        "ge80_ratio": sum(1 for value in values if value >= 0.80) / len(values),
        "max_active_positions": max(active_positions) if active_positions else None,
        "exposure_points": len(values),
    }


def _count_signals(signal_file: Path) -> dict:
    count = 0
    buy_dates = set()
    with signal_file.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            count += 1
            if row.get("buy_date"):
                buy_dates.add(str(row["buy_date"]))
    return {"signal_count": count, "buy_days": len(buy_dates)}


def _run(candidate: str, signal_file: Path, period: str, start: str, end: str, log_file: Path) -> int:
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
        str(signal_file),
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
    if proc.returncode != 0:
        print(f"{candidate} {period} failed: {proc.stderr.strip()}")
    return proc.returncode


def main() -> None:
    REPORT_DIR.joinpath("logs").mkdir(parents=True, exist_ok=True)
    rows = []
    for candidate, signal_file in CANDIDATES:
        if not signal_file.exists():
            raise SystemExit(f"Missing signal file for {candidate}: {signal_file}")
        signal_stats = _count_signals(signal_file)
        for period, start, end in PERIODS:
            log_file = REPORT_DIR / "logs" / f"{candidate}_{period}.log"
            returncode = 0 if log_file.exists() else _run(candidate, signal_file, period, start, end, log_file)
            indicator = _extract_indicator(log_file)
            row = {
                "candidate": candidate,
                "period": period,
                "start": start,
                "end": end,
                "returncode": returncode,
                "signal_file": str(signal_file),
                "log_file": str(log_file),
                **signal_stats,
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
            print(json.dumps(row, ensure_ascii=False, default=str))

    summary_file = REPORT_DIR / "summary.csv"
    fieldnames = list(rows[0].keys())
    with summary_file.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    best_by_period = []
    for period, _, _ in PERIODS:
        period_rows = [row for row in rows if row["period"] == period and row.get("sharpe") is not None]
        best_by_period.extend(sorted(period_rows, key=lambda row: float(row["sharpe"]), reverse=True)[:5])
    with (REPORT_DIR / "best_by_period.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(best_by_period)


if __name__ == "__main__":
    main()
