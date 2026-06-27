from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import os
import re
import subprocess
from pathlib import Path

from research_list_age_current_best_probe_20260620 import JUEJIN_PYTHON, MAIN, MARKET_DB, PRED_DB, STRATEGY_DIR, TABLE_10D


ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_model_application_20260621"
SOURCE_SIGNAL = REPORT_ROOT / "core10d_original_signal_score_cut_20260621" / "signals" / "drop_gt0p5_all.csv"
REPORT_DIR = REPORT_ROOT / "core10d_drop_highscore_target_scale_20260621"

CONFIGS = [
    {"name": "target0p196", "target_pct": 0.196},
    {"name": "target0p205", "target_pct": 0.205},
    {"name": "target0p210", "target_pct": 0.210},
    {"name": "target0p220", "target_pct": 0.220},
    {"name": "target0p235", "target_pct": 0.235},
    {"name": "target0p250", "target_pct": 0.250},
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
    active_positions = []
    pattern = re.compile(r"EXPOSURE .* invested_pct=([0-9.]+).*active_positions=(\d+)")
    if not log_file.exists():
        return {"avg_invested_pct": None, "ge80_ratio": None, "max_active_positions": None, "exposure_points": 0}
    for match in pattern.finditer(log_file.read_text(encoding="utf-8", errors="ignore")):
        values.append(float(match.group(1)))
        active_positions.append(int(match.group(2)))
    if not values:
        return {"avg_invested_pct": None, "ge80_ratio": None, "max_active_positions": None, "exposure_points": 0}
    return {
        "avg_invested_pct": sum(values) / len(values),
        "ge80_ratio": sum(1 for value in values if value >= 0.80) / len(values),
        "max_active_positions": max(active_positions) if active_positions else None,
        "exposure_points": len(values),
    }


def _write_scaled_signal(config: dict) -> Path:
    signal_file = REPORT_DIR / "signals" / f"{config['name']}.csv"
    if signal_file.exists() and signal_file.stat().st_size > 0:
        return signal_file
    signal_file.parent.mkdir(parents=True, exist_ok=True)
    with SOURCE_SIGNAL.open("r", encoding="utf-8-sig", newline="") as src:
        rows = list(csv.DictReader(src))
        fieldnames = list(rows[0].keys())
    for row in rows:
        row["target_pct"] = str(config["target_pct"])
    with signal_file.open("w", encoding="utf-8-sig", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return signal_file


def _run_backtest(config: dict, signal_file: Path) -> tuple[int, Path]:
    log_file = REPORT_DIR / "logs" / f"{config['name']}.log"
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
        str(config["target_pct"]),
        "--score-db",
        str(PRED_DB),
        "--score-table",
        TABLE_10D,
        "--market-db",
        str(MARKET_DB),
        "--backtest-adjust",
        "none",
        "--backtest-initial-cash",
        "600000",
        "--backtest-slippage-ratio",
        "0.0015",
    ]
    with log_file.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            command,
            cwd=str(MAIN),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return proc.returncode, log_file


def _signal_count(signal_file: Path) -> int:
    with signal_file.open("r", encoding="utf-8-sig", newline="") as file:
        return sum(1 for _ in csv.DictReader(file))


def main() -> None:
    if not SOURCE_SIGNAL.exists():
        raise SystemExit(f"Source signal not found: {SOURCE_SIGNAL}")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for config in CONFIGS:
        signal_file = _write_scaled_signal(config)
        returncode, log_file = _run_backtest(config, signal_file)
        indicator = _extract_indicator(log_file) or {}
        row = {
            **config,
            "slippage_ratio": 0.0015,
            "returncode": returncode,
            "signal_file": str(signal_file),
            "signal_count": _signal_count(signal_file),
            "log_file": str(log_file),
            "annual": indicator.get("pnl_ratio_annual"),
            "sharpe": indicator.get("sharp_ratio"),
            "max_drawdown": indicator.get("max_drawdown"),
            "win_ratio": indicator.get("win_ratio"),
            **_exposure_stats(log_file),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False, default=str), flush=True)
    with (REPORT_DIR / "summary.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (REPORT_DIR / "summary_by_sharpe.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: float(row.get("sharpe") or -999), reverse=True))
    qualified = [
        row
        for row in rows
        if float(row.get("annual") or -999) >= 2.0
        and float(row.get("sharpe") or -999) >= 3.0
        and float(row.get("avg_invested_pct") or -999) >= 0.8
    ]
    with (REPORT_DIR / "qualified.json").open("w", encoding="utf-8") as file:
        json.dump(qualified, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
