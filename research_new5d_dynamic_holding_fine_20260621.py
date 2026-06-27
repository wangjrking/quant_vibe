from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import os
import re
import subprocess
from pathlib import Path

from research_list_age_current_best_probe_20260620 import JUEJIN_PYTHON, MAIN, MARKET_DB, PRED_DB, TABLE_10D, STRATEGY_DIR


ROOT = Path(__file__).resolve().parents[2]
SOURCE_SIGNAL = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "new5d_confirm_trade_param_grid_20260621"
    / "signals"
    / "c0p5_cr0p965_1p085_tk5_h5_mp6_cap0p6.csv"
)
REPORT_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "new5d_dynamic_holding_fine_20260621"
)


CONFIGS: list[dict] = []
for threshold in (0.35, 0.40, 0.45, 0.50):
    for hold in (7, 8):
        CONFIGS.append(
            {
                "name": f"h{hold}_top_ge{str(threshold).replace('.', 'p')}_targetbase",
                "hold": hold,
                "threshold": threshold,
                "target_pct": None,
            }
        )
for target_pct in (0.17, 0.18, 0.19):
    CONFIGS.append(
        {
            "name": f"h7_top_ge0p4_target{str(target_pct).replace('.', 'p')}",
            "hold": 7,
            "threshold": 0.40,
            "target_pct": target_pct,
        }
    )
for target_pct in (0.17, 0.18):
    CONFIGS.append(
        {
            "name": f"h8_top_ge0p4_target{str(target_pct).replace('.', 'p')}",
            "hold": 8,
            "threshold": 0.40,
            "target_pct": target_pct,
        }
    )


def _load_rows() -> tuple[list[dict], list[str]]:
    with SOURCE_SIGNAL.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader), list(reader.fieldnames or [])


def _day_top(rows: list[dict]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        try:
            value = float(row.get("pred_prob") or 0.0)
        except (TypeError, ValueError):
            continue
        grouped.setdefault(str(row["signal_date"]), []).append(value)
    return {day: max(values) for day, values in grouped.items() if values}


def _write_signal(config: dict) -> tuple[Path, int]:
    signal_file = REPORT_DIR / "signals" / f"{config['name']}.csv"
    rows, fieldnames = _load_rows()
    tops = _day_top(rows)
    extended_days = sum(1 for value in tops.values() if value >= float(config["threshold"]))
    if signal_file.exists() and signal_file.stat().st_size > 0:
        return signal_file, extended_days
    for row in rows:
        if tops.get(str(row["signal_date"]), -999.0) >= float(config["threshold"]):
            row["holding_days"] = str(config["hold"])
        if config.get("target_pct") is not None:
            row["target_pct"] = str(config["target_pct"])
    signal_file.parent.mkdir(parents=True, exist_ok=True)
    with signal_file.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return signal_file, extended_days


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


def _run(signal_file: Path, log_file: Path) -> int:
    if log_file.exists() and _extract_indicator(log_file):
        return 0
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
            "GM_SCORE_CONTINUE_ENTRY_RATIO": "999",
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
        "6",
        "--holding-days",
        "5",
        "--max-holding-days",
        "8",
        "--target-position-pct",
        str(0.98 / 6.0),
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
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as log:
        proc = subprocess.run(command, cwd=str(MAIN), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    return proc.returncode


def _signal_count(signal_file: Path) -> int:
    with signal_file.open("r", encoding="utf-8-sig", newline="") as file:
        return sum(1 for _ in csv.DictReader(file))


def main() -> None:
    if not SOURCE_SIGNAL.exists():
        raise SystemExit(f"Missing source signal: {SOURCE_SIGNAL}")
    REPORT_DIR.joinpath("signals").mkdir(parents=True, exist_ok=True)
    REPORT_DIR.joinpath("logs").mkdir(parents=True, exist_ok=True)
    results = []
    for config in CONFIGS:
        signal_file, extended_days = _write_signal(config)
        log_file = REPORT_DIR / "logs" / f"{config['name']}.log"
        returncode = _run(signal_file, log_file)
        indicator = _extract_indicator(log_file) or {}
        result = {
            **config,
            "extended_days": extended_days,
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
        results.append(result)
        print(json.dumps(result, ensure_ascii=False, default=str), flush=True)
    fieldnames = list(results[0].keys())
    for filename, data in {
        "summary.csv": results,
        "summary_by_sharpe.csv": sorted(results, key=lambda row: float(row.get("sharpe") or -999), reverse=True),
        "summary_qualified_target_by_sharpe.csv": [
            row
            for row in sorted(results, key=lambda item: float(item.get("sharpe") or -999), reverse=True)
            if float(row.get("annual") or -999) >= 2.0
            and float(row.get("sharpe") or -999) >= 3.0
            and float(row.get("avg_invested_pct") or -999) >= 0.80
        ],
    }.items():
        with (REPORT_DIR / filename).open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)


if __name__ == "__main__":
    main()
