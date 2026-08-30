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
SOURCE_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "new5d_confirm_trade_param_grid_20260621"
)
SOURCE_SIGNAL = SOURCE_DIR / "signals" / "c0p5_cr0p965_1p085_tk5_h5_mp6_cap0p6.csv"
REPORT_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "new5d_dynamic_holding_probe_v2_20260621"
)

CONFIGS = [
    {"name": "base_h5_copy", "hold": 5, "metric": "all", "op": "all", "threshold": None},
    {"name": "h6_count_lt5", "hold": 6, "metric": "count", "op": "lt", "threshold": 5},
    {"name": "h7_count_lt5", "hold": 7, "metric": "count", "op": "lt", "threshold": 5},
    {"name": "h6_top_lt0p3", "hold": 6, "metric": "top", "op": "lt", "threshold": 0.30},
    {"name": "h7_top_lt0p3", "hold": 7, "metric": "top", "op": "lt", "threshold": 0.30},
    {"name": "h6_top_ge0p3", "hold": 6, "metric": "top", "op": "ge", "threshold": 0.30},
    {"name": "h7_top_ge0p3", "hold": 7, "metric": "top", "op": "ge", "threshold": 0.30},
    {"name": "h6_top_ge0p4", "hold": 6, "metric": "top", "op": "ge", "threshold": 0.40},
    {"name": "h7_top_ge0p4", "hold": 7, "metric": "top", "op": "ge", "threshold": 0.40},
    {"name": "h6_mean_lt0p08", "hold": 6, "metric": "mean", "op": "lt", "threshold": 0.08},
    {"name": "h7_mean_lt0p08", "hold": 7, "metric": "mean", "op": "lt", "threshold": 0.08},
    {"name": "h6_mean_ge0p08", "hold": 6, "metric": "mean", "op": "ge", "threshold": 0.08},
    {"name": "h7_mean_ge0p08", "hold": 7, "metric": "mean", "op": "ge", "threshold": 0.08},
]


def _load_rows() -> tuple[list[dict], list[str]]:
    with SOURCE_SIGNAL.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader), list(reader.fieldnames or [])


def _day_stats(rows: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        try:
            value = float(row.get("pred_prob") or 0.0)
        except (TypeError, ValueError):
            continue
        grouped.setdefault(str(row["signal_date"]), []).append(value)
    stats = {}
    for day, values in grouped.items():
        stats[day] = {
            "count": len(values),
            "top": max(values),
            "mean": sum(values) / len(values),
        }
    return stats


def _match(stat: dict, config: dict) -> bool:
    if config["op"] == "all":
        return False
    value = stat.get(config["metric"])
    threshold = config["threshold"]
    if value is None or threshold is None:
        return False
    if config["op"] == "lt":
        return float(value) < float(threshold)
    if config["op"] == "ge":
        return float(value) >= float(threshold)
    raise ValueError(f"unsupported op: {config['op']}")


def _write_signal(config: dict) -> tuple[Path, int]:
    signal_file = REPORT_DIR / "signals" / f"{config['name']}.csv"
    if signal_file.exists() and signal_file.stat().st_size > 0:
        rows, _fields = _load_rows()
        stats = _day_stats(rows)
        extended_days = sum(1 for stat in stats.values() if _match(stat, config))
        return signal_file, extended_days
    rows, fieldnames = _load_rows()
    stats = _day_stats(rows)
    extended_days = 0
    for row in rows:
        if _match(stats.get(str(row["signal_date"]), {}), config):
            row["holding_days"] = str(config["hold"])
    extended_days = sum(1 for stat in stats.values() if _match(stat, config))
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
        "7",
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
