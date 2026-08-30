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
SOURCE_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "new5d_dynamic_holding_fine_20260621"
    / "signals"
)
REPORT_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "new5d_exit_risk_grid_20260621"
)


SIGNALS = {
    "h7_top_ge0p4": SOURCE_DIR / "h7_top_ge0p4_targetbase.csv",
    "h7_top_ge0p45": SOURCE_DIR / "h7_top_ge0p45_targetbase.csv",
}


BASE_CONFIGS = [
    {"risk_name": "base_open0", "open_daily_score_exit": 0, "max_daily_sells": 1},
    {"risk_name": "score0p90_min1", "open_daily_score_exit": 1, "score_exit_entry_ratio": 0.90, "min_score_exit": 1, "max_daily_sells": 1},
    {"risk_name": "score0p90_min2", "open_daily_score_exit": 1, "score_exit_entry_ratio": 0.90, "min_score_exit": 2, "max_daily_sells": 1},
    {"risk_name": "score0p95_min1", "open_daily_score_exit": 1, "score_exit_entry_ratio": 0.95, "min_score_exit": 1, "max_daily_sells": 1},
    {"risk_name": "score0p95_min2", "open_daily_score_exit": 1, "score_exit_entry_ratio": 0.95, "min_score_exit": 2, "max_daily_sells": 1},
    {"risk_name": "score0p90_min1_sells2", "open_daily_score_exit": 1, "score_exit_entry_ratio": 0.90, "min_score_exit": 1, "max_daily_sells": 2},
    {"risk_name": "score0p95_min2_sells2", "open_daily_score_exit": 1, "score_exit_entry_ratio": 0.95, "min_score_exit": 2, "max_daily_sells": 2},
    {"risk_name": "daydrop0p80", "open_daily_score_exit": 1, "score_stop_loss_day_drop_ratio": 0.80, "min_score_exit": 1, "max_daily_sells": 1},
    {"risk_name": "daydrop0p85", "open_daily_score_exit": 1, "score_stop_loss_day_drop_ratio": 0.85, "min_score_exit": 1, "max_daily_sells": 1},
    {"risk_name": "light0p03_min1", "open_daily_score_exit": 0, "light_stop_loss_pct": 0.03, "min_light_stop": 1, "max_daily_sells": 1},
    {"risk_name": "light0p05_min1", "open_daily_score_exit": 0, "light_stop_loss_pct": 0.05, "min_light_stop": 1, "max_daily_sells": 1},
    {"risk_name": "light0p05_min2", "open_daily_score_exit": 0, "light_stop_loss_pct": 0.05, "min_light_stop": 2, "max_daily_sells": 1},
    {"risk_name": "take0p12", "open_daily_score_exit": 0, "take_profit_pct": 0.12, "max_daily_sells": 1},
    {"risk_name": "take0p18", "open_daily_score_exit": 0, "take_profit_pct": 0.18, "max_daily_sells": 1},
    {"risk_name": "take0p25", "open_daily_score_exit": 0, "take_profit_pct": 0.25, "max_daily_sells": 1},
    {
        "risk_name": "score0p90_min1_light0p05",
        "open_daily_score_exit": 1,
        "score_exit_entry_ratio": 0.90,
        "min_score_exit": 1,
        "light_stop_loss_pct": 0.05,
        "min_light_stop": 1,
        "max_daily_sells": 1,
    },
    {
        "risk_name": "score0p95_min2_take0p18",
        "open_daily_score_exit": 1,
        "score_exit_entry_ratio": 0.95,
        "min_score_exit": 2,
        "take_profit_pct": 0.18,
        "max_daily_sells": 1,
    },
    {
        "risk_name": "light0p05_take0p18",
        "open_daily_score_exit": 0,
        "light_stop_loss_pct": 0.05,
        "min_light_stop": 1,
        "take_profit_pct": 0.18,
        "max_daily_sells": 1,
    },
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


def _signal_stats(signal_file: Path) -> dict:
    with signal_file.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    return {
        "signal_count": len(rows),
        "buy_days": len({row.get("buy_date") for row in rows if row.get("buy_date")}),
    }


def _append_arg(command: list[str], option: str, value) -> None:
    if value is None:
        return
    command.extend([option, str(value)])


def _run(signal_file: Path, log_file: Path, config: dict) -> int:
    if log_file.exists() and _extract_indicator(log_file):
        return 0
    env = os.environ.copy()
    env.update(
        {
            "GM_OPEN_DAILY_SCORE_EXIT": str(int(config.get("open_daily_score_exit", 0))),
            "GM_MAX_DAILY_SELLS": str(int(config.get("max_daily_sells", 1))),
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
        "--score-continue-entry-ratio",
        "999",
        "--backtest-adjust",
        "none",
        "--backtest-initial-cash",
        "600000",
        "--backtest-slippage-ratio",
        "0.0015",
    ]
    _append_arg(command, "--score-exit-entry-ratio", config.get("score_exit_entry_ratio"))
    _append_arg(command, "--min-holding-days-before-score-exit", config.get("min_score_exit"))
    _append_arg(command, "--score-stop-loss-day-drop-ratio", config.get("score_stop_loss_day_drop_ratio"))
    _append_arg(command, "--light-stop-loss-pct", config.get("light_stop_loss_pct"))
    _append_arg(command, "--min-holding-days-before-light-stop", config.get("min_light_stop"))
    _append_arg(command, "--take-profit-pct", config.get("take_profit_pct"))
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as log:
        proc = subprocess.run(command, cwd=str(MAIN), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    return proc.returncode


def main() -> None:
    REPORT_DIR.joinpath("logs").mkdir(parents=True, exist_ok=True)
    missing = [str(path) for path in SIGNALS.values() if not path.exists()]
    if missing:
        raise SystemExit("Missing source signals: " + json.dumps(missing, ensure_ascii=False))
    results = []
    for signal_name, signal_file in SIGNALS.items():
        signal_stats = _signal_stats(signal_file)
        for base_config in BASE_CONFIGS:
            config = dict(base_config)
            run_name = f"{signal_name}_{config['risk_name']}"
            log_file = REPORT_DIR / "logs" / f"{run_name}.log"
            returncode = _run(signal_file, log_file, config)
            indicator = _extract_indicator(log_file) or {}
            result = {
                "name": run_name,
                "signal_name": signal_name,
                **config,
                "default_stop_loss_pct": 0.08,
                "score_continue_entry_ratio": 999,
                "returncode": returncode,
                "signal_file": str(signal_file),
                "log_file": str(log_file),
                **signal_stats,
                "annual": indicator.get("pnl_ratio_annual"),
                "sharpe": indicator.get("sharp_ratio"),
                "max_drawdown": indicator.get("max_drawdown"),
                "win_ratio": indicator.get("win_ratio"),
                **_exposure_stats(log_file),
            }
            results.append(result)
            print(json.dumps(result, ensure_ascii=False, default=str), flush=True)

    fieldnames = sorted({key for row in results for key in row.keys()})
    outputs = {
        "summary.csv": results,
        "summary_by_sharpe.csv": sorted(results, key=lambda row: float(row.get("sharpe") or -999), reverse=True),
        "summary_by_annual.csv": sorted(results, key=lambda row: float(row.get("annual") or -999), reverse=True),
        "summary_avg80_by_sharpe.csv": [
            row
            for row in sorted(results, key=lambda item: float(item.get("sharpe") or -999), reverse=True)
            if float(row.get("avg_invested_pct") or -999) >= 0.80
        ],
        "summary_qualified_target_by_sharpe.csv": [
            row
            for row in sorted(results, key=lambda item: float(item.get("sharpe") or -999), reverse=True)
            if float(row.get("annual") or -999) >= 2.0
            and float(row.get("sharpe") or -999) >= 3.0
            and float(row.get("avg_invested_pct") or -999) >= 0.80
        ],
    }
    for filename, data in outputs.items():
        with (REPORT_DIR / filename).open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(data)


if __name__ == "__main__":
    main()
