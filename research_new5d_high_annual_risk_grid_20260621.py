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
SOURCE_DIR = REPORT_ROOT / "new5d_confirm_trade_param_grid_20260621" / "signals"
REPORT_DIR = REPORT_ROOT / "new5d_high_annual_risk_grid_20260621"


SIGNALS = {
    "c0p5_tk3_h7": {
        "path": SOURCE_DIR / "c0p5_cr0p955_1p085_tk3_h7_mp6_capnone.csv",
        "holding_days": 7,
        "max_positions": 6,
        "target_pct": 0.98 / 6.0,
    },
    "c0p45_tk3_h7": {
        "path": SOURCE_DIR / "c0p45_cr0p955_1p085_tk3_h7_mp6_capnone.csv",
        "holding_days": 7,
        "max_positions": 6,
        "target_pct": 0.98 / 6.0,
    },
}


RISK_CONFIGS = [
    {"risk_name": "base", "open_daily_score_exit": 0},
    {"risk_name": "stop0p05", "open_daily_score_exit": 0, "stop_loss_pct": 0.05},
    {"risk_name": "stop0p06", "open_daily_score_exit": 0, "stop_loss_pct": 0.06},
    {"risk_name": "stop0p10", "open_daily_score_exit": 0, "stop_loss_pct": 0.10},
    {"risk_name": "light0p03_min1", "open_daily_score_exit": 0, "light_stop_loss_pct": 0.03, "min_light_stop": 1},
    {"risk_name": "light0p05_min1", "open_daily_score_exit": 0, "light_stop_loss_pct": 0.05, "min_light_stop": 1},
    {"risk_name": "light0p05_min2", "open_daily_score_exit": 0, "light_stop_loss_pct": 0.05, "min_light_stop": 2},
    {"risk_name": "take0p12", "open_daily_score_exit": 0, "take_profit_pct": 0.12},
    {"risk_name": "take0p18", "open_daily_score_exit": 0, "take_profit_pct": 0.18},
    {"risk_name": "take0p25", "open_daily_score_exit": 0, "take_profit_pct": 0.25},
    {"risk_name": "rankexit0p5_min1", "open_daily_score_exit": 1, "score_exit_rank": 0.50, "min_score_exit": 1},
    {"risk_name": "rankexit0p5_min2", "open_daily_score_exit": 1, "score_exit_rank": 0.50, "min_score_exit": 2},
    {"risk_name": "rankexit0p7_min1", "open_daily_score_exit": 1, "score_exit_rank": 0.70, "min_score_exit": 1},
    {"risk_name": "rankexit0p7_min2", "open_daily_score_exit": 1, "score_exit_rank": 0.70, "min_score_exit": 2},
    {
        "risk_name": "rankexit0p5_min2_light0p05",
        "open_daily_score_exit": 1,
        "score_exit_rank": 0.50,
        "min_score_exit": 2,
        "light_stop_loss_pct": 0.05,
        "min_light_stop": 2,
    },
    {
        "risk_name": "rankexit0p7_min2_take0p18",
        "open_daily_score_exit": 1,
        "score_exit_rank": 0.70,
        "min_score_exit": 2,
        "take_profit_pct": 0.18,
    },
    {
        "risk_name": "light0p05_take0p18",
        "open_daily_score_exit": 0,
        "light_stop_loss_pct": 0.05,
        "min_light_stop": 1,
        "take_profit_pct": 0.18,
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
    if value is not None:
        command.extend([option, str(value)])


def _run(signal: dict, config: dict, log_file: Path) -> int:
    if log_file.exists() and _extract_indicator(log_file):
        return 0
    env = os.environ.copy()
    env.update(
        {
            "GM_OPEN_DAILY_SCORE_EXIT": str(int(config.get("open_daily_score_exit", 0))),
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
        str(signal["path"]),
        "--log-file",
        str(log_file),
        "--max-positions",
        str(signal["max_positions"]),
        "--holding-days",
        str(signal["holding_days"]),
        "--max-holding-days",
        str(signal["holding_days"]),
        "--target-position-pct",
        str(signal["target_pct"]),
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
    _append_arg(command, "--stop-loss-pct", config.get("stop_loss_pct"))
    _append_arg(command, "--take-profit-pct", config.get("take_profit_pct"))
    _append_arg(command, "--light-stop-loss-pct", config.get("light_stop_loss_pct"))
    _append_arg(command, "--min-holding-days-before-light-stop", config.get("min_light_stop"))
    if config.get("score_exit_rank") is not None:
        _append_arg(command, "--score-exit-entry-ratio", 0)
        _append_arg(command, "--score-exit-rank", config.get("score_exit_rank"))
        _append_arg(command, "--min-holding-days-before-score-exit", config.get("min_score_exit"))
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as log:
        proc = subprocess.run(command, cwd=str(MAIN), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    return proc.returncode


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    REPORT_DIR.joinpath("logs").mkdir(parents=True, exist_ok=True)
    missing = [str(signal["path"]) for signal in SIGNALS.values() if not signal["path"].exists()]
    if missing:
        raise SystemExit("Missing source signals: " + json.dumps(missing, ensure_ascii=False))
    results = []
    for signal_name, signal in SIGNALS.items():
        signal_stats = _signal_stats(signal["path"])
        for config in RISK_CONFIGS:
            run_name = f"{signal_name}_{config['risk_name']}"
            log_file = REPORT_DIR / "logs" / f"{run_name}.log"
            returncode = _run(signal, config, log_file)
            indicator = _extract_indicator(log_file) or {}
            result = {
                "name": run_name,
                "signal_name": signal_name,
                **signal,
                **config,
                "returncode": returncode,
                "signal_file": str(signal["path"]),
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
    by_sharpe = sorted(results, key=lambda row: float(row.get("sharpe") or -999), reverse=True)
    by_annual = sorted(results, key=lambda row: float(row.get("annual") or -999), reverse=True)
    _write_csv(REPORT_DIR / "summary.csv", results)
    _write_csv(REPORT_DIR / "summary_by_sharpe.csv", by_sharpe)
    _write_csv(REPORT_DIR / "summary_by_annual.csv", by_annual)
    _write_csv(REPORT_DIR / "summary_avg80_by_sharpe.csv", [row for row in by_sharpe if float(row.get("avg_invested_pct") or -999) >= 0.80])
    _write_csv(
        REPORT_DIR / "summary_qualified_target_by_sharpe.csv",
        [
            row
            for row in by_sharpe
            if float(row.get("annual") or -999) >= 2.0
            and float(row.get("sharpe") or -999) >= 3.0
            and float(row.get("avg_invested_pct") or -999) >= 0.80
        ],
    )


if __name__ == "__main__":
    main()
