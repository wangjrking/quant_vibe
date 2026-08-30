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
FINE_SIGNAL_DIR = REPORT_ROOT / "new5d_dynamic_holding_fine_20260621" / "signals"
MAXPOS_SIGNAL_DIR = REPORT_ROOT / "new5d_maxpos_structure_grid_20260621" / "signals"
REPORT_DIR = REPORT_ROOT / "new5d_max_daily_sells_grid_20260621"


SIGNALS = {
    "h7_top_ge0p45_mp6": {
        "path": FINE_SIGNAL_DIR / "h7_top_ge0p45_targetbase.csv",
        "max_positions": 6,
        "target_pct": 0.98 / 6.0,
    },
    "h7_top_ge0p4_mp6": {
        "path": FINE_SIGNAL_DIR / "h7_top_ge0p4_targetbase.csv",
        "max_positions": 6,
        "target_pct": 0.98 / 6.0,
    },
    "h7_top_ge0p45_mp7": {
        "path": MAXPOS_SIGNAL_DIR / "h7_top_ge0p45_mp7_target098div.csv",
        "max_positions": 7,
        "target_pct": 0.98 / 7.0,
    },
    "h7_top_ge0p4_mp7": {
        "path": MAXPOS_SIGNAL_DIR / "h7_top_ge0p4_mp7_target098div.csv",
        "max_positions": 7,
        "target_pct": 0.98 / 7.0,
    },
}


GRID: list[dict] = []
for signal_name, signal_config in SIGNALS.items():
    for max_daily_sells in (0, 2, 3):
        GRID.append(
            {
                "name": f"{signal_name}_base_sells{max_daily_sells}",
                "signal_name": signal_name,
                "signal_file": signal_config["path"],
                "max_positions": signal_config["max_positions"],
                "target_pct": signal_config["target_pct"],
                "max_daily_sells": max_daily_sells,
                "open_daily_score_exit": 0,
                "score_exit_rank": None,
                "min_score_exit": None,
            }
        )
        for min_hold in (1, 2):
            GRID.append(
                {
                    "name": f"{signal_name}_rankexit0p7_min{min_hold}_sells{max_daily_sells}",
                    "signal_name": signal_name,
                    "signal_file": signal_config["path"],
                    "max_positions": signal_config["max_positions"],
                    "target_pct": signal_config["target_pct"],
                    "max_daily_sells": max_daily_sells,
                    "open_daily_score_exit": 1,
                    "score_exit_rank": 0.70,
                    "min_score_exit": min_hold,
                }
            )


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


def _run(config: dict, log_file: Path) -> int:
    if log_file.exists() and _extract_indicator(log_file):
        return 0
    env = os.environ.copy()
    env.update(
        {
            "GM_OPEN_DAILY_SCORE_EXIT": str(config["open_daily_score_exit"]),
            "GM_MAX_DAILY_SELLS": str(config["max_daily_sells"]),
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
        str(config["signal_file"]),
        "--log-file",
        str(log_file),
        "--max-positions",
        str(config["max_positions"]),
        "--holding-days",
        "5",
        "--max-holding-days",
        "8",
        "--target-position-pct",
        str(config["target_pct"]),
        "--score-db",
        str(PRED_DB),
        "--score-table",
        str(TABLE_10D),
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
    if config["score_exit_rank"] is not None:
        command.extend(
            [
                "--score-exit-entry-ratio",
                "0",
                "--score-exit-rank",
                str(config["score_exit_rank"]),
                "--min-holding-days-before-score-exit",
                str(config["min_score_exit"]),
            ]
        )
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
    results = []
    for config in GRID:
        signal_file = Path(config["signal_file"])
        if not signal_file.exists():
            raise SystemExit(f"Missing signal file: {signal_file}")
        log_file = REPORT_DIR / "logs" / f"{config['name']}.log"
        returncode = _run(config, log_file)
        indicator = _extract_indicator(log_file) or {}
        result = {
            **config,
            "signal_file": str(signal_file),
            "returncode": returncode,
            "log_file": str(log_file),
            "annual": indicator.get("pnl_ratio_annual"),
            "sharpe": indicator.get("sharp_ratio"),
            "max_drawdown": indicator.get("max_drawdown"),
            "win_ratio": indicator.get("win_ratio"),
            **_signal_stats(signal_file),
            **_exposure_stats(log_file),
        }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False, default=str), flush=True)

    sorted_by_sharpe = sorted(results, key=lambda row: float(row.get("sharpe") or -999), reverse=True)
    sorted_by_annual = sorted(results, key=lambda row: float(row.get("annual") or -999), reverse=True)
    _write_csv(REPORT_DIR / "summary.csv", results)
    _write_csv(REPORT_DIR / "summary_by_sharpe.csv", sorted_by_sharpe)
    _write_csv(REPORT_DIR / "summary_by_annual.csv", sorted_by_annual)
    _write_csv(
        REPORT_DIR / "summary_avg80_by_sharpe.csv",
        [row for row in sorted_by_sharpe if float(row.get("avg_invested_pct") or -999) >= 0.80],
    )
    _write_csv(
        REPORT_DIR / "summary_qualified_target_by_sharpe.csv",
        [
            row
            for row in sorted_by_sharpe
            if float(row.get("annual") or -999) >= 2.0
            and float(row.get("sharpe") or -999) >= 3.0
            and float(row.get("avg_invested_pct") or -999) >= 0.80
        ],
    )


if __name__ == "__main__":
    main()
