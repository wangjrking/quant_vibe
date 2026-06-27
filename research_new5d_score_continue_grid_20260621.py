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
REPORT_DIR = REPORT_ROOT / "new5d_score_continue_grid_20260621"


SOURCES = {
    "mp6_h7_ge0p45": {
        "path": REPORT_ROOT / "new5d_dynamic_holding_fine_20260621" / "signals" / "h7_top_ge0p45_targetbase.csv",
        "max_positions": 6,
        "target_pct": 0.98 / 6.0,
    },
    "mp7_h7_ge0p45": {
        "path": REPORT_ROOT / "new5d_maxpos_structure_grid_20260621" / "signals" / "h7_top_ge0p45_mp7_target098div.csv",
        "max_positions": 7,
        "target_pct": 0.98 / 7.0,
    },
    "mp8_h7_ge0p45": {
        "path": REPORT_ROOT / "new5d_maxpos_structure_grid_20260621" / "signals" / "h7_top_ge0p45_mp8_target098div.csv",
        "max_positions": 8,
        "target_pct": 0.98 / 8.0,
    },
}


GRID: list[dict] = []
for source_name, source in SOURCES.items():
    for max_holding_days in (9, 10, 12):
        for continue_ratio in (0.75, 0.85, 0.95, 1.00):
            GRID.append(
                {
                    "name": (
                        f"{source_name}_cont{str(continue_ratio).replace('.', 'p')}"
                        f"_maxh{max_holding_days}"
                    ),
                    "source_name": source_name,
                    "signal_file": source["path"],
                    "max_positions": source["max_positions"],
                    "target_pct": source["target_pct"],
                    "continue_ratio": continue_ratio,
                    "max_holding_days": max_holding_days,
                    "open_daily_score_exit": 0,
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
        str(config["signal_file"]),
        "--log-file",
        str(log_file),
        "--max-positions",
        str(config["max_positions"]),
        "--holding-days",
        "7",
        "--max-holding-days",
        str(config["max_holding_days"]),
        "--target-position-pct",
        str(config["target_pct"]),
        "--score-db",
        str(PRED_DB),
        "--score-table",
        TABLE_10D,
        "--market-db",
        str(MARKET_DB),
        "--score-continue-entry-ratio",
        str(config["continue_ratio"]),
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
            "returncode": returncode,
            "signal_file": str(signal_file),
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
