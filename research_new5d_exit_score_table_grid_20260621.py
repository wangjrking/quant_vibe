from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import os
import re
import subprocess
from pathlib import Path

import research_10d_with_new5d_confirm_probe_20260621 as base
from research_list_age_current_best_probe_20260620 import JUEJIN_PYTHON, MAIN, MARKET_DB, PRED_DB, STRATEGY_DIR, TABLE_10D


ROOT = Path(__file__).resolve().parents[2]
FINE_SIGNAL_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "new5d_dynamic_holding_fine_20260621"
    / "signals"
)
MAXPOS_SIGNAL_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "new5d_maxpos_structure_grid_20260621"
    / "signals"
)
REPORT_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "new5d_exit_score_table_grid_20260621"
)
MANIFEST_3D = ROOT / "quant" / "main" / "config" / "prediction_manifests" / "executable_3d_open_return_l4_formal_20260617.json"


def _load_formal_table(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("approval_status") != "approved_for_l5":
        raise SystemExit(f"manifest is not approved_for_l5: {path}")
    return str(data["table"])


TABLE_5D = base.TABLE_5D_NEW
TABLE_3D = _load_formal_table(MANIFEST_3D)


SIGNALS = {
    "h7_top_ge0p45_mp6": {
        "path": FINE_SIGNAL_DIR / "h7_top_ge0p45_targetbase.csv",
        "max_positions": 6,
        "target_pct": 0.98 / 6.0,
    },
    "h7_top_ge0p45_mp7": {
        "path": MAXPOS_SIGNAL_DIR / "h7_top_ge0p45_mp7_target098div.csv",
        "max_positions": 7,
        "target_pct": 0.98 / 7.0,
    },
}
SCORE_TABLES = {
    "exit10d": TABLE_10D,
    "exit5d": TABLE_5D,
    "exit3d": TABLE_3D,
}


GRID: list[dict] = []
for signal_name, signal_config in SIGNALS.items():
    for score_table_name, score_table in SCORE_TABLES.items():
        for score_exit_rank in (0.30, 0.50, 0.70):
            for min_hold in (1, 2):
                GRID.append(
                    {
                        "name": (
                            f"{signal_name}_{score_table_name}"
                            f"_rank{str(score_exit_rank).replace('.', 'p')}_min{min_hold}"
                        ),
                        "signal_name": signal_name,
                        "signal_file": signal_config["path"],
                        "max_positions": signal_config["max_positions"],
                        "target_pct": signal_config["target_pct"],
                        "score_table_name": score_table_name,
                        "score_table": score_table,
                        "score_exit_rank": score_exit_rank,
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
            "GM_OPEN_DAILY_SCORE_EXIT": "1",
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
        str(config["score_table"]),
        "--market-db",
        str(MARKET_DB),
        "--score-exit-entry-ratio",
        "0",
        "--score-exit-rank",
        str(config["score_exit_rank"]),
        "--min-holding-days-before-score-exit",
        str(config["min_score_exit"]),
        "--score-continue-entry-ratio",
        "999",
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
    fieldnames = list(results[0].keys())
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
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)


if __name__ == "__main__":
    main()
