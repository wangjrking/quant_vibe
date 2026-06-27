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
    / "new5d_position_scale_grid_20260621"
)


SIGNALS = {
    "h7_top_ge0p4": SOURCE_DIR / "h7_top_ge0p4_targetbase.csv",
    "h7_top_ge0p45": SOURCE_DIR / "h7_top_ge0p45_targetbase.csv",
}

RISK_CONFIGS = [
    {"risk_name": "base"},
    {
        "risk_name": "index_cc3_intraday3_scale0p5",
        "env": {
            "GM_INDEX_RISK_EXIT_MODE": "1",
            "GM_INDEX_RISK_CC_THRESHOLD": "-0.03",
            "GM_INDEX_RISK_INTRADAY_THRESHOLD": "-0.03",
            "GM_INDEX_RISK_BUY_SCALE": "0.5",
        },
    },
    {
        "risk_name": "equitydd_soft8_hard14",
        "env": {
            "GM_EQUITY_DD_RISK_MODE": "1",
            "GM_EQUITY_DD_SOFT_TRIGGER": "0.08",
            "GM_EQUITY_DD_HARD_TRIGGER": "0.14",
            "GM_EQUITY_DD_RECOVER_TRIGGER": "0.04",
            "GM_EQUITY_DD_SOFT_SCALE": "0.85",
            "GM_EQUITY_DD_HARD_SCALE": "0.65",
            "GM_EQUITY_DD_RESIZE_EXISTING": "0",
        },
    },
]

GRID: list[dict] = []
for signal_name, source in SIGNALS.items():
    for target_pct in (0.168, 0.173, 0.178, 0.183):
        GRID.append(
            {
                "name": f"{signal_name}_target{str(target_pct).replace('.', 'p')}_base",
                "signal_name": signal_name,
                "source_signal": source,
                "target_pct": target_pct,
                "max_positions": 6,
                "risk_name": "base",
                "extra_env": {},
            }
        )
    for target_pct in (0.178, 0.183):
        for risk in RISK_CONFIGS[1:]:
            GRID.append(
                {
                    "name": f"{signal_name}_target{str(target_pct).replace('.', 'p')}_{risk['risk_name']}",
                    "signal_name": signal_name,
                    "source_signal": source,
                    "target_pct": target_pct,
                    "max_positions": 6,
                    "risk_name": risk["risk_name"],
                    "extra_env": dict(risk.get("env", {})),
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


def _rewrite_signal(source: Path, output: Path, target_pct: float) -> None:
    if output.exists() and output.stat().st_size > 0:
        return
    with source.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if "target_pct" not in fieldnames:
        fieldnames.append("target_pct")
    for row in rows:
        row["target_pct"] = str(target_pct)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _signal_stats(signal_file: Path) -> dict:
    with signal_file.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    return {
        "signal_count": len(rows),
        "buy_days": len({row.get("buy_date") for row in rows if row.get("buy_date")}),
    }


def _run(signal_file: Path, log_file: Path, config: dict) -> int:
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
    env.update(config.get("extra_env") or {})
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
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as log:
        proc = subprocess.run(command, cwd=str(MAIN), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    return proc.returncode


def main() -> None:
    REPORT_DIR.joinpath("signals").mkdir(parents=True, exist_ok=True)
    REPORT_DIR.joinpath("logs").mkdir(parents=True, exist_ok=True)
    results = []
    for config in GRID:
        source = Path(config["source_signal"])
        if not source.exists():
            raise SystemExit(f"Missing source signal: {source}")
        signal_file = REPORT_DIR / "signals" / f"{config['name']}.csv"
        log_file = REPORT_DIR / "logs" / f"{config['name']}.log"
        _rewrite_signal(source, signal_file, float(config["target_pct"]))
        returncode = _run(signal_file, log_file, config)
        indicator = _extract_indicator(log_file) or {}
        result = {
            **{key: value for key, value in config.items() if key != "extra_env"},
            "extra_env": json.dumps(config.get("extra_env") or {}, ensure_ascii=False, sort_keys=True),
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
