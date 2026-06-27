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
REPORT_DIR = REPORT_ROOT / "new5d_signal_filter_continue_grid_20260621"
DIAG_FILE = REPORT_ROOT / "new5d_signal_return_diagnostics_20260621" / "signal_forward_returns.csv"


SOURCES = {
    "h7_ge0p45": REPORT_ROOT / "new5d_dynamic_holding_fine_20260621" / "signals" / "h7_top_ge0p45_targetbase.csv",
    "h7_ge0p4": REPORT_ROOT / "new5d_dynamic_holding_fine_20260621" / "signals" / "h7_top_ge0p4_targetbase.csv",
}


FILTERS = {
    "mv120_200": {"min_total_mv": 120000.0, "max_total_mv": 200000.0},
    "mv120_200_age2500": {"min_total_mv": 120000.0, "max_total_mv": 200000.0, "min_list_age_days": 2500.0},
    "mv120_200_r3ge0p3": {"min_total_mv": 120000.0, "max_total_mv": 200000.0, "min_rank3d": 0.30},
    "mv120_200_turn0_4": {"min_total_mv": 120000.0, "max_total_mv": 200000.0, "min_turnover": 0.0, "max_turnover": 4.0},
    "mv120_200_cr955_103": {"min_total_mv": 120000.0, "max_total_mv": 200000.0, "min_close_rate": 0.955, "max_close_rate": 1.03},
    "mv120_200_r5lte0p99": {"min_total_mv": 120000.0, "max_total_mv": 200000.0, "max_rank5d": 0.99},
    "age2500_cr955_106": {"min_list_age_days": 2500.0, "min_close_rate": 0.955, "max_close_rate": 1.06},
}


GRID: list[dict] = []
for source_name, source_signal in SOURCES.items():
    for filter_name, filter_config in FILTERS.items():
        for max_positions in (7, 8):
            for continue_ratio, max_holding_days in ((None, 8), (1.0, 10), (0.75, 10)):
                GRID.append(
                    {
                        "name": (
                            f"{source_name}_{filter_name}_mp{max_positions}_"
                            f"cont{str(continue_ratio).replace('.', 'p') if continue_ratio is not None else 'off'}"
                            f"_maxh{max_holding_days}"
                        ),
                        "source_name": source_name,
                        "source_signal": source_signal,
                        "filter_name": filter_name,
                        **filter_config,
                        "max_positions": max_positions,
                        "target_pct": 0.98 / max_positions,
                        "continue_ratio": continue_ratio,
                        "max_holding_days": max_holding_days,
                    }
                )


def _to_float(value):
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_diag() -> dict[tuple[str, str], dict]:
    if not DIAG_FILE.exists():
        return {}
    with DIAG_FILE.open("r", encoding="utf-8-sig", newline="") as file:
        return {
            (str(row.get("signal_date")), str(row.get("stock_code"))): row
            for row in csv.DictReader(file)
        }


def _passes(row: dict, config: dict) -> bool:
    total_mv = _to_float(row.get("total_mv"))
    if config.get("min_total_mv") is not None and (total_mv is None or total_mv < float(config["min_total_mv"])):
        return False
    if config.get("max_total_mv") is not None and (total_mv is None or total_mv > float(config["max_total_mv"])):
        return False
    age = _to_float(row.get("list_age_days"))
    if config.get("min_list_age_days") is not None and (age is None or age < float(config["min_list_age_days"])):
        return False
    close_rate = _to_float(row.get("close_rate"))
    if config.get("min_close_rate") is not None and (close_rate is None or close_rate < float(config["min_close_rate"])):
        return False
    if config.get("max_close_rate") is not None and (close_rate is None or close_rate > float(config["max_close_rate"])):
        return False
    rank5d = _to_float(row.get("rank5d_new"))
    if config.get("max_rank5d") is not None and (rank5d is None or rank5d > float(config["max_rank5d"])):
        return False
    rank3d = _to_float(row.get("rank3d"))
    if config.get("min_rank3d") is not None and (rank3d is None or rank3d < float(config["min_rank3d"])):
        return False
    turnover = _to_float(row.get("turnover_rate"))
    if config.get("min_turnover") is not None and (turnover is None or turnover < float(config["min_turnover"])):
        return False
    if config.get("max_turnover") is not None and (turnover is None or turnover > float(config["max_turnover"])):
        return False
    return True


def _write_filtered_signal(config: dict, output: Path) -> dict:
    if output.exists() and output.stat().st_size > 0:
        return _signal_stats(output)
    diag = _load_diag()
    source = Path(config["source_signal"])
    with source.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = []
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            extra = diag.get((str(row.get("signal_date")), str(row.get("stock_code"))), {})
            for key in ("amount", "turnover_rate", "rank3d"):
                if key not in row or row.get(key) in (None, "", "None"):
                    row[key] = extra.get(key)
            rows.append(row)
    for key in ("amount", "turnover_rate", "rank3d"):
        if key not in fieldnames:
            fieldnames.append(key)
    filtered = []
    for row in rows:
        if _passes(row, config):
            row = dict(row)
            row["target_pct"] = str(config["target_pct"])
            filtered.append(row)
    if not filtered:
        raise ValueError(f"No signals after filter: {config['name']}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(filtered)
    return _signal_stats(output)


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


def _run(config: dict, signal_file: Path, log_file: Path) -> int:
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
        str(signal_file),
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
        "--backtest-adjust",
        "none",
        "--backtest-initial-cash",
        "600000",
        "--backtest-slippage-ratio",
        "0.0015",
    ]
    if config.get("continue_ratio") is not None:
        command.extend(["--score-continue-entry-ratio", str(config["continue_ratio"])])
    else:
        command.extend(["--score-continue-entry-ratio", "999"])
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
    REPORT_DIR.joinpath("signals").mkdir(parents=True, exist_ok=True)
    REPORT_DIR.joinpath("logs").mkdir(parents=True, exist_ok=True)
    results = []
    for config in GRID:
        source = Path(config["source_signal"])
        if not source.exists():
            raise SystemExit(f"Missing source signal: {source}")
        signal_file = REPORT_DIR / "signals" / f"{config['name']}.csv"
        log_file = REPORT_DIR / "logs" / f"{config['name']}.log"
        signal_stats = _write_filtered_signal(config, signal_file)
        returncode = _run(config, signal_file, log_file)
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
            **signal_stats,
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
