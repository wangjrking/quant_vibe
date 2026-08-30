from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import os
import re
import subprocess
from pathlib import Path

from research_list_age_current_best_probe_20260620 import JUEJIN_PYTHON, MAIN, MARKET_DB, PRED_DB, STRATEGY_DIR


ROOT = Path(__file__).resolve().parents[2]
BASE_REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_model_application_20260620"
REPORT_DIR = BASE_REPORT_DIR / "latest_10d_dynamic_low_score_filter_probe_20260620"
TABLE_10D = "stock_predict_data_model_agent_toprank_latestfactor_20260618_executable_10d_open_return_score_20240604_20260618"

BASELINE_SIGNAL = (
    BASE_REPORT_DIR
    / "latest_10d_confirm5d_close_rate_refine_20260620"
    / "signals"
    / "tk5_h5_mv200_5d50_cr0p965_1p085.csv"
)
AMOUNT_10000_SIGNAL = (
    BASE_REPORT_DIR / "latest_10d_liquidity_filter_probe_20260620" / "signals" / "amt10000p0_turnnone_volnone.csv"
)
AMOUNT_20000_SIGNAL = (
    BASE_REPORT_DIR / "latest_10d_liquidity_filter_probe_20260620" / "signals" / "amt20000p0_turnnone_volnone.csv"
)
CR_9675_SIGNAL = BASE_REPORT_DIR / "latest_10d_close_rate_micro_refine_20260620" / "signals" / "cr0p9675_1p085.csv"

SOURCES = {
    "amount10000": AMOUNT_10000_SIGNAL,
    "amount20000": AMOUNT_20000_SIGNAL,
    "cr9675": CR_9675_SIGNAL,
}

CONFIGS = []
for threshold in (0.05, 0.08, 0.10, 0.12, 0.15):
    CONFIGS.append(("mean", threshold, "amount10000"))
for threshold in (0.10, 0.15, 0.20, 0.25, 0.30):
    CONFIGS.append(("top", threshold, "amount10000"))
for threshold in (0.05, 0.08, 0.10, 0.12):
    CONFIGS.append(("mean", threshold, "amount20000"))
for threshold in (0.10, 0.15, 0.20, 0.25):
    CONFIGS.append(("top", threshold, "cr9675"))


def _safe(value) -> str:
    return str(value).replace(".", "p").replace("-", "neg")


def _load_by_day(path: Path) -> tuple[dict[str, list[dict]], list[str]]:
    rows_by_day: dict[str, list[dict]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            rows_by_day.setdefault(str(row["signal_date"]), []).append(row)
    return rows_by_day, fieldnames


def _baseline_stats(rows_by_day: dict[str, list[dict]]) -> dict[str, dict]:
    stats = {}
    for trade_date, rows in rows_by_day.items():
        values = [float(row["pred_prob"]) for row in rows if row.get("pred_prob") not in ("", None)]
        if not values:
            continue
        stats[trade_date] = {
            "mean": sum(values) / len(values),
            "top": max(values),
        }
    return stats


def _write_dynamic_signal(metric: str, threshold: float, alt_name: str) -> tuple[Path, dict]:
    base_rows, base_fields = _load_by_day(BASELINE_SIGNAL)
    alt_rows, alt_fields = _load_by_day(SOURCES[alt_name])
    stats = _baseline_stats(base_rows)
    output = REPORT_DIR / "signals" / f"{metric}_lt_{_safe(threshold)}_{alt_name}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for field in [*base_fields, *alt_fields]:
        if field not in fields:
            fields.append(field)
    rows = []
    use_alt_days = 0
    missing_alt_days = 0
    for trade_date in sorted(base_rows):
        use_alt = stats.get(trade_date, {}).get(metric, 999.0) < threshold
        source_rows = alt_rows.get(trade_date) if use_alt else base_rows.get(trade_date)
        if use_alt:
            use_alt_days += 1
            if not source_rows:
                missing_alt_days += 1
                source_rows = base_rows.get(trade_date)
        rows.extend(source_rows or [])
    with output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return output, {
        "metric": metric,
        "threshold": threshold,
        "alt_name": alt_name,
        "use_alt_days": use_alt_days,
        "missing_alt_days": missing_alt_days,
        "signal_count": len(rows),
    }


def _extract_indicator(log_file: Path) -> dict | None:
    marker = "GM_BACKTEST_INDICATOR:"
    if not log_file.exists():
        return None
    for line in reversed(log_file.read_text(encoding="utf-8", errors="ignore").splitlines()):
        if marker not in line:
            continue
        payload = line.split(marker, 1)[1].strip()
        try:
            return ast.literal_eval(payload)
        except Exception:
            return eval(payload, {"__builtins__": {}}, {"datetime": datetime_module})
    return None


def _exposure_stats(log_file: Path) -> dict:
    values = []
    active_positions = []
    pattern = re.compile(r"invested_pct=([0-9.]+).*active_positions=(\d+)")
    if not log_file.exists():
        return {"avg_invested_pct": None, "ge80_ratio": None, "max_active_positions": None, "exposure_points": 0}
    for line in log_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if not match:
            continue
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


def _run(signal_file: Path, log_file: Path) -> int:
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
        "0.98",
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
    proc = subprocess.run(command, cwd=str(MAIN), env=env, capture_output=True, text=True)
    if proc.stdout.strip():
        try:
            payload = json.loads(proc.stdout.strip().splitlines()[-1])
            return int(payload.get("returncode", proc.returncode))
        except Exception:
            pass
    return proc.returncode


def main() -> None:
    REPORT_DIR.joinpath("logs").mkdir(parents=True, exist_ok=True)
    rows = []
    for metric, threshold, alt_name in CONFIGS:
        signal_file, config_stats = _write_dynamic_signal(metric, threshold, alt_name)
        name = f"{metric}_lt_{_safe(threshold)}_{alt_name}"
        log_file = REPORT_DIR / "logs" / f"{name}.log"
        returncode = 0 if log_file.exists() and log_file.stat().st_size > 0 else _run(signal_file, log_file)
        indicator = _extract_indicator(log_file)
        row = {
            "name": name,
            "returncode": returncode,
            "signal_file": str(signal_file),
            "log_file": str(log_file),
            **config_stats,
            **_exposure_stats(log_file),
        }
        if indicator:
            row.update(
                {
                    "annual": indicator.get("pnl_ratio_annual"),
                    "sharpe": indicator.get("sharp_ratio"),
                    "max_drawdown": indicator.get("max_drawdown"),
                    "open_count": indicator.get("open_count"),
                    "close_count": indicator.get("close_count"),
                    "win_ratio": indicator.get("win_ratio"),
                    "calmar_ratio": indicator.get("calmar_ratio"),
                }
            )
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False, default=str))
    fieldnames = list(rows[0].keys())
    with (REPORT_DIR / "summary.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    qualified = [
        row
        for row in rows
        if row.get("annual") is not None
        and float(row["annual"]) > 2.0
        and row.get("avg_invested_pct") is not None
        and float(row["avg_invested_pct"]) >= 0.80
    ]
    with (REPORT_DIR / "summary_qualified_annual2_avg80_by_sharpe.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(qualified, key=lambda row: float(row["sharpe"]), reverse=True))


if __name__ == "__main__":
    main()
