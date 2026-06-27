from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import math
import os
import re
import subprocess
from pathlib import Path


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = DATA_DIR / "reports" / "strategy_agent_model_application_20260621" / "formal_10d_gap02_target_shape_20260622"
SOURCE_SIGNAL = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_10d_gap_strong_weak_fill_20260622"
    / "signals"
    / "gap02_s0p32_w0p18_h6_mh6.csv"
)
SCORE_DB = DATA_DIR / "reports" / "strategy_agent_model_application_20260621" / "formal_5d10d_weak_day_pool_replace_20260621" / "weak_day_pool_scores.db"
SCORE_TABLE = "weak_f60_liq_primary_count_le1_prank_10d90_5d10_frank_10d60_5d40_20000p0_0p5"
MARKET_DB = DATA_DIR / "STOCK_DAILY_DATA.db"
STRATEGY_DIR = Path(r"D:\work\dfcf\juejin\strategy\ce327750-634e-11f1-b8d7-10ffe0295517")
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")
BACKTEST_START = "2024-06-05 09:00:00"
BACKTEST_END = "2026-06-18 15:30:00"


BASE_ENV = {
    "GM_OPEN_DAILY_SCORE_EXIT": "1",
    "GM_SCORE_EXIT_ENTRY_RATIO": "1.0",
    "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "4",
    "GM_MAX_DAILY_SELLS": "1",
    "GM_STOP_LOSS_PCT": "0.08",
    "GM_TAKE_PROFIT_PCT": "none",
    "GM_LIGHT_STOP_LOSS_PCT": "none",
    "GM_SCORE_EXIT_RANK": "none",
    "GM_LOG_EXPOSURE": "1",
    "GM_SKIP_OPEN_LIMIT_UP_BUY": "1",
    "GM_FORCE_MARKET_ORDER": "0",
    "GM_FORCE_BUY_MARKET_ORDER": "0",
    "GM_SYNC_POSITIONS": "1",
    "GM_CASH_BUFFER": "0.99",
    "GM_VERBOSE_TRADES": "0",
    "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.02",
    "GM_EQUITY_DD_RISK_MODE": "1",
    "GM_EQUITY_DD_SOFT_TRIGGER": "0.10",
    "GM_EQUITY_DD_HARD_TRIGGER": "0.18",
    "GM_EQUITY_DD_RECOVER_TRIGGER": "0.05",
    "GM_EQUITY_DD_SOFT_SCALE": "0.85",
    "GM_EQUITY_DD_HARD_SCALE": "0.60",
}


def _variant(name: str, strong: float, weak: float, *, weak_low_liq: float | None = None, strong_boost: float | None = None) -> dict:
    return {
        "name": name,
        "strong_target": strong,
        "weak_target": weak,
        "weak_low_liq_target": weak_low_liq,
        "strong_boost_target": strong_boost,
        "holding_days": 6,
        "max_holding_days": 6,
        "max_positions": 6,
        "default_target": strong,
        "env": dict(BASE_ENV),
    }


VARIANTS = [
    _variant("s030_w018", 0.30, 0.18),
    _variant("s032_w016", 0.32, 0.16),
    _variant("s034_w014", 0.34, 0.14),
    _variant("s036_w012", 0.36, 0.12),
    _variant("s030_w016_lowliq010", 0.30, 0.16, weak_low_liq=0.10),
    _variant("s032_w014_lowliq010", 0.32, 0.14, weak_low_liq=0.10),
    _variant("s034_w012_lowliq008", 0.34, 0.12, weak_low_liq=0.08),
    _variant("s030_w018_boost034", 0.30, 0.18, strong_boost=0.34),
    _variant("s032_w016_boost036", 0.32, 0.16, strong_boost=0.36),
    _variant("s030_w014_boost036_lowliq008", 0.30, 0.14, weak_low_liq=0.08, strong_boost=0.36),
]


def _to_float(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return value


def _load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _is_low_liq_weak(row: dict) -> bool:
    amount = _to_float(row.get("amount"), 0.0)
    turnover = _to_float(row.get("turnover_rate"), 0.0)
    total_mv = _to_float(row.get("total_mv"), 999999999.0)
    pred_gap = _to_float(row.get("pred_gap"), 1.0)
    return amount < 20000.0 or turnover < 0.5 or total_mv > 200000.0 or pred_gap >= 0.08


def _is_boost_strong(row: dict) -> bool:
    amount = _to_float(row.get("amount"), 0.0)
    turnover = _to_float(row.get("turnover_rate"), 0.0)
    total_mv = _to_float(row.get("total_mv"), 999999999.0)
    pred_gap = _to_float(row.get("pred_gap"), 1.0)
    return pred_gap < 0.015 and amount >= 20000.0 and turnover >= 0.5 and total_mv <= 200000.0


def _write_signal(variant: dict, signal_file: Path) -> None:
    rows = _load_rows(SOURCE_SIGNAL)
    fieldnames = list(rows[0].keys())
    for row in rows:
        role = str(row.get("pool_role") or "")
        if role == "strong":
            target = float(variant["strong_target"])
            if variant.get("strong_boost_target") is not None and _is_boost_strong(row):
                target = float(variant["strong_boost_target"])
        else:
            target = float(variant["weak_target"])
            if variant.get("weak_low_liq_target") is not None and _is_low_liq_weak(row):
                target = float(variant["weak_low_liq_target"])
        row["target_pct"] = f"{target:.5f}"
        row["holding_days"] = str(int(variant["holding_days"]))
    signal_file.parent.mkdir(parents=True, exist_ok=True)
    with signal_file.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in fieldnames} for row in rows])


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


def _signal_stats(path: Path) -> dict:
    rows = _load_rows(path)
    targets = [_to_float(row.get("target_pct")) for row in rows]
    targets = [value for value in targets if value is not None]
    return {
        "signal_count": len(rows),
        "buy_days": len({row.get("buy_date") for row in rows if row.get("buy_date")}),
        "avg_signal_target_pct": sum(targets) / len(targets) if targets else None,
        "max_signal_target_pct": max(targets) if targets else None,
    }


def _run_backtest(variant: dict, signal_file: Path, log_file: Path) -> int:
    if log_file.exists() and _extract_indicator(log_file):
        return 0
    env = os.environ.copy()
    env.update({str(key): str(value) for key, value in variant["env"].items()})
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
        str(int(variant["max_positions"])),
        "--holding-days",
        str(int(variant["holding_days"])),
        "--max-holding-days",
        str(int(variant["max_holding_days"])),
        "--target-position-pct",
        str(float(variant["default_target"])),
        "--score-db",
        str(SCORE_DB),
        "--score-table",
        SCORE_TABLE,
        "--market-db",
        str(MARKET_DB),
        "--backtest-start",
        BACKTEST_START,
        "--backtest-end",
        BACKTEST_END,
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


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _metric(row: dict, key: str) -> float:
    return _to_float(row.get(key), float("-inf"))


def main() -> int:
    (REPORT_DIR / "signals").mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for index, variant in enumerate(VARIANTS, start=1):
        signal_file = REPORT_DIR / "signals" / f"{variant['name']}.csv"
        log_file = REPORT_DIR / "logs" / f"{variant['name']}.log"
        _write_signal(variant, signal_file)
        returncode = _run_backtest(variant, signal_file, log_file)
        indicator = _extract_indicator(log_file) or {}
        row = {
            "name": variant["name"],
            "strong_target": variant["strong_target"],
            "weak_target": variant["weak_target"],
            "weak_low_liq_target": variant.get("weak_low_liq_target"),
            "strong_boost_target": variant.get("strong_boost_target"),
            "returncode": returncode,
            "signal_file": str(signal_file),
            "log_file": str(log_file),
            "annual": indicator.get("pnl_ratio_annual", indicator.get("annual_return")),
            "sharpe": indicator.get("sharp_ratio", indicator.get("sharpe_ratio")),
            "max_drawdown": indicator.get("max_drawdown"),
            "win_ratio": indicator.get("win_ratio"),
            "open_count": indicator.get("open_count"),
            "close_count": indicator.get("close_count"),
        }
        row.update(_signal_stats(signal_file))
        row.update(_exposure_stats(log_file))
        results.append(row)
        _write_rows(REPORT_DIR / "summary.csv", results)
        print(f"[{index}/{len(VARIANTS)}] {variant['name']} annual={row.get('annual')} sharpe={row.get('sharpe')} avg_inv={row.get('avg_invested_pct')}")
    _write_rows(
        REPORT_DIR / "summary_by_objective.csv",
        sorted(results, key=lambda row: min(_metric(row, "annual") / 3.0, _metric(row, "sharpe") / 4.0, _metric(row, "avg_invested_pct") / 0.80), reverse=True),
    )
    _write_rows(REPORT_DIR / "summary_by_sharpe.csv", sorted(results, key=lambda row: _metric(row, "sharpe"), reverse=True))
    _write_rows(REPORT_DIR / "summary_by_annual.csv", sorted(results, key=lambda row: _metric(row, "annual"), reverse=True))
    _write_rows(REPORT_DIR / "summary_target_hits.csv", [row for row in results if _metric(row, "annual") >= 3.0 and _metric(row, "sharpe") >= 4.0 and _metric(row, "avg_invested_pct") >= 0.80])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
