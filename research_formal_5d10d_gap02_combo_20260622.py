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
REPORT_DIR = DATA_DIR / "reports" / "strategy_agent_model_application_20260621" / "formal_5d10d_gap02_combo_20260622"
CORE_SIGNAL = MAIN / "strategy_library" / "production" / "prod_formal_5d10d_gap_v20260622" / "signals" / "backtest_signals_gap_t0p03_g1p08_b0p88.csv"
GAP02_SIGNAL = DATA_DIR / "reports" / "strategy_agent_model_application_20260621" / "formal_10d_gap_strong_weak_fill_20260622" / "signals" / "gap02_s0p32_w0p18_h6_mh6.csv"
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


def _variant(name: str, *, priority: tuple[str, ...], core_scale: float, gap_strong: float, gap_weak: float, max_per_day: int, max_positions: int, cap: float) -> dict:
    return {
        "name": name,
        "priority": priority,
        "core_scale": core_scale,
        "gap_strong": gap_strong,
        "gap_weak": gap_weak,
        "max_per_day": max_per_day,
        "max_positions": max_positions,
        "cap": cap,
        "env": dict(BASE_ENV),
    }


VARIANTS = [
    _variant("core_first_gap_fill8", priority=("core", "gap"), core_scale=1.0, gap_strong=0.18, gap_weak=0.10, max_per_day=8, max_positions=10, cap=0.30),
    _variant("core_first_gap_fill10", priority=("core", "gap"), core_scale=1.0, gap_strong=0.16, gap_weak=0.08, max_per_day=10, max_positions=12, cap=0.30),
    _variant("gap_first_core_fill8", priority=("gap", "core"), core_scale=0.70, gap_strong=0.30, gap_weak=0.18, max_per_day=8, max_positions=10, cap=0.30),
    _variant("gap_first_core_fill10", priority=("gap", "core"), core_scale=0.60, gap_strong=0.30, gap_weak=0.16, max_per_day=10, max_positions=12, cap=0.30),
    _variant("balanced_core_gap8", priority=("core", "gap"), core_scale=0.85, gap_strong=0.22, gap_weak=0.12, max_per_day=8, max_positions=10, cap=0.26),
    _variant("balanced_gap_core8", priority=("gap", "core"), core_scale=0.85, gap_strong=0.26, gap_weak=0.14, max_per_day=8, max_positions=10, cap=0.30),
]


def _to_float(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return value


def _load(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _normalize(row: dict, source: str, cfg: dict) -> dict:
    out = dict(row)
    out["source_signal"] = source
    out["holding_days"] = "6" if source == "gap" else str(row.get("holding_days") or "7")
    if source == "core":
        target = (_to_float(row.get("target_pct"), 0.0) or 0.0) * float(cfg["core_scale"])
    else:
        target = float(cfg["gap_strong"] if str(row.get("pool_role") or "") == "strong" else cfg["gap_weak"])
    out["target_pct"] = f"{target:.5f}"
    out["signal_score_exit_entry_ratio"] = str(row.get("score_exit_entry_ratio") or "1.0")
    out["signal_min_holding_days_before_score_exit"] = str(row.get("min_holding_days_before_score_exit") or "4")
    return out


def _write_signal(cfg: dict, output: Path) -> None:
    core_rows = [_normalize(row, "core", cfg) for row in _load(CORE_SIGNAL)]
    gap_rows = [_normalize(row, "gap", cfg) for row in _load(GAP02_SIGNAL)]
    by_date_source = {
        "core": {},
        "gap": {},
    }
    for row in core_rows:
        by_date_source["core"].setdefault(str(row.get("buy_date")), []).append(row)
    for row in gap_rows:
        by_date_source["gap"].setdefault(str(row.get("buy_date")), []).append(row)
    dates = sorted(set(by_date_source["core"]) | set(by_date_source["gap"]))
    fields: list[str] = []
    merged: list[dict] = []
    for date in dates:
        used: set[str] = set()
        day_rows: list[dict] = []
        for source in cfg["priority"]:
            rows = by_date_source[source].get(date, [])
            rows = sorted(rows, key=lambda row: int(float(row.get("rank") or 999999)))
            for row in rows:
                stock = str(row.get("stock_code") or "")
                if not stock or stock in used:
                    continue
                used.add(stock)
                day_rows.append(row)
                if len(day_rows) >= int(cfg["max_per_day"]):
                    break
            if len(day_rows) >= int(cfg["max_per_day"]):
                break
        merged.extend(day_rows)
        for row in day_rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    for field in ["source_signal", "signal_score_exit_entry_ratio", "signal_min_holding_days_before_score_exit"]:
        if field not in fields:
            fields.append(field)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in fields} for row in merged])


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
    rows = _load(path)
    counts = {}
    sources = {}
    for row in rows:
        counts[str(row.get("buy_date"))] = counts.get(str(row.get("buy_date")), 0) + 1
        sources[str(row.get("source_signal"))] = sources.get(str(row.get("source_signal")), 0) + 1
    return {
        "signal_count": len(rows),
        "buy_days": len(counts),
        "avg_signals_per_day": len(rows) / len(counts) if counts else None,
        "source_counts_json": json.dumps(sources, ensure_ascii=False, sort_keys=True),
    }


def _run_backtest(cfg: dict, signal_file: Path, log_file: Path) -> int:
    if log_file.exists() and _extract_indicator(log_file):
        return 0
    env = os.environ.copy()
    env.update({str(key): str(value) for key, value in cfg["env"].items()})
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
        str(int(cfg["max_positions"])),
        "--holding-days",
        "6",
        "--max-holding-days",
        "8",
        "--target-position-pct",
        str(float(cfg["cap"])),
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
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _metric(row: dict, key: str) -> float:
    return _to_float(row.get(key), float("-inf"))


def main() -> int:
    (REPORT_DIR / "signals").mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for index, cfg in enumerate(VARIANTS, start=1):
        signal_file = REPORT_DIR / "signals" / f"{cfg['name']}.csv"
        log_file = REPORT_DIR / "logs" / f"{cfg['name']}.log"
        _write_signal(cfg, signal_file)
        returncode = _run_backtest(cfg, signal_file, log_file)
        indicator = _extract_indicator(log_file) or {}
        row = {
            "name": cfg["name"],
            "returncode": returncode,
            "annual": indicator.get("pnl_ratio_annual", indicator.get("annual_return")),
            "sharpe": indicator.get("sharp_ratio", indicator.get("sharpe_ratio")),
            "max_drawdown": indicator.get("max_drawdown"),
            "win_ratio": indicator.get("win_ratio"),
            "open_count": indicator.get("open_count"),
            "close_count": indicator.get("close_count"),
            "signal_file": str(signal_file),
            "log_file": str(log_file),
            "config_json": json.dumps({key: value for key, value in cfg.items() if key != "env"}, ensure_ascii=False, sort_keys=True),
        }
        row.update(_signal_stats(signal_file))
        row.update(_exposure_stats(log_file))
        results.append(row)
        _write_rows(REPORT_DIR / "summary.csv", results)
        print(f"[{index}/{len(VARIANTS)}] {cfg['name']} annual={row.get('annual')} sharpe={row.get('sharpe')} avg={row.get('avg_invested_pct')}")
    _write_rows(REPORT_DIR / "summary_by_objective.csv", sorted(results, key=lambda row: min(_metric(row, "annual") / 3.0, _metric(row, "sharpe") / 4.0, _metric(row, "avg_invested_pct") / 0.80), reverse=True))
    _write_rows(REPORT_DIR / "summary_target_hits.csv", [row for row in results if _metric(row, "annual") >= 3.0 and _metric(row, "sharpe") >= 4.0 and _metric(row, "avg_invested_pct") >= 0.80])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
