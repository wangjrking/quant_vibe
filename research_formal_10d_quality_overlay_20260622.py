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
REPORT_DIR = DATA_DIR / "reports" / "strategy_agent_model_application_20260621" / "formal_10d_quality_overlay_20260622"
SOURCE_SIGNAL = DATA_DIR / "reports" / "strategy_agent_model_application_20260621" / "formal_10d_gap02_bucket_target_20260622" / "signals" / "good_lowamount036_s030_w020_bad014.csv"
SCORE_DB = DATA_DIR / "reports" / "strategy_agent_model_application_20260621" / "formal_5d10d_weak_day_pool_replace_20260621" / "weak_day_pool_scores.db"
SCORE_TABLE = "weak_f60_liq_primary_count_le1_prank_10d90_5d10_frank_10d60_5d40_20000p0_0p5"
MARKET_DB = DATA_DIR / "STOCK_DAILY_DATA.db"
STRATEGY_DIR = Path(r"D:\work\dfcf\juejin\strategy\ce327750-634e-11f1-b8d7-10ffe0295517")
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")


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


def _variant(
    name: str,
    good_boost: float,
    excellent_boost: float,
    bad_cap: float,
    weak_bad_cap: float,
    global_cap: float,
    extra_env: dict[str, str] | None = None,
) -> dict:
    env = dict(BASE_ENV)
    if extra_env:
        env.update(extra_env)
    return {
        "name": name,
        "good_boost": good_boost,
        "excellent_boost": excellent_boost,
        "bad_cap": bad_cap,
        "weak_bad_cap": weak_bad_cap,
        "global_cap": global_cap,
        "env": env,
    }


VARIANTS = [
    _variant("q1_good040_bad010_cap40", 0.40, 0.42, 0.10, 0.08, 0.40),
    _variant("q1_good038_bad010_cap38", 0.38, 0.40, 0.10, 0.08, 0.38),
    _variant("q1_good036_bad010_cap36", 0.36, 0.38, 0.10, 0.08, 0.36),
    _variant("q1_good040_bad012_cap40", 0.40, 0.42, 0.12, 0.10, 0.40),
    _variant("q1_good038_bad012_cap38", 0.38, 0.40, 0.12, 0.10, 0.38),
    _variant("q1_good036_bad012_cap36", 0.36, 0.38, 0.12, 0.10, 0.36),
    _variant("q2_good040_bad008_cap40", 0.40, 0.44, 0.08, 0.06, 0.40),
    _variant("q2_good038_bad008_cap38", 0.38, 0.42, 0.08, 0.06, 0.38),
    _variant("q2_good036_bad008_cap36", 0.36, 0.40, 0.08, 0.06, 0.36),
    _variant("q3_good042_bad010_cap42", 0.42, 0.46, 0.10, 0.08, 0.42),
    _variant("q3_good040_bad010_cap42", 0.40, 0.44, 0.10, 0.08, 0.42),
    _variant("q3_good038_bad010_cap42", 0.38, 0.42, 0.10, 0.08, 0.42),
    _variant("q4_good040_bad014_cap40", 0.40, 0.42, 0.14, 0.12, 0.40),
    _variant("q4_good038_bad014_cap38", 0.38, 0.40, 0.14, 0.12, 0.38),
    _variant("q5_good040_bad010_strict", 0.40, 0.42, 0.10, 0.08, 0.40, {"GM_EQUITY_DD_SOFT_TRIGGER": "0.08", "GM_EQUITY_DD_HARD_TRIGGER": "0.16", "GM_EQUITY_DD_SOFT_SCALE": "0.80", "GM_EQUITY_DD_HARD_SCALE": "0.55"}),
    _variant("q5_good038_bad010_strict", 0.38, 0.40, 0.10, 0.08, 0.38, {"GM_EQUITY_DD_SOFT_TRIGGER": "0.08", "GM_EQUITY_DD_HARD_TRIGGER": "0.16", "GM_EQUITY_DD_SOFT_SCALE": "0.80", "GM_EQUITY_DD_HARD_SCALE": "0.55"}),
    _variant("q6_good040_bad012_loose", 0.40, 0.42, 0.12, 0.10, 0.40, {"GM_EQUITY_DD_SOFT_TRIGGER": "0.12", "GM_EQUITY_DD_HARD_TRIGGER": "0.22", "GM_EQUITY_DD_SOFT_SCALE": "0.90", "GM_EQUITY_DD_HARD_SCALE": "0.70"}),
    _variant("q6_good038_bad012_loose", 0.38, 0.40, 0.12, 0.10, 0.38, {"GM_EQUITY_DD_SOFT_TRIGGER": "0.12", "GM_EQUITY_DD_HARD_TRIGGER": "0.22", "GM_EQUITY_DD_SOFT_SCALE": "0.90", "GM_EQUITY_DD_HARD_SCALE": "0.70"}),
]


def _to_float(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return value


def _read_rows() -> tuple[list[dict], list[str]]:
    with SOURCE_SIGNAL.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader), list(reader.fieldnames or [])


def _is_good(row: dict) -> bool:
    amount = _to_float(row.get("amount"))
    turnover = _to_float(row.get("turnover_rate"))
    total_mv = _to_float(row.get("total_mv"))
    return (
        (amount is not None and amount < 5000)
        or (turnover is not None and turnover < 1)
        or (total_mv is not None and 50000 <= total_mv < 80000)
    )


def _is_excellent(row: dict) -> bool:
    amount = _to_float(row.get("amount"))
    turnover = _to_float(row.get("turnover_rate"))
    total_mv = _to_float(row.get("total_mv"))
    return (
        amount is not None
        and amount < 5000
        and (
            (turnover is not None and turnover < 1)
            or (total_mv is not None and 50000 <= total_mv < 80000)
        )
    )


def _is_bad(row: dict) -> bool:
    gap = _to_float(row.get("pred_gap"))
    pred10 = _to_float(row.get("pred_10d"))
    total_mv = _to_float(row.get("total_mv"))
    amount = _to_float(row.get("amount"))
    return (
        (gap is not None and 0.05 <= gap < 0.07)
        or (pred10 is not None and 0.10 <= pred10 < 0.13)
        or (total_mv is not None and 80000 <= total_mv < 120000)
        or (amount is not None and 20000 <= amount < 50000)
    )


def _write_signal(cfg: dict, signal_file: Path) -> dict:
    rows, fields = _read_rows()
    if "quality_role" not in fields:
        fields.append("quality_role")
    for row in rows:
        target = _to_float(row.get("target_pct"), 0.0) or 0.0
        quality_role = "base"
        if _is_excellent(row):
            target = max(target, float(cfg["excellent_boost"]))
            quality_role = "excellent"
        elif _is_good(row):
            target = max(target, float(cfg["good_boost"]))
            quality_role = "good"
        if _is_bad(row):
            target = min(target, float(cfg["bad_cap"]))
            quality_role = "bad"
        if quality_role == "bad" and str(row.get("pool_role") or "") == "weak":
            target = min(target, float(cfg["weak_bad_cap"]))
            quality_role = "weak_bad"
        target = min(target, float(cfg["global_cap"]))
        row["target_pct"] = f"{target:.5f}"
        row["quality_role"] = quality_role
    signal_file.parent.mkdir(parents=True, exist_ok=True)
    with signal_file.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    day_sums = {}
    role_counts = {}
    for row in rows:
        day_sums.setdefault(row["buy_date"], 0.0)
        day_sums[row["buy_date"]] += _to_float(row.get("target_pct"), 0.0) or 0.0
        role_counts[row["quality_role"]] = role_counts.get(row["quality_role"], 0) + 1
    return {
        "signal_count": len(rows),
        "buy_days": len(day_sums),
        "avg_day_target_sum": sum(day_sums.values()) / len(day_sums) if day_sums else None,
        "max_day_target_sum": max(day_sums.values()) if day_sums else None,
        "role_counts_json": json.dumps(role_counts, ensure_ascii=False, sort_keys=True),
    }


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


def _run_backtest(cfg: dict, signal_file: Path, log_file: Path) -> int:
    if log_file.exists() and _extract_indicator(log_file):
        return 0
    env = os.environ.copy()
    env.update({str(key): str(value) for key, value in cfg["env"].items()})
    cmd = [
        str(JUEJIN_PYTHON), str(MAIN / "run_juejin_signal_backtest.py"),
        "--strategy-dir", str(STRATEGY_DIR),
        "--signal-file", str(signal_file),
        "--log-file", str(log_file),
        "--max-positions", "6",
        "--holding-days", "6",
        "--max-holding-days", "6",
        "--target-position-pct", str(cfg["global_cap"]),
        "--score-db", str(SCORE_DB),
        "--score-table", SCORE_TABLE,
        "--market-db", str(MARKET_DB),
        "--backtest-start", "2024-06-05 09:00:00",
        "--backtest-end", "2026-06-18 15:30:00",
        "--backtest-adjust", "none",
        "--backtest-initial-cash", "600000",
        "--backtest-slippage-ratio", "0.0015",
    ]
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=str(MAIN), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    return proc.returncode


def _write_rows(path: Path, rows: list[dict]) -> None:
    fields = []
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


def _objective(row: dict) -> float:
    return min(_metric(row, "annual") / 3.0, _metric(row, "sharpe") / 4.0, _metric(row, "avg_invested_pct") / 0.80)


def main() -> int:
    (REPORT_DIR / "signals").mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for index, cfg in enumerate(VARIANTS, start=1):
        signal_file = REPORT_DIR / "signals" / f"{cfg['name']}.csv"
        log_file = REPORT_DIR / "logs" / f"{cfg['name']}.log"
        signal_stats = _write_signal(cfg, signal_file)
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
            "config_json": json.dumps({key: value for key, value in cfg.items() if key != "env"}, ensure_ascii=False, sort_keys=True),
            "signal_file": str(signal_file),
            "log_file": str(log_file),
        }
        row.update(signal_stats)
        row.update(_exposure_stats(log_file))
        results.append(row)
        _write_rows(REPORT_DIR / "summary.csv", results)
        print(f"[{index}/{len(VARIANTS)}] {cfg['name']} annual={row.get('annual')} sharpe={row.get('sharpe')} avg={row.get('avg_invested_pct')}")
    _write_rows(REPORT_DIR / "summary_by_objective.csv", sorted(results, key=_objective, reverse=True))
    _write_rows(REPORT_DIR / "summary_target_hits.csv", [row for row in results if _metric(row, "annual") >= 3.0 and _metric(row, "sharpe") >= 4.0 and _metric(row, "avg_invested_pct") >= 0.80])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
