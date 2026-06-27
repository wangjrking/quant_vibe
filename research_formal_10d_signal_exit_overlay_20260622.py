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
REPORT_DIR = DATA_DIR / "reports" / "strategy_agent_model_application_20260621" / "formal_10d_signal_exit_overlay_20260622"
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
    bad_hold: int,
    base_hold: int,
    good_hold: int,
    bad_exit_ratio: float,
    base_exit_ratio: float,
    good_exit_ratio: float,
    bad_min_hold: int,
    base_min_hold: int,
    good_min_hold: int,
    good_continue_ratio: float,
    max_holding_days: int,
    extra_env: dict[str, str] | None = None,
) -> dict:
    env = dict(BASE_ENV)
    if extra_env:
        env.update(extra_env)
    return {
        "name": name,
        "bad_hold": bad_hold,
        "base_hold": base_hold,
        "good_hold": good_hold,
        "bad_exit_ratio": bad_exit_ratio,
        "base_exit_ratio": base_exit_ratio,
        "good_exit_ratio": good_exit_ratio,
        "bad_min_hold": bad_min_hold,
        "base_min_hold": base_min_hold,
        "good_min_hold": good_min_hold,
        "good_continue_ratio": good_continue_ratio,
        "max_holding_days": max_holding_days,
        "env": env,
    }


VARIANTS = [
    _variant("e1_bad4_base6_good7", 4, 6, 7, 1.02, 1.00, 0.98, 2, 4, 5, 1.00, 7),
    _variant("e1_bad4_base6_good8", 4, 6, 8, 1.02, 1.00, 0.98, 2, 4, 5, 1.00, 8),
    _variant("e2_bad5_base6_good7", 5, 6, 7, 1.02, 1.00, 0.98, 2, 4, 5, 1.00, 7),
    _variant("e2_bad5_base6_good8", 5, 6, 8, 1.02, 1.00, 0.98, 2, 4, 5, 1.00, 8),
    _variant("e3_bad4_base6_good7_exit100", 4, 6, 7, 1.00, 1.00, 1.00, 2, 4, 5, 1.00, 7),
    _variant("e3_bad4_base6_good8_exit100", 4, 6, 8, 1.00, 1.00, 1.00, 2, 4, 5, 1.00, 8),
    _variant("e4_bad3_base6_good7", 3, 6, 7, 1.04, 1.00, 0.98, 1, 4, 5, 1.00, 7),
    _variant("e4_bad3_base6_good8", 3, 6, 8, 1.04, 1.00, 0.98, 1, 4, 5, 1.00, 8),
    _variant("e5_bad4_base5_good7", 4, 5, 7, 1.02, 1.00, 0.98, 2, 3, 5, 1.00, 7),
    _variant("e5_bad4_base5_good8", 4, 5, 8, 1.02, 1.00, 0.98, 2, 3, 5, 1.00, 8),
    _variant("e6_bad4_base6_good7_continue105", 4, 6, 7, 1.02, 1.00, 0.98, 2, 4, 5, 1.05, 9),
    _variant("e6_bad4_base6_good8_continue105", 4, 6, 8, 1.02, 1.00, 0.98, 2, 4, 5, 1.05, 10),
    _variant("e7_bad4_base6_good7_strict", 4, 6, 7, 1.02, 1.00, 0.98, 2, 4, 5, 1.00, 7, {"GM_EQUITY_DD_SOFT_TRIGGER": "0.08", "GM_EQUITY_DD_HARD_TRIGGER": "0.16", "GM_EQUITY_DD_SOFT_SCALE": "0.80", "GM_EQUITY_DD_HARD_SCALE": "0.55"}),
    _variant("e7_bad4_base6_good8_strict", 4, 6, 8, 1.02, 1.00, 0.98, 2, 4, 5, 1.00, 8, {"GM_EQUITY_DD_SOFT_TRIGGER": "0.08", "GM_EQUITY_DD_HARD_TRIGGER": "0.16", "GM_EQUITY_DD_SOFT_SCALE": "0.80", "GM_EQUITY_DD_HARD_SCALE": "0.55"}),
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


def _quality(row: dict) -> str:
    amount = _to_float(row.get("amount"))
    turnover = _to_float(row.get("turnover_rate"))
    total_mv = _to_float(row.get("total_mv"))
    gap = _to_float(row.get("pred_gap"))
    pred10 = _to_float(row.get("pred_10d"))
    bad = (
        (gap is not None and 0.05 <= gap < 0.07)
        or (pred10 is not None and 0.10 <= pred10 < 0.13)
        or (total_mv is not None and 80000 <= total_mv < 120000)
        or (amount is not None and 20000 <= amount < 50000)
    )
    good = (
        (amount is not None and amount < 5000)
        or (turnover is not None and turnover < 1)
        or (total_mv is not None and 50000 <= total_mv < 80000)
    )
    if bad:
        return "bad"
    if good:
        return "good"
    return "base"


def _write_signal(cfg: dict, signal_file: Path) -> dict:
    rows, fields = _read_rows()
    for field in [
        "quality_role",
        "signal_score_exit_entry_ratio",
        "signal_min_holding_days_before_score_exit",
        "signal_score_continue_entry_ratio",
    ]:
        if field not in fields:
            fields.append(field)
    counts = {"good": 0, "base": 0, "bad": 0}
    for row in rows:
        quality = _quality(row)
        counts[quality] += 1
        row["quality_role"] = quality
        if quality == "bad":
            row["holding_days"] = str(int(cfg["bad_hold"]))
            row["signal_score_exit_entry_ratio"] = f"{float(cfg['bad_exit_ratio']):.4f}"
            row["signal_min_holding_days_before_score_exit"] = str(int(cfg["bad_min_hold"]))
            row["signal_score_continue_entry_ratio"] = "1.10"
        elif quality == "good":
            row["holding_days"] = str(int(cfg["good_hold"]))
            row["signal_score_exit_entry_ratio"] = f"{float(cfg['good_exit_ratio']):.4f}"
            row["signal_min_holding_days_before_score_exit"] = str(int(cfg["good_min_hold"]))
            row["signal_score_continue_entry_ratio"] = f"{float(cfg['good_continue_ratio']):.4f}"
        else:
            row["holding_days"] = str(int(cfg["base_hold"]))
            row["signal_score_exit_entry_ratio"] = f"{float(cfg['base_exit_ratio']):.4f}"
            row["signal_min_holding_days_before_score_exit"] = str(int(cfg["base_min_hold"]))
            row["signal_score_continue_entry_ratio"] = "1.02"
    signal_file.parent.mkdir(parents=True, exist_ok=True)
    with signal_file.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return {"signal_count": len(rows), "quality_counts_json": json.dumps(counts, ensure_ascii=False, sort_keys=True)}


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
        "--max-holding-days", str(cfg["max_holding_days"]),
        "--target-position-pct", "0.36",
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
