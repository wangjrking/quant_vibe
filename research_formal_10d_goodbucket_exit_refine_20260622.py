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
REPORT_DIR = DATA_DIR / "reports" / "strategy_agent_model_application_20260621" / "formal_10d_goodbucket_exit_refine_20260622"
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
    extra_env: dict[str, str] | None = None,
    max_daily_sells: int = 1,
    signal_holding_days: int | None = None,
    max_holding_days: int = 6,
) -> dict:
    env = dict(BASE_ENV)
    env["GM_MAX_DAILY_SELLS"] = str(max_daily_sells)
    if extra_env:
        env.update(extra_env)
    return {
        "name": name,
        "env": env,
        "signal_holding_days": signal_holding_days,
        "max_holding_days": max_holding_days,
    }


VARIANTS = [
    _variant("base"),
    _variant("score_exit098_min3", {"GM_SCORE_EXIT_ENTRY_RATIO": "0.98", "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "3"}),
    _variant("score_exit100_min5", {"GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "5"}),
    _variant("score_exit_off", {"GM_OPEN_DAILY_SCORE_EXIT": "0"}),
    _variant("max_sells2", max_daily_sells=2),
    _variant("eqdd_loose", {"GM_EQUITY_DD_SOFT_TRIGGER": "0.12", "GM_EQUITY_DD_HARD_TRIGGER": "0.22", "GM_EQUITY_DD_SOFT_SCALE": "0.90", "GM_EQUITY_DD_HARD_SCALE": "0.70"}),
    _variant("eqdd_strict", {"GM_EQUITY_DD_SOFT_TRIGGER": "0.08", "GM_EQUITY_DD_HARD_TRIGGER": "0.16", "GM_EQUITY_DD_SOFT_SCALE": "0.80", "GM_EQUITY_DD_HARD_SCALE": "0.55"}),
    _variant("resize_recover03", {"GM_EQUITY_DD_RESIZE_EXISTING": "1", "GM_EQUITY_DD_RECOVER_TRIGGER": "0.03"}),
    _variant("light_stop05_h1", {"GM_LIGHT_STOP_LOSS_PCT": "0.05", "GM_MIN_HOLDING_DAYS_BEFORE_LIGHT_STOP": "1"}),
    _variant("intraday_sl06_half", {"GM_INTRADAY_RISK_MODE": "1", "GM_STOP_LOSS_PCT": "0.06", "GM_INTRADAY_RISK_SELL_FRACTION": "0.5"}),
    _variant("intraday_sl06", {"GM_INTRADAY_RISK_MODE": "1", "GM_STOP_LOSS_PCT": "0.06"}),
    _variant("intraday_sl08_tp16", {"GM_INTRADAY_RISK_MODE": "1", "GM_STOP_LOSS_PCT": "0.08", "GM_TAKE_PROFIT_PCT": "0.16"}),
    _variant("hold5", signal_holding_days=5, max_holding_days=5),
    _variant("hold7", signal_holding_days=7, max_holding_days=7),
    _variant("hold8", signal_holding_days=8, max_holding_days=8),
    _variant("hold7_continue100", {"GM_SCORE_CONTINUE_ENTRY_RATIO": "1.00"}, signal_holding_days=7, max_holding_days=9),
    _variant("hold7_continue105", {"GM_SCORE_CONTINUE_ENTRY_RATIO": "1.05"}, signal_holding_days=7, max_holding_days=9),
    _variant("sell_unlimited", max_daily_sells=0),
    _variant("defer_no_signal_h8", {"GM_DEFER_EXITS_WITHOUT_BUY_SIGNAL": "1", "GM_NO_SIGNAL_MAX_HOLDING_DAYS": "8"}),
    _variant("defer_no_signal_h10", {"GM_DEFER_EXITS_WITHOUT_BUY_SIGNAL": "1", "GM_NO_SIGNAL_MAX_HOLDING_DAYS": "10"}),
]


def _to_float(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return value


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


def _load_signal_rows() -> tuple[list[dict], list[str]]:
    with SOURCE_SIGNAL.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader), list(reader.fieldnames or [])


def _write_signal(variant: dict, signal_file: Path) -> None:
    rows, fields = _load_signal_rows()
    holding_days = variant.get("signal_holding_days")
    if holding_days is not None:
        if "holding_days" not in fields:
            fields.append("holding_days")
        for row in rows:
            row["holding_days"] = str(int(holding_days))
    signal_file.parent.mkdir(parents=True, exist_ok=True)
    with signal_file.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _run_backtest(variant: dict, signal_file: Path, log_file: Path) -> int:
    if log_file.exists() and _extract_indicator(log_file):
        return 0
    env = os.environ.copy()
    env.update({str(key): str(value) for key, value in variant["env"].items()})
    cmd = [
        str(JUEJIN_PYTHON), str(MAIN / "run_juejin_signal_backtest.py"),
        "--strategy-dir", str(STRATEGY_DIR),
        "--signal-file", str(signal_file),
        "--log-file", str(log_file),
        "--max-positions", "6",
        "--holding-days", "6",
        "--max-holding-days", str(variant.get("max_holding_days") or 6),
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
    (REPORT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "signals").mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for index, variant in enumerate(VARIANTS, start=1):
        signal_file = REPORT_DIR / "signals" / f"{variant['name']}.csv"
        log_file = REPORT_DIR / "logs" / f"{variant['name']}.log"
        _write_signal(variant, signal_file)
        returncode = _run_backtest(variant, signal_file, log_file)
        indicator = _extract_indicator(log_file) or {}
        row = {
            "name": variant["name"],
            "returncode": returncode,
            "annual": indicator.get("pnl_ratio_annual", indicator.get("annual_return")),
            "sharpe": indicator.get("sharp_ratio", indicator.get("sharpe_ratio")),
            "max_drawdown": indicator.get("max_drawdown"),
            "win_ratio": indicator.get("win_ratio"),
            "open_count": indicator.get("open_count"),
            "close_count": indicator.get("close_count"),
            "env_json": json.dumps(variant["env"], ensure_ascii=False, sort_keys=True),
            "signal_holding_days": variant.get("signal_holding_days"),
            "max_holding_days": variant.get("max_holding_days"),
            "signal_file": str(signal_file),
            "log_file": str(log_file),
        }
        row.update(_exposure_stats(log_file))
        results.append(row)
        _write_rows(REPORT_DIR / "summary.csv", results)
        print(f"[{index}/{len(VARIANTS)}] {variant['name']} annual={row.get('annual')} sharpe={row.get('sharpe')} avg={row.get('avg_invested_pct')}")
    _write_rows(REPORT_DIR / "summary_by_objective.csv", sorted(results, key=lambda row: min(_metric(row, "annual") / 3.0, _metric(row, "sharpe") / 4.0, _metric(row, "avg_invested_pct") / 0.80), reverse=True))
    _write_rows(REPORT_DIR / "summary_target_hits.csv", [row for row in results if _metric(row, "annual") >= 3.0 and _metric(row, "sharpe") >= 4.0 and _metric(row, "avg_invested_pct") >= 0.80])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
