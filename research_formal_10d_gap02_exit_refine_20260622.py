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
REPORT_DIR = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_10d_gap02_exit_refine_20260622"
)
SOURCE_SIGNAL = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_10d_gap_strong_weak_fill_20260622"
    / "signals"
    / "gap02_s0p32_w0p18_h6_mh6.csv"
)
SCORE_DB = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_5d10d_weak_day_pool_replace_20260621"
    / "weak_day_pool_scores.db"
)
SCORE_TABLE = "weak_f60_liq_primary_count_le1_prank_10d90_5d10_frank_10d60_5d40_20000p0_0p5"
MARKET_DB = DATA_DIR / "STOCK_DAILY_DATA.db"
STRATEGY_DIR = Path(r"D:\work\dfcf\juejin\strategy\ce327750-634e-11f1-b8d7-10ffe0295517")
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")
BACKTEST_START = "2024-06-05 09:00:00"
BACKTEST_END = "2026-06-18 15:30:00"


BASE_ENV = {
    "GM_OPEN_DAILY_SCORE_EXIT": "1",
    "GM_SCORE_EXIT_ENTRY_RATIO": "1.0",
    "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "3",
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


def _env(**kwargs: str) -> dict[str, str]:
    out = dict(BASE_ENV)
    out.update({key: str(value) for key, value in kwargs.items()})
    return out


VARIANTS = [
    {"name": "base_eqdd_best", "env": _env()},
    {"name": "exit098_min2_sell1", "env": _env(GM_SCORE_EXIT_ENTRY_RATIO="0.98", GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT="2")},
    {"name": "exit098_min2_sell2", "env": _env(GM_SCORE_EXIT_ENTRY_RATIO="0.98", GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT="2", GM_MAX_DAILY_SELLS="2")},
    {"name": "exit095_min2_sell2", "env": _env(GM_SCORE_EXIT_ENTRY_RATIO="0.95", GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT="2", GM_MAX_DAILY_SELLS="2")},
    {"name": "exit102_min2_sell2", "env": _env(GM_SCORE_EXIT_ENTRY_RATIO="1.02", GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT="2", GM_MAX_DAILY_SELLS="2")},
    {"name": "exit100_min1_sell2", "env": _env(GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT="1", GM_MAX_DAILY_SELLS="2")},
    {"name": "exit100_min4_sell1", "env": _env(GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT="4")},
    {"name": "exit100_min3_sell2", "env": _env(GM_MAX_DAILY_SELLS="2")},
    {"name": "cont100", "env": _env(GM_SCORE_CONTINUE_ENTRY_RATIO="1.00")},
    {"name": "cont105", "env": _env(GM_SCORE_CONTINUE_ENTRY_RATIO="1.05")},
    {"name": "loose_eqdd", "env": _env(GM_EQUITY_DD_SOFT_TRIGGER="0.12", GM_EQUITY_DD_HARD_TRIGGER="0.22", GM_EQUITY_DD_SOFT_SCALE="0.90", GM_EQUITY_DD_HARD_SCALE="0.70")},
    {"name": "strict_eqdd", "env": _env(GM_EQUITY_DD_SOFT_TRIGGER="0.08", GM_EQUITY_DD_HARD_TRIGGER="0.16", GM_EQUITY_DD_SOFT_SCALE="0.80", GM_EQUITY_DD_HARD_SCALE="0.55")},
    {"name": "intraday_sl06", "env": _env(GM_INTRADAY_RISK_MODE="1", GM_STOP_LOSS_PCT="0.06", GM_TAKE_PROFIT_PCT="none")},
    {"name": "intraday_sl06_half", "env": _env(GM_INTRADAY_RISK_MODE="1", GM_STOP_LOSS_PCT="0.06", GM_TAKE_PROFIT_PCT="none", GM_INTRADAY_RISK_SELL_FRACTION="0.5")},
    {"name": "intraday_sl08_tp16", "env": _env(GM_INTRADAY_RISK_MODE="1", GM_STOP_LOSS_PCT="0.08", GM_TAKE_PROFIT_PCT="0.16")},
    {"name": "exit098_min2_intraday_sl06", "env": _env(GM_SCORE_EXIT_ENTRY_RATIO="0.98", GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT="2", GM_INTRADAY_RISK_MODE="1", GM_STOP_LOSS_PCT="0.06")},
    {"name": "exit100_min5_sell1", "env": _env(GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT="5")},
    {"name": "exit100_min6_sell1", "env": _env(GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT="6")},
    {"name": "score_exit_off", "env": _env(GM_OPEN_DAILY_SCORE_EXIT="0")},
    {"name": "hold7_mh7_min4", "holding_days": 7, "max_holding_days": 7, "env": _env(GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT="4")},
    {"name": "hold7_mh8_min4", "holding_days": 7, "max_holding_days": 8, "env": _env(GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT="4")},
    {"name": "hold6_mh8_min4", "holding_days": 6, "max_holding_days": 8, "env": _env(GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT="4")},
    {"name": "maxpos5_min4", "max_positions": 5, "env": _env(GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT="4")},
    {"name": "maxpos7_min4", "max_positions": 7, "env": _env(GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT="4")},
    {"name": "target030_min4", "target_pct": 0.30, "env": _env(GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT="4")},
    {"name": "target034_min4", "target_pct": 0.34, "env": _env(GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT="4")},
    {"name": "target026_min4", "target_pct": 0.26, "env": _env(GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT="4")},
    {"name": "target028_min4", "target_pct": 0.28, "env": _env(GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT="4")},
    {"name": "target029_min4", "target_pct": 0.29, "env": _env(GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT="4")},
    {"name": "target031_min4", "target_pct": 0.31, "env": _env(GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT="4")},
    {"name": "target030_min5", "target_pct": 0.30, "env": _env(GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT="5")},
    {"name": "target030_exitoff", "target_pct": 0.30, "env": _env(GM_OPEN_DAILY_SCORE_EXIT="0")},
    {"name": "target030_breadth_up06_scale80", "target_pct": 0.30, "env": _env(GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT="4", GM_BREADTH_RISK_EXIT_MODE="1", GM_BREADTH_RISK_UP_RATIO_THRESHOLD="0.06", GM_BREADTH_RISK_AVG_PCT_THRESHOLD="-3.0", GM_BREADTH_RISK_BUY_SCALE="0.80")},
    {"name": "target030_breadth_up08_scale80", "target_pct": 0.30, "env": _env(GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT="4", GM_BREADTH_RISK_EXIT_MODE="1", GM_BREADTH_RISK_UP_RATIO_THRESHOLD="0.08", GM_BREADTH_RISK_AVG_PCT_THRESHOLD="-3.0", GM_BREADTH_RISK_BUY_SCALE="0.80")},
    {"name": "target030_breadth_up10_scale80", "target_pct": 0.30, "env": _env(GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT="4", GM_BREADTH_RISK_EXIT_MODE="1", GM_BREADTH_RISK_UP_RATIO_THRESHOLD="0.10", GM_BREADTH_RISK_AVG_PCT_THRESHOLD="-2.5", GM_BREADTH_RISK_BUY_SCALE="0.80")},
    {"name": "target030_breadth_up12_scale85", "target_pct": 0.30, "env": _env(GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT="4", GM_BREADTH_RISK_EXIT_MODE="1", GM_BREADTH_RISK_UP_RATIO_THRESHOLD="0.12", GM_BREADTH_RISK_AVG_PCT_THRESHOLD="-2.0", GM_BREADTH_RISK_BUY_SCALE="0.85")},
    {"name": "target030_index_m03_scale80", "target_pct": 0.30, "env": _env(GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT="4", GM_INDEX_RISK_EXIT_MODE="1", GM_INDEX_RISK_CC_THRESHOLD="-0.030", GM_INDEX_RISK_INTRADAY_THRESHOLD="-0.030", GM_INDEX_RISK_BUY_SCALE="0.80")},
    {"name": "target030_index_m04_scale85", "target_pct": 0.30, "env": _env(GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT="4", GM_INDEX_RISK_EXIT_MODE="1", GM_INDEX_RISK_CC_THRESHOLD="-0.040", GM_INDEX_RISK_INTRADAY_THRESHOLD="-0.040", GM_INDEX_RISK_BUY_SCALE="0.85")},
    {"name": "target030_breadth_up08_index_m04", "target_pct": 0.30, "env": _env(GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT="4", GM_BREADTH_RISK_EXIT_MODE="1", GM_BREADTH_RISK_UP_RATIO_THRESHOLD="0.08", GM_BREADTH_RISK_AVG_PCT_THRESHOLD="-3.0", GM_BREADTH_RISK_BUY_SCALE="0.85", GM_INDEX_RISK_EXIT_MODE="1", GM_INDEX_RISK_CC_THRESHOLD="-0.040", GM_INDEX_RISK_INTRADAY_THRESHOLD="-0.040", GM_INDEX_RISK_BUY_SCALE="0.85")},
    {"name": "target030_resize_eqdd_mid", "target_pct": 0.30, "env": _env(GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT="4", GM_EQUITY_DD_RESIZE_EXISTING="1")},
    {"name": "target030_resize_eqdd_loose", "target_pct": 0.30, "env": _env(GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT="4", GM_EQUITY_DD_RESIZE_EXISTING="1", GM_EQUITY_DD_SOFT_TRIGGER="0.12", GM_EQUITY_DD_HARD_TRIGGER="0.22", GM_EQUITY_DD_SOFT_SCALE="0.90", GM_EQUITY_DD_HARD_SCALE="0.70")},
    {"name": "target030_resize_eqdd_strict", "target_pct": 0.30, "env": _env(GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT="4", GM_EQUITY_DD_RESIZE_EXISTING="1", GM_EQUITY_DD_SOFT_TRIGGER="0.08", GM_EQUITY_DD_HARD_TRIGGER="0.16", GM_EQUITY_DD_SOFT_SCALE="0.80", GM_EQUITY_DD_HARD_SCALE="0.55")},
    {"name": "target030_resize_recover03", "target_pct": 0.30, "env": _env(GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT="4", GM_EQUITY_DD_RESIZE_EXISTING="1", GM_EQUITY_DD_RECOVER_TRIGGER="0.03")},
    {"name": "target032_resize_recover03", "target_pct": 0.32, "env": _env(GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT="4", GM_EQUITY_DD_RESIZE_EXISTING="1", GM_EQUITY_DD_RECOVER_TRIGGER="0.03")},
    {"name": "target034_resize_recover03", "target_pct": 0.34, "env": _env(GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT="4", GM_EQUITY_DD_RESIZE_EXISTING="1", GM_EQUITY_DD_RECOVER_TRIGGER="0.03")},
    {"name": "target036_resize_recover03", "target_pct": 0.36, "env": _env(GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT="4", GM_EQUITY_DD_RESIZE_EXISTING="1", GM_EQUITY_DD_RECOVER_TRIGGER="0.03")},
    {"name": "min4_noeqdd", "env": _env(GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT="4", GM_EQUITY_DD_RISK_MODE="0")},
    {"name": "min4_strict_light", "env": _env(GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT="4", GM_LIGHT_STOP_LOSS_PCT="0.05", GM_MIN_HOLDING_DAYS_BEFORE_LIGHT_STOP="1")},
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


def _load_signal_stats() -> dict:
    with SOURCE_SIGNAL.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    targets = [_to_float(row.get("target_pct")) for row in rows]
    targets = [value for value in targets if value is not None]
    return {
        "signal_count": len(rows),
        "buy_days": len({row.get("buy_date") for row in rows if row.get("buy_date")}),
        "avg_signal_target_pct": sum(targets) / len(targets) if targets else None,
        "max_signal_target_pct": max(targets) if targets else None,
    }


def _run_backtest(variant: dict, log_file: Path) -> int:
    if log_file.exists() and _extract_indicator(log_file):
        return 0
    env = os.environ.copy()
    env.update(variant["env"])
    command = [
        str(JUEJIN_PYTHON),
        str(MAIN / "run_juejin_signal_backtest.py"),
        "--strategy-dir",
        str(STRATEGY_DIR),
        "--signal-file",
        str(SOURCE_SIGNAL),
        "--log-file",
        str(log_file),
        "--max-positions",
        str(int(variant.get("max_positions", 6))),
        "--holding-days",
        str(int(variant.get("holding_days", 6))),
        "--max-holding-days",
        str(int(variant.get("max_holding_days", 6))),
        "--target-position-pct",
        str(float(variant.get("target_pct", 0.32))),
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
    (REPORT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    signal_stats = _load_signal_stats()
    results: list[dict] = []
    for index, variant in enumerate(VARIANTS, start=1):
        log_file = REPORT_DIR / "logs" / f"{variant['name']}.log"
        returncode = _run_backtest(variant, log_file)
        indicator = _extract_indicator(log_file) or {}
        row = {
            "name": variant["name"],
            "returncode": returncode,
            "signal_file": str(SOURCE_SIGNAL),
            "log_file": str(log_file),
            "extra_env_json": json.dumps(variant["env"], ensure_ascii=False, sort_keys=True),
            "annual": indicator.get("pnl_ratio_annual", indicator.get("annual_return")),
            "sharpe": indicator.get("sharp_ratio", indicator.get("sharpe_ratio")),
            "max_drawdown": indicator.get("max_drawdown"),
            "win_ratio": indicator.get("win_ratio"),
            "open_count": indicator.get("open_count"),
            "close_count": indicator.get("close_count"),
        }
        row.update(signal_stats)
        row.update(_exposure_stats(log_file))
        results.append(row)
        _write_rows(REPORT_DIR / "summary.csv", results)
        print(
            f"[{index}/{len(VARIANTS)}] {variant['name']} "
            f"annual={row.get('annual')} sharpe={row.get('sharpe')} avg_inv={row.get('avg_invested_pct')}"
        )
    _write_rows(
        REPORT_DIR / "summary_by_objective.csv",
        sorted(
            results,
            key=lambda row: min(_metric(row, "annual") / 3.0, _metric(row, "sharpe") / 4.0, _metric(row, "avg_invested_pct") / 0.80),
            reverse=True,
        ),
    )
    _write_rows(REPORT_DIR / "summary_by_sharpe.csv", sorted(results, key=lambda row: _metric(row, "sharpe"), reverse=True))
    _write_rows(REPORT_DIR / "summary_by_annual.csv", sorted(results, key=lambda row: _metric(row, "annual"), reverse=True))
    _write_rows(
        REPORT_DIR / "summary_target_hits.csv",
        [
            row
            for row in results
            if _metric(row, "annual") >= 3.0 and _metric(row, "sharpe") >= 4.0 and _metric(row, "avg_invested_pct") >= 0.80
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
