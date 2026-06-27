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
REPORT_DIR = DATA_DIR / "reports" / "strategy_agent_model_application_20260621" / "formal_10d_gap02_role_exit_20260622"
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


def _variant(name: str, **kwargs) -> dict:
    cfg = {
        "name": name,
        "target_cap": 0.30,
        "weak_stop": None,
        "strong_stop": None,
        "weak_take": None,
        "strong_take": None,
        "weak_score_exit": None,
        "strong_score_exit": None,
        "weak_min_score_hold": None,
        "strong_min_score_hold": None,
        "intraday": False,
        "sell_fraction": None,
        "env": dict(BASE_ENV),
    }
    cfg.update(kwargs)
    if cfg["intraday"]:
        cfg["env"]["GM_INTRADAY_RISK_MODE"] = "1"
    if cfg["sell_fraction"] is not None:
        cfg["env"]["GM_INTRADAY_RISK_SELL_FRACTION"] = str(cfg["sell_fraction"])
    return cfg


VARIANTS = [
    _variant("base_target030"),
    _variant("weak_stop06_intraday", intraday=True, weak_stop=0.06),
    _variant("weak_stop05_intraday", intraday=True, weak_stop=0.05),
    _variant("weak_stop04_intraday", intraday=True, weak_stop=0.04),
    _variant("weak_stop06_half_intraday", intraday=True, weak_stop=0.06, sell_fraction=0.5),
    _variant("weak_take12_intraday", intraday=True, weak_take=0.12),
    _variant("weak_stop06_take12_intraday", intraday=True, weak_stop=0.06, weak_take=0.12),
    _variant("weak_score098_min2", weak_score_exit=0.98, weak_min_score_hold=2),
    _variant("weak_score102_min2", weak_score_exit=1.02, weak_min_score_hold=2),
    _variant("weak_score098_min2_stop06", intraday=True, weak_score_exit=0.98, weak_min_score_hold=2, weak_stop=0.06),
    _variant("strong_loose_weak_tight", intraday=True, weak_stop=0.06, strong_stop=0.10, weak_score_exit=0.98, weak_min_score_hold=2, strong_score_exit=1.02, strong_min_score_hold=5),
    _variant("weak_stop06_strong_take18", intraday=True, weak_stop=0.06, strong_take=0.18),
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


def _write_signal(cfg: dict, signal_file: Path) -> None:
    rows = _load_rows(SOURCE_SIGNAL)
    extra_fields = [
        "signal_stop_loss_pct",
        "signal_take_profit_pct",
        "signal_score_exit_entry_ratio",
        "signal_min_holding_days_before_score_exit",
    ]
    fieldnames = list(rows[0].keys())
    for field in extra_fields:
        if field not in fieldnames:
            fieldnames.append(field)
    for row in rows:
        role = str(row.get("pool_role") or "weak")
        row["holding_days"] = "6"
        if role == "strong":
            stop = cfg.get("strong_stop")
            take = cfg.get("strong_take")
            score_exit = cfg.get("strong_score_exit")
            min_hold = cfg.get("strong_min_score_hold")
        else:
            stop = cfg.get("weak_stop")
            take = cfg.get("weak_take")
            score_exit = cfg.get("weak_score_exit")
            min_hold = cfg.get("weak_min_score_hold")
        if stop is not None:
            row["signal_stop_loss_pct"] = f"{float(stop):.5f}"
        if take is not None:
            row["signal_take_profit_pct"] = f"{float(take):.5f}"
        if score_exit is not None:
            row["signal_score_exit_entry_ratio"] = f"{float(score_exit):.5f}"
        if min_hold is not None:
            row["signal_min_holding_days_before_score_exit"] = str(int(min_hold))
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
    return {"signal_count": len(rows), "buy_days": len({row.get("buy_date") for row in rows if row.get("buy_date")})}


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
        "6",
        "--holding-days",
        "6",
        "--max-holding-days",
        "6",
        "--target-position-pct",
        str(float(cfg["target_cap"])),
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
            "signal_file": str(signal_file),
            "log_file": str(log_file),
            "annual": indicator.get("pnl_ratio_annual", indicator.get("annual_return")),
            "sharpe": indicator.get("sharp_ratio", indicator.get("sharpe_ratio")),
            "max_drawdown": indicator.get("max_drawdown"),
            "win_ratio": indicator.get("win_ratio"),
            "open_count": indicator.get("open_count"),
            "close_count": indicator.get("close_count"),
            "config_json": json.dumps({key: value for key, value in cfg.items() if key != "env"}, ensure_ascii=False, sort_keys=True),
            "env_json": json.dumps(cfg["env"], ensure_ascii=False, sort_keys=True),
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
