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
SOURCE_DIR = (
    DATA_DIR
    / "reports"
    / "strategy_agent_goal_high_annual_20260623"
    / "current_formal_pool_filter_tune"
)
REPORT_DIR = (
    DATA_DIR
    / "reports"
    / "strategy_agent_goal_high_annual_20260623"
    / "current_formal_top5_noexit_execution_tune"
)
SOURCE_SIGNAL = SOURCE_DIR / "signals" / "all_w80_top5_h5_noexit.csv"
SCORE_DB = SOURCE_DIR / "pool_filter_scores.db"
SCORE_TABLE = "pool_filter_all_w80_top5_h5_noexit"
MARKET_DB = DATA_DIR / "STOCK_DAILY_DATA.db"
STRATEGY_DIR = Path(r"D:\work\dfcf\juejin\strategy\ce327750-634e-11f1-b8d7-10ffe0295517")
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")
BACKTEST_START = "2024-06-05 09:00:00"
BACKTEST_END = "2026-06-23 15:30:00"


BASE_ENV = {
    "GM_LOG_EXPOSURE": "1",
    "GM_SKIP_OPEN_LIMIT_UP_BUY": "1",
    "GM_SYNC_POSITIONS": "1",
    "GM_CASH_BUFFER": "0.99",
    "GM_VERBOSE_TRADES": "0",
    "GM_TAKE_PROFIT_PCT": "none",
    "GM_LIGHT_STOP_LOSS_PCT": "none",
    "GM_SCORE_EXIT_RANK": "none",
    "GM_EQUITY_DD_RISK_MODE": "0",
}


def _variant(
    name: str,
    *,
    scale: float,
    cap: float,
    holding_days: int,
    score_exit: bool,
    min_score_hold: int | None = None,
    stop_loss: str = "0.08",
    cash_buffer: str = "0.99",
    dd_risk: bool = False,
) -> dict:
    env = dict(BASE_ENV)
    env["GM_CASH_BUFFER"] = cash_buffer
    env["GM_OPEN_DAILY_SCORE_EXIT"] = "1" if score_exit else "0"
    env["GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT"] = str(min_score_hold if min_score_hold is not None else holding_days)
    env["GM_MAX_DAILY_SELLS"] = "5"
    env["GM_SCORE_EXIT_ENTRY_RATIO"] = "1.0"
    env["GM_SCORE_CONTINUE_ENTRY_RATIO"] = "1.02"
    env["GM_STOP_LOSS_PCT"] = stop_loss
    if dd_risk:
        env.update(
            {
                "GM_EQUITY_DD_RISK_MODE": "1",
                "GM_EQUITY_DD_SOFT_TRIGGER": "0.12",
                "GM_EQUITY_DD_HARD_TRIGGER": "0.22",
                "GM_EQUITY_DD_RECOVER_TRIGGER": "0.05",
                "GM_EQUITY_DD_SOFT_SCALE": "0.90",
                "GM_EQUITY_DD_HARD_SCALE": "0.70",
            }
        )
    return {
        "name": name,
        "scale": scale,
        "cap": cap,
        "holding_days": holding_days,
        "score_exit": score_exit,
        "stop_loss": stop_loss,
        "cash_buffer": cash_buffer,
        "dd_risk": dd_risk,
        "env": env,
    }


VARIANTS = [
    _variant("base_recheck", scale=1.0, cap=0.26, holding_days=5, score_exit=False),
    _variant("scale110_cap30_h5_noexit", scale=1.10, cap=0.30, holding_days=5, score_exit=False),
    _variant("scale125_cap34_h5_noexit", scale=1.25, cap=0.34, holding_days=5, score_exit=False),
    _variant("scale150_cap40_h5_noexit", scale=1.50, cap=0.40, holding_days=5, score_exit=False),
    _variant("scale175_cap45_h5_noexit", scale=1.75, cap=0.45, holding_days=5, score_exit=False),
    _variant("scale125_cap34_h4_noexit", scale=1.25, cap=0.34, holding_days=4, score_exit=False),
    _variant("scale125_cap34_h6_noexit", scale=1.25, cap=0.34, holding_days=6, score_exit=False),
    _variant("scale125_cap34_h7_noexit", scale=1.25, cap=0.34, holding_days=7, score_exit=False),
    _variant("scale125_cap34_h8_noexit", scale=1.25, cap=0.34, holding_days=8, score_exit=False),
    _variant("scale125_cap34_h5_exit1", scale=1.25, cap=0.34, holding_days=5, score_exit=True, min_score_hold=1),
    _variant("scale125_cap34_h5_exit2", scale=1.25, cap=0.34, holding_days=5, score_exit=True, min_score_hold=2),
    _variant("scale125_cap34_h5_exit3", scale=1.25, cap=0.34, holding_days=5, score_exit=True, min_score_hold=3),
    _variant("scale125_cap34_h5_nostop", scale=1.25, cap=0.34, holding_days=5, score_exit=False, stop_loss="none"),
    _variant("scale125_cap34_h5_stop12", scale=1.25, cap=0.34, holding_days=5, score_exit=False, stop_loss="0.12"),
    _variant("scale125_cap34_h5_cash995", scale=1.25, cap=0.34, holding_days=5, score_exit=False, cash_buffer="0.995"),
    _variant("scale125_cap34_h5_ddrisk", scale=1.25, cap=0.34, holding_days=5, score_exit=False, dd_risk=True),
]


def _to_float(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return value


def _load_rows(path: Path) -> tuple[list[dict], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader), list(reader.fieldnames or [])


def _write_signal(cfg: dict, output: Path) -> dict:
    rows, fields = _load_rows(SOURCE_SIGNAL)
    for field in ["raw_target_pct"]:
        if field not in fields:
            fields.append(field)
    day_sums: dict[str, float] = {}
    for row in rows:
        raw = _to_float(row.get("target_pct"), 0.0) or 0.0
        target = min(raw * float(cfg["scale"]), float(cfg["cap"]))
        row["raw_target_pct"] = f"{raw:.5f}"
        row["target_pct"] = f"{target:.5f}"
        row["holding_days"] = str(int(cfg["holding_days"]))
        row["max_holding_days"] = str(int(cfg["holding_days"]))
        day = str(row.get("buy_date") or "")
        day_sums[day] = day_sums.get(day, 0.0) + target
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in fields} for row in rows])
    return {
        "signal_count": len(rows),
        "buy_days": len(day_sums),
        "avg_day_target_sum": sum(day_sums.values()) / len(day_sums) if day_sums else None,
        "min_day_target_sum": min(day_sums.values()) if day_sums else None,
        "max_day_target_sum": max(day_sums.values()) if day_sums else None,
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
        str(int(cfg["holding_days"])),
        "--max-holding-days",
        str(int(cfg["holding_days"])),
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
    if not SOURCE_SIGNAL.exists():
        raise SystemExit(f"source signal not found: {SOURCE_SIGNAL}")
    if not STRATEGY_DIR.joinpath("main.py").exists():
        raise SystemExit(f"juejin strategy main.py not found: {STRATEGY_DIR}")
    results = []
    REPORT_DIR.joinpath("signals").mkdir(parents=True, exist_ok=True)
    REPORT_DIR.joinpath("logs").mkdir(parents=True, exist_ok=True)
    for index, cfg in enumerate(VARIANTS, start=1):
        signal_file = REPORT_DIR / "signals" / f"{cfg['name']}.csv"
        log_file = REPORT_DIR / "logs" / f"{cfg['name']}.log"
        stats = _write_signal(cfg, signal_file)
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
            "env_json": json.dumps(cfg["env"], ensure_ascii=False, sort_keys=True),
            "signal_file": str(signal_file),
            "log_file": str(log_file),
            **stats,
            **_exposure_stats(log_file),
        }
        results.append(row)
        _write_rows(REPORT_DIR / "summary.csv", results)
        print(
            f"[{index}/{len(VARIANTS)}] {cfg['name']} annual={row.get('annual')} "
            f"sharpe={row.get('sharpe')} maxdd={row.get('max_drawdown')} avg={row.get('avg_invested_pct')}",
            flush=True,
        )
    _write_rows(REPORT_DIR / "summary_by_annual.csv", sorted(results, key=lambda row: _metric(row, "annual"), reverse=True))
    _write_rows(REPORT_DIR / "summary_target_hits.csv", [row for row in results if _metric(row, "annual") >= 5.0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
