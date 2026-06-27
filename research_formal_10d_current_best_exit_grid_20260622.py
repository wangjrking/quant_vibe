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

import research_formal_10d_bucket_cap_grid_20260622 as base_grid


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_10d_current_best_exit_grid_20260622"
)
SOURCE_SIGNAL = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_10d_bucket_risk_reweight_v2_20260622"
    / "signals"
    / "turn6_down85.csv"
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


BASE_ENV = dict(base_grid.BASE_ENV)
BASE_ENV.update(base_grid.EQDD_PROFILES["mid"])


def _variant(
    name: str,
    *,
    holding_days: int = 6,
    max_holding_days: int = 6,
    max_positions: int = 7,
    target_position_pct: float = 0.42,
    per_row_mode: str = "none",
    extra_env: dict[str, str] | None = None,
) -> dict:
    env = dict(BASE_ENV)
    if extra_env:
        env.update(extra_env)
    return {
        "name": name,
        "holding_days": holding_days,
        "max_holding_days": max_holding_days,
        "max_positions": max_positions,
        "target_position_pct": target_position_pct,
        "per_row_mode": per_row_mode,
        "env": env,
    }


VARIANTS = [
    _variant("repro_current_best"),
    _variant("sl06", extra_env={"GM_STOP_LOSS_PCT": "0.06"}),
    _variant("sl07", extra_env={"GM_STOP_LOSS_PCT": "0.07"}),
    _variant("sl09", extra_env={"GM_STOP_LOSS_PCT": "0.09"}),
    _variant("tp14", extra_env={"GM_TAKE_PROFIT_PCT": "0.14"}),
    _variant("tp18", extra_env={"GM_TAKE_PROFIT_PCT": "0.18"}),
    _variant("tp22", extra_env={"GM_TAKE_PROFIT_PCT": "0.22"}),
    _variant("sl06_tp18", extra_env={"GM_STOP_LOSS_PCT": "0.06", "GM_TAKE_PROFIT_PCT": "0.18"}),
    _variant("sl07_tp18", extra_env={"GM_STOP_LOSS_PCT": "0.07", "GM_TAKE_PROFIT_PCT": "0.18"}),
    _variant("light04_mh2", extra_env={"GM_LIGHT_STOP_LOSS_PCT": "0.04", "GM_MIN_HOLDING_DAYS_BEFORE_LIGHT_STOP": "2"}),
    _variant("light05_mh2", extra_env={"GM_LIGHT_STOP_LOSS_PCT": "0.05", "GM_MIN_HOLDING_DAYS_BEFORE_LIGHT_STOP": "2"}),
    _variant("score97_mh2", extra_env={"GM_SCORE_EXIT_ENTRY_RATIO": "0.97", "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "2"}),
    _variant("score98_mh2", extra_env={"GM_SCORE_EXIT_ENTRY_RATIO": "0.98", "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "2"}),
    _variant("score99_mh2", extra_env={"GM_SCORE_EXIT_ENTRY_RATIO": "0.99", "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "2"}),
    _variant("score102_mh2", extra_env={"GM_SCORE_EXIT_ENTRY_RATIO": "1.02", "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "2"}),
    _variant("score98_mh3", extra_env={"GM_SCORE_EXIT_ENTRY_RATIO": "0.98", "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "3"}),
    _variant("score102_mh3", extra_env={"GM_SCORE_EXIT_ENTRY_RATIO": "1.02", "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "3"}),
    _variant("h5_mh5", holding_days=5, max_holding_days=5),
    _variant("h6_mh7_continue103", max_holding_days=7, extra_env={"GM_SCORE_CONTINUE_ENTRY_RATIO": "1.03"}),
    _variant("h6_mh8_continue105", max_holding_days=8, extra_env={"GM_SCORE_CONTINUE_ENTRY_RATIO": "1.05"}),
    _variant("max_sells2_score99", extra_env={"GM_MAX_DAILY_SELLS": "2", "GM_SCORE_EXIT_ENTRY_RATIO": "0.99", "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "2"}),
    _variant("max_sells2_score102", extra_env={"GM_MAX_DAILY_SELLS": "2", "GM_SCORE_EXIT_ENTRY_RATIO": "1.02", "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "2"}),
    _variant("row_risk_hold_short", per_row_mode="risk_hold_short"),
    _variant("row_quality_hold_long", per_row_mode="quality_hold_long"),
    _variant("row_quality_exit_soft", per_row_mode="quality_exit_soft"),
    _variant("row_quality_exit_tight", per_row_mode="quality_exit_tight"),
    _variant("score_daydrop90_mh2", extra_env={"GM_SCORE_STOP_LOSS_DAY_DROP_RATIO": "0.90", "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "2"}),
    _variant("score_daydrop85_mh2", extra_env={"GM_SCORE_STOP_LOSS_DAY_DROP_RATIO": "0.85", "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "2"}),
    _variant("score_daydrop80_mh2", extra_env={"GM_SCORE_STOP_LOSS_DAY_DROP_RATIO": "0.80", "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "2"}),
    _variant("score_daydrop90_mh3", extra_env={"GM_SCORE_STOP_LOSS_DAY_DROP_RATIO": "0.90", "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "3"}),
    _variant("score_rank20_mh2", extra_env={"GM_SCORE_EXIT_RANK": "0.20", "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "2"}),
    _variant("score_rank30_mh2", extra_env={"GM_SCORE_EXIT_RANK": "0.30", "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "2"}),
    _variant("score_stop_ratio85_mh2", extra_env={"GM_SCORE_STOP_LOSS_RATIO": "0.85", "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "2"}),
    _variant("score_stop_ratio75_mh2", extra_env={"GM_SCORE_STOP_LOSS_RATIO": "0.75", "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "2"}),
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


def _risk_role(row: dict) -> str:
    turnover = _to_float(row.get("turnover_rate"))
    amount = _to_float(row.get("amount"))
    total_mv = _to_float(row.get("total_mv"))
    gap = _to_float(row.get("pred_gap"))
    pred10 = _to_float(row.get("pred_10d"))
    if (
        (turnover is not None and turnover >= 6.0)
        or (gap is not None and gap >= 0.07)
        or (pred10 is not None and pred10 >= 0.16)
        or (total_mv is not None and total_mv >= 180000)
    ):
        return "risk"
    if (
        (turnover is not None and turnover < 2.0)
        or (amount is not None and amount < 10000)
        or (total_mv is not None and 50000 <= total_mv < 90000)
    ):
        return "quality"
    return "base"


def _write_signal(cfg: dict, signal_file: Path) -> dict:
    rows, fields = _read_rows()
    for field in [
        "max_holding_days",
        "risk_role",
        "signal_score_exit_entry_ratio",
        "signal_min_holding_days_before_score_exit",
        "signal_score_continue_entry_ratio",
    ]:
        if field not in fields:
            fields.append(field)
    role_counts = {"quality": 0, "base": 0, "risk": 0}
    mode = str(cfg["per_row_mode"])
    for row in rows:
        role = _risk_role(row)
        role_counts[role] += 1
        row["risk_role"] = role
        row["holding_days"] = str(int(cfg["holding_days"]))
        row["max_holding_days"] = str(int(cfg["max_holding_days"]))
        row["signal_score_exit_entry_ratio"] = ""
        row["signal_min_holding_days_before_score_exit"] = ""
        row["signal_score_continue_entry_ratio"] = ""
        if mode == "risk_hold_short" and role == "risk":
            row["holding_days"] = "4"
            row["max_holding_days"] = "5"
        elif mode == "quality_hold_long":
            if role == "quality":
                row["holding_days"] = "7"
                row["max_holding_days"] = "8"
                row["signal_score_continue_entry_ratio"] = "1.04"
            elif role == "risk":
                row["holding_days"] = "5"
                row["max_holding_days"] = "5"
        elif mode == "quality_exit_soft":
            if role == "risk":
                row["holding_days"] = "5"
                row["max_holding_days"] = "5"
                row["signal_score_exit_entry_ratio"] = "1.02"
                row["signal_min_holding_days_before_score_exit"] = "2"
            elif role == "quality":
                row["holding_days"] = "7"
                row["max_holding_days"] = "8"
                row["signal_score_exit_entry_ratio"] = "0.98"
                row["signal_min_holding_days_before_score_exit"] = "4"
                row["signal_score_continue_entry_ratio"] = "1.04"
        elif mode == "quality_exit_tight":
            if role == "risk":
                row["holding_days"] = "4"
                row["max_holding_days"] = "4"
                row["signal_score_exit_entry_ratio"] = "1.03"
                row["signal_min_holding_days_before_score_exit"] = "2"
            elif role == "quality":
                row["holding_days"] = "7"
                row["max_holding_days"] = "7"
                row["signal_score_exit_entry_ratio"] = "0.99"
                row["signal_min_holding_days_before_score_exit"] = "3"

    signal_file.parent.mkdir(parents=True, exist_ok=True)
    with signal_file.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in fields} for row in rows])
    day_sums: dict[str, float] = {}
    for row in rows:
        day = str(row.get("signal_date"))
        day_sums[day] = day_sums.get(day, 0.0) + (_to_float(row.get("target_pct"), 0.0) or 0.0)
    return {
        "signal_count": len(rows),
        "buy_days": len({row.get("buy_date") for row in rows if row.get("buy_date")}),
        "avg_day_target_sum": sum(day_sums.values()) / len(day_sums) if day_sums else None,
        "min_day_target_sum": min(day_sums.values()) if day_sums else None,
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
        str(JUEJIN_PYTHON),
        str(MAIN / "run_juejin_signal_backtest.py"),
        "--strategy-dir",
        str(STRATEGY_DIR),
        "--signal-file",
        str(signal_file),
        "--log-file",
        str(log_file),
        "--max-positions",
        str(cfg["max_positions"]),
        "--holding-days",
        str(cfg["holding_days"]),
        "--max-holding-days",
        str(cfg["max_holding_days"]),
        "--target-position-pct",
        str(cfg["target_position_pct"]),
        "--score-db",
        str(SCORE_DB),
        "--score-table",
        SCORE_TABLE,
        "--market-db",
        str(MARKET_DB),
        "--backtest-start",
        "2024-06-05 09:00:00",
        "--backtest-end",
        "2026-06-18 15:30:00",
        "--backtest-adjust",
        "none",
        "--backtest-initial-cash",
        "600000",
        "--backtest-slippage-ratio",
        "0.0015",
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
            "env_json": json.dumps(cfg["env"], ensure_ascii=False, sort_keys=True),
            "signal_file": str(signal_file),
            "log_file": str(log_file),
        }
        row.update(signal_stats)
        row.update(_exposure_stats(log_file))
        results.append(row)
        _write_rows(REPORT_DIR / "summary.csv", results)
        print(
            f"[{index}/{len(VARIANTS)}] {cfg['name']} "
            f"annual={row.get('annual')} sharpe={row.get('sharpe')} avg={row.get('avg_invested_pct')}"
        )
    _write_rows(REPORT_DIR / "summary_by_objective.csv", sorted(results, key=_objective, reverse=True))
    _write_rows(REPORT_DIR / "summary_by_sharpe.csv", sorted(results, key=lambda row: _metric(row, "sharpe"), reverse=True))
    _write_rows(
        REPORT_DIR / "summary_avg80_by_sharpe.csv",
        [
            row
            for row in sorted(results, key=lambda item: _metric(item, "sharpe"), reverse=True)
            if _metric(row, "avg_invested_pct") >= 0.80
        ],
    )
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
