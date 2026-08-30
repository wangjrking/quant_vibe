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

import research_formal_10d_bucket_risk_reweight_20260622 as risk_base


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_10d_nav_vol_target_scale_v2_20260622"
)
SOURCE_SIGNAL = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_10d_current_best_exit_grid_20260622"
    / "signals"
    / "repro_current_best.csv"
)
BASELINE_LOG = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_10d_current_best_exit_grid_20260622"
    / "logs"
    / "repro_current_best.log"
)
SCORE_DB = risk_base.SCORE_DB
SCORE_TABLE = risk_base.SCORE_TABLE
MARKET_DB = risk_base.MARKET_DB
STRATEGY_DIR = risk_base.STRATEGY_DIR
JUEJIN_PYTHON = risk_base.JUEJIN_PYTHON

BASE_ENV = dict(risk_base.BASE_ENV)
BASE_ENV.update(
    {
        "GM_OPEN_DAILY_SCORE_EXIT": "1",
        "GM_SCORE_EXIT_ENTRY_RATIO": "1.0",
        "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "4",
        "GM_MAX_DAILY_SELLS": "1",
        "GM_EQUITY_DD_RISK_MODE": "1",
        "GM_EQUITY_DD_SOFT_TRIGGER": "0.09",
        "GM_EQUITY_DD_HARD_TRIGGER": "0.17",
        "GM_EQUITY_DD_RECOVER_TRIGGER": "0.05",
        "GM_EQUITY_DD_SOFT_SCALE": "0.82",
        "GM_EQUITY_DD_HARD_SCALE": "0.58",
        "GM_STOP_LOSS_PCT": "0.08",
        "GM_TAKE_PROFIT_PCT": "none",
        "GM_LIGHT_STOP_LOSS_PCT": "none",
        "GM_LOG_EXPOSURE": "1",
        "GM_SKIP_OPEN_LIMIT_UP_BUY": "1",
        "GM_SYNC_POSITIONS": "1",
        "GM_CASH_BUFFER": "0.99",
        "GM_VERBOSE_TRADES": "0",
        "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.02",
    }
)


def _variant(name: str, *, mode: str, window: int = 20, target_vol: float = 0.018, floor: float = 0.70, cap: float = 1.20, extra_env: dict[str, str] | None = None) -> dict:
    env = dict(BASE_ENV)
    if extra_env:
        env.update(extra_env)
    return {
        "name": name,
        "mode": mode,
        "window": window,
        "target_vol": target_vol,
        "floor": floor,
        "cap": cap,
        "env": env,
    }


VARIANTS = [
    _variant("repro", mode="none"),
    _variant("vol10_t18_f70_c120", mode="vol", window=10, target_vol=0.018, floor=0.70, cap=1.20),
    _variant("vol15_t18_f70_c120", mode="vol", window=15, target_vol=0.018, floor=0.70, cap=1.20),
    _variant("vol20_t18_f70_c120", mode="vol", window=20, target_vol=0.018, floor=0.70, cap=1.20),
    _variant("vol20_t16_f65_c125", mode="vol", window=20, target_vol=0.016, floor=0.65, cap=1.25),
    _variant("vol20_t20_f75_c115", mode="vol", window=20, target_vol=0.020, floor=0.75, cap=1.15),
    _variant("mom10_vol20_soft", mode="mom_vol", window=20, target_vol=0.018, floor=0.72, cap=1.18),
    _variant("mom5_vol15_soft", mode="mom_vol5", window=15, target_vol=0.018, floor=0.72, cap=1.18),
    _variant("vol20_t18_no_eqdd", mode="vol", window=20, target_vol=0.018, floor=0.70, cap=1.20, extra_env={"GM_EQUITY_DD_RISK_MODE": "0"}),
]


def _to_float(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return value


def _read_signal() -> tuple[list[dict], list[str]]:
    with SOURCE_SIGNAL.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader), list(reader.fieldnames or [])


def _nav_series() -> list[dict]:
    pattern = re.compile(r"EXPOSURE (\d{8}) post_buy .* nav=([0-9.]+)")
    rows: list[dict] = []
    for match in pattern.finditer(BASELINE_LOG.read_text(encoding="utf-8", errors="ignore")):
        rows.append({"date": match.group(1), "nav": float(match.group(2))})
    rows.sort(key=lambda row: row["date"])
    previous_nav = None
    for row in rows:
        row["ret"] = 0.0 if previous_nav in (None, 0) else row["nav"] / previous_nav - 1.0
        previous_nav = row["nav"]
    return rows


def _std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _rolling_scales(cfg: dict) -> dict[str, float]:
    if cfg["mode"] == "none":
        return {}
    rows = _nav_series()
    scales: dict[str, float] = {}
    returns: list[float] = []
    for row in rows:
        date = row["date"]
        window_returns = returns[-int(cfg["window"]) :]
        scale = 1.0
        vol = _std(window_returns)
        if vol and vol > 0:
            scale = float(cfg["target_vol"]) / vol
        if cfg["mode"] in {"mom_vol", "mom_vol5"} and window_returns:
            lookback = 5 if cfg["mode"] == "mom_vol5" else 10
            recent = sum(returns[-lookback:])
            if recent < -0.035:
                scale *= 0.82
            elif recent > 0.035:
                scale *= 1.08
        scale = min(max(scale, float(cfg["floor"])), float(cfg["cap"]))
        scales[date] = scale
        returns.append(float(row["ret"]))
    return scales


def _write_signal(cfg: dict, signal_file: Path) -> dict:
    rows, fields = _read_signal()
    for field in ["nav_vol_scale", "nav_scale_mode"]:
        if field not in fields:
            fields.append(field)
    scales = _rolling_scales(cfg)
    day_sums: dict[str, float] = {}
    for row in rows:
        buy_date = str(row.get("buy_date") or "")
        scale = scales.get(buy_date, 1.0)
        base_target = _to_float(row.get("target_pct"), 0.0) or 0.0
        row["target_pct"] = f"{min(base_target * scale, 0.42):.5f}"
        row["nav_vol_scale"] = f"{scale:.6f}"
        row["nav_scale_mode"] = str(cfg["mode"])
        day_sums[buy_date] = day_sums.get(buy_date, 0.0) + (_to_float(row.get("target_pct"), 0.0) or 0.0)
    signal_file.parent.mkdir(parents=True, exist_ok=True)
    with signal_file.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    scale_values = list(scales.values()) or [1.0]
    return {
        "signal_count": len(rows),
        "buy_days": len(day_sums),
        "avg_day_target_sum": sum(day_sums.values()) / len(day_sums) if day_sums else None,
        "min_day_target_sum": min(day_sums.values()) if day_sums else None,
        "max_day_target_sum": max(day_sums.values()) if day_sums else None,
        "avg_nav_vol_scale": sum(scale_values) / len(scale_values),
        "min_nav_vol_scale": min(scale_values),
        "max_nav_vol_scale": max(scale_values),
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
        "7",
        "--holding-days",
        "6",
        "--max-holding-days",
        "6",
        "--target-position-pct",
        "0.42",
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
        print(f"[{index}/{len(VARIANTS)}] {cfg['name']} annual={row.get('annual')} sharpe={row.get('sharpe')} avg={row.get('avg_invested_pct')}", flush=True)
    _write_rows(REPORT_DIR / "summary_by_objective.csv", sorted(results, key=_objective, reverse=True))
    _write_rows(REPORT_DIR / "summary_by_sharpe.csv", sorted(results, key=lambda row: _metric(row, "sharpe"), reverse=True))
    _write_rows(
        REPORT_DIR / "summary_target_hits.csv",
        [
            row
            for row in results
            if _metric(row, "annual") >= 3.0
            and _metric(row, "sharpe") >= 4.0
            and _metric(row, "avg_invested_pct") >= 0.80
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
