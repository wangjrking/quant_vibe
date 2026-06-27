from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import os
import re
import shutil
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
    / "formal_10d_current_best_risk_control_grid_20260622"
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


def _variant(name: str, extra_env: dict[str, str] | None = None, *, max_positions: int = 7, holding_days: int = 6) -> dict:
    env = dict(BASE_ENV)
    if extra_env:
        env.update(extra_env)
    return {
        "name": name,
        "env": env,
        "max_positions": max_positions,
        "holding_days": holding_days,
        "cap": 0.42,
    }


VARIANTS = [
    _variant("repro"),
    _variant("dd_resize_mid", {"GM_EQUITY_DD_RESIZE_EXISTING": "1"}),
    _variant("dd_resize_soft", {
        "GM_EQUITY_DD_RESIZE_EXISTING": "1",
        "GM_EQUITY_DD_SOFT_TRIGGER": "0.11",
        "GM_EQUITY_DD_HARD_TRIGGER": "0.20",
        "GM_EQUITY_DD_SOFT_SCALE": "0.90",
        "GM_EQUITY_DD_HARD_SCALE": "0.70",
    }),
    _variant("dd_resize_strict", {
        "GM_EQUITY_DD_RESIZE_EXISTING": "1",
        "GM_EQUITY_DD_SOFT_TRIGGER": "0.08",
        "GM_EQUITY_DD_HARD_TRIGGER": "0.15",
        "GM_EQUITY_DD_SOFT_SCALE": "0.78",
        "GM_EQUITY_DD_HARD_SCALE": "0.50",
    }),
    _variant("intraday_sl05_half", {
        "GM_INTRADAY_RISK_MODE": "1",
        "GM_STOP_LOSS_PCT": "0.05",
        "GM_TAKE_PROFIT_PCT": "none",
        "GM_INTRADAY_RISK_SELL_FRACTION": "0.50",
    }),
    _variant("intraday_sl06_half", {
        "GM_INTRADAY_RISK_MODE": "1",
        "GM_STOP_LOSS_PCT": "0.06",
        "GM_TAKE_PROFIT_PCT": "none",
        "GM_INTRADAY_RISK_SELL_FRACTION": "0.50",
    }),
    _variant("intraday_sl05_full", {
        "GM_INTRADAY_RISK_MODE": "1",
        "GM_STOP_LOSS_PCT": "0.05",
        "GM_TAKE_PROFIT_PCT": "none",
        "GM_INTRADAY_RISK_SELL_FRACTION": "1.0",
    }),
    _variant("intraday_sl06_full", {
        "GM_INTRADAY_RISK_MODE": "1",
        "GM_STOP_LOSS_PCT": "0.06",
        "GM_TAKE_PROFIT_PCT": "none",
        "GM_INTRADAY_RISK_SELL_FRACTION": "1.0",
    }),
    _variant("intraday_sl05_replace", {
        "GM_INTRADAY_RISK_MODE": "1",
        "GM_STOP_LOSS_PCT": "0.05",
        "GM_TAKE_PROFIT_PCT": "none",
        "GM_INTRADAY_RISK_SELL_FRACTION": "1.0",
        "GM_INTRADAY_REPLACE_BUY": "1",
        "GM_INTRADAY_REPLACE_MAX_BUYS": "2",
    }),
    _variant("intraday_sl06_replace", {
        "GM_INTRADAY_RISK_MODE": "1",
        "GM_STOP_LOSS_PCT": "0.06",
        "GM_TAKE_PROFIT_PCT": "none",
        "GM_INTRADAY_RISK_SELL_FRACTION": "1.0",
        "GM_INTRADAY_REPLACE_BUY": "1",
        "GM_INTRADAY_REPLACE_MAX_BUYS": "2",
    }),
    _variant("intraday_sl06_tp18_half", {
        "GM_INTRADAY_RISK_MODE": "1",
        "GM_STOP_LOSS_PCT": "0.06",
        "GM_TAKE_PROFIT_PCT": "0.18",
        "GM_INTRADAY_RISK_SELL_FRACTION": "0.50",
    }),
    _variant("defer_losses_to_8", {
        "GM_DEFER_EXITS_WITHOUT_BUY_SIGNAL": "1",
        "GM_NO_SIGNAL_MAX_HOLDING_DAYS": "8",
        "GM_DEFER_EXIT_MAX_POSITION_RETURN": "0.00",
    }),
    _variant("defer_small_losses_to_8", {
        "GM_DEFER_EXITS_WITHOUT_BUY_SIGNAL": "1",
        "GM_NO_SIGNAL_MAX_HOLDING_DAYS": "8",
        "GM_DEFER_EXIT_MIN_POSITION_RETURN": "-0.05",
        "GM_DEFER_EXIT_MAX_POSITION_RETURN": "0.00",
    }),
    _variant("index_buy_scale_soft", {
        "GM_INDEX_RISK_EXIT_MODE": "1",
        "GM_INDEX_RISK_CC_THRESHOLD": "-0.055",
        "GM_INDEX_RISK_INTRADAY_THRESHOLD": "-0.055",
        "GM_INDEX_RISK_BUY_SCALE": "0.75",
    }),
    _variant("breadth_buy_scale_soft", {
        "GM_BREADTH_RISK_EXIT_MODE": "1",
        "GM_BREADTH_RISK_UP_RATIO_THRESHOLD": "0.04",
        "GM_BREADTH_RISK_AVG_PCT_THRESHOLD": "-4.5",
        "GM_BREADTH_RISK_BUY_SCALE": "0.75",
    }),
    _variant("market_buy_scale_soft", {
        "GM_INDEX_RISK_EXIT_MODE": "1",
        "GM_INDEX_RISK_CC_THRESHOLD": "-0.055",
        "GM_INDEX_RISK_INTRADAY_THRESHOLD": "-0.055",
        "GM_INDEX_RISK_BUY_SCALE": "0.80",
        "GM_BREADTH_RISK_EXIT_MODE": "1",
        "GM_BREADTH_RISK_UP_RATIO_THRESHOLD": "0.04",
        "GM_BREADTH_RISK_AVG_PCT_THRESHOLD": "-4.5",
        "GM_BREADTH_RISK_BUY_SCALE": "0.80",
    }),
    _variant("dd_resize_intraday_half", {
        "GM_EQUITY_DD_RESIZE_EXISTING": "1",
        "GM_INTRADAY_RISK_MODE": "1",
        "GM_STOP_LOSS_PCT": "0.06",
        "GM_TAKE_PROFIT_PCT": "none",
        "GM_INTRADAY_RISK_SELL_FRACTION": "0.50",
    }),
    _variant("dd_resize_market_soft", {
        "GM_EQUITY_DD_RESIZE_EXISTING": "1",
        "GM_INDEX_RISK_EXIT_MODE": "1",
        "GM_INDEX_RISK_CC_THRESHOLD": "-0.055",
        "GM_INDEX_RISK_INTRADAY_THRESHOLD": "-0.055",
        "GM_INDEX_RISK_BUY_SCALE": "0.85",
        "GM_BREADTH_RISK_EXIT_MODE": "1",
        "GM_BREADTH_RISK_UP_RATIO_THRESHOLD": "0.04",
        "GM_BREADTH_RISK_AVG_PCT_THRESHOLD": "-4.5",
        "GM_BREADTH_RISK_BUY_SCALE": "0.85",
    }),
]


def _to_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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


def _signal_copy(name: str) -> Path:
    out = REPORT_DIR / "signals" / f"{name}.csv"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE_SIGNAL, out)
    return out


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
        str(cfg["holding_days"]),
        "--target-position-pct",
        str(cfg["cap"]),
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
        signal_file = _signal_copy(str(cfg["name"]))
        log_file = REPORT_DIR / "logs" / f"{cfg['name']}.log"
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
        row.update(_exposure_stats(log_file))
        results.append(row)
        _write_rows(REPORT_DIR / "summary.csv", results)
        print(f"[{index}/{len(VARIANTS)}] {cfg['name']} annual={row.get('annual')} sharpe={row.get('sharpe')} avg={row.get('avg_invested_pct')}")
    _write_rows(REPORT_DIR / "summary_by_objective.csv", sorted(results, key=_objective, reverse=True))
    _write_rows(REPORT_DIR / "summary_by_sharpe.csv", sorted(results, key=lambda row: _metric(row, "sharpe"), reverse=True))
    _write_rows(REPORT_DIR / "summary_avg80_by_sharpe.csv", [row for row in sorted(results, key=lambda item: _metric(item, "sharpe"), reverse=True) if _metric(row, "avg_invested_pct") >= 0.80])
    _write_rows(REPORT_DIR / "summary_target_hits.csv", [row for row in results if _metric(row, "annual") >= 3.0 and _metric(row, "sharpe") >= 4.0 and _metric(row, "avg_invested_pct") >= 0.80])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
