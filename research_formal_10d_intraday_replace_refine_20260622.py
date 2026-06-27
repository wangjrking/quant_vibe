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

import research_formal_10d_current_best_risk_control_grid_20260622 as risk_grid


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_10d_intraday_replace_refine_20260622"
)
SOURCE_SIGNAL = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_10d_current_best_exit_grid_20260622"
    / "signals"
    / "repro_current_best.csv"
)
SCORE_DB = risk_grid.SCORE_DB
SCORE_TABLE = risk_grid.SCORE_TABLE
MARKET_DB = risk_grid.MARKET_DB
STRATEGY_DIR = risk_grid.STRATEGY_DIR
JUEJIN_PYTHON = risk_grid.JUEJIN_PYTHON

BASE_ENV = dict(risk_grid.BASE_ENV)
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
        "GM_LIGHT_STOP_LOSS_PCT": "none",
        "GM_LOG_EXPOSURE": "1",
        "GM_SKIP_OPEN_LIMIT_UP_BUY": "1",
        "GM_SYNC_POSITIONS": "1",
        "GM_CASH_BUFFER": "0.99",
        "GM_VERBOSE_TRADES": "0",
        "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.02",
        "GM_TAKE_PROFIT_PCT": "none",
    }
)


def _variant(name: str, extra_env: dict[str, str] | None = None, *, max_positions: int = 7, holding_days: int = 6, cap: float = 0.42) -> dict:
    env = dict(BASE_ENV)
    if extra_env:
        env.update(extra_env)
    return {
        "name": name,
        "env": env,
        "max_positions": max_positions,
        "holding_days": holding_days,
        "cap": cap,
    }


def _intraday_env(stop: str, fraction: str, replace_buys: str, extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {
        "GM_INTRADAY_RISK_MODE": "1",
        "GM_STOP_LOSS_PCT": stop,
        "GM_INTRADAY_RISK_SELL_FRACTION": fraction,
        "GM_INTRADAY_REPLACE_BUY": "1",
        "GM_INTRADAY_REPLACE_MAX_BUYS": replace_buys,
    }
    if extra:
        env.update(extra)
    return env


VARIANTS = [
    _variant("repro"),
    _variant("sl045_full_replace1", _intraday_env("0.045", "1.0", "1")),
    _variant("sl05_full_replace1", _intraday_env("0.05", "1.0", "1")),
    _variant("sl05_full_replace2", _intraday_env("0.05", "1.0", "2")),
    _variant("sl055_full_replace1", _intraday_env("0.055", "1.0", "1")),
    _variant("sl06_full_replace1", _intraday_env("0.06", "1.0", "1")),
    _variant("sl05_part75_replace1", _intraday_env("0.05", "0.75", "1")),
    _variant("sl05_part75_replace2", _intraday_env("0.05", "0.75", "2")),
    _variant("sl05_half_replace2", _intraday_env("0.05", "0.50", "2")),
    _variant("sl05_full_replace1_no_score_exit", _intraday_env("0.05", "1.0", "1", {"GM_OPEN_DAILY_SCORE_EXIT": "0"})),
    _variant("sl05_full_replace2_no_score_exit", _intraday_env("0.05", "1.0", "2", {"GM_OPEN_DAILY_SCORE_EXIT": "0"})),
    _variant(
        "sl05_full_replace1_defer_small_loss",
        _intraday_env(
            "0.05",
            "1.0",
            "1",
            {
                "GM_DEFER_EXITS_WITHOUT_BUY_SIGNAL": "1",
                "GM_NO_SIGNAL_MAX_HOLDING_DAYS": "8",
                "GM_DEFER_EXIT_MIN_POSITION_RETURN": "-0.05",
                "GM_DEFER_EXIT_MAX_POSITION_RETURN": "0.00",
            },
        ),
    ),
    _variant("sl05_full_replace1_soft_dd", _intraday_env("0.05", "1.0", "1", {"GM_EQUITY_DD_SOFT_TRIGGER": "0.11", "GM_EQUITY_DD_HARD_TRIGGER": "0.20", "GM_EQUITY_DD_SOFT_SCALE": "0.90", "GM_EQUITY_DD_HARD_SCALE": "0.70"})),
    _variant("sl05_full_replace1_resize_dd", _intraday_env("0.05", "1.0", "1", {"GM_EQUITY_DD_RESIZE_EXISTING": "1"})),
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
        print(f"[{index}/{len(VARIANTS)}] {cfg['name']} annual={row.get('annual')} sharpe={row.get('sharpe')} avg={row.get('avg_invested_pct')}", flush=True)
    _write_rows(REPORT_DIR / "summary_by_objective.csv", sorted(results, key=_objective, reverse=True))
    _write_rows(REPORT_DIR / "summary_by_sharpe.csv", sorted(results, key=lambda row: _metric(row, "sharpe"), reverse=True))
    _write_rows(
        REPORT_DIR / "summary_avg80_by_sharpe.csv",
        [row for row in sorted(results, key=lambda item: _metric(item, "sharpe"), reverse=True) if _metric(row, "avg_invested_pct") >= 0.80],
    )
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
