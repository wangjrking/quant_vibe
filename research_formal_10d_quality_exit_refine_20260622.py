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
    / "formal_10d_quality_exit_refine_20260622"
)
SIGNALS = {
    "turn1_b118": DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_10d_current_best_quality_boost_20260622"
    / "signals"
    / "turn1_b118_same.csv",
    "mv8w_b130": DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_10d_current_best_quality_boost_20260622"
    / "signals"
    / "mv8w_b130_same.csv",
    "avgmv_low75_mid108": DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_10d_day_state_mv_gap_scale_20260622"
    / "signals"
    / "avgmv_low75_mid108.csv",
}
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


def _env(profile: str = "mid", extra: dict[str, str] | None = None) -> dict[str, str]:
    out = dict(base_grid.BASE_ENV)
    out.update(base_grid.EQDD_PROFILES[profile])
    if extra:
        out.update(extra)
    return out


def _variant(
    name: str,
    *,
    signal_name: str,
    env_profile: str = "mid",
    extra_env: dict[str, str] | None = None,
    max_positions: int = 7,
    holding_days: int = 6,
    max_holding_days: int = 6,
    target_position_pct: float = 0.42,
) -> dict:
    return {
        "name": name,
        "signal_name": signal_name,
        "max_positions": max_positions,
        "holding_days": holding_days,
        "max_holding_days": max_holding_days,
        "target_position_pct": target_position_pct,
        "env_profile": env_profile,
        "env": _env(env_profile, extra_env),
    }


EXIT_VARIANTS = [
    ("repro", {}),
    ("score98_mh2", {"GM_SCORE_EXIT_ENTRY_RATIO": "0.98", "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "2"}),
    ("score99_mh2", {"GM_SCORE_EXIT_ENTRY_RATIO": "0.99", "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "2"}),
    ("score102_mh3", {"GM_SCORE_EXIT_ENTRY_RATIO": "1.02", "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "3"}),
    ("daydrop90_mh2", {"GM_SCORE_STOP_LOSS_DAY_DROP_RATIO": "0.90", "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "2"}),
    ("rank30_mh2", {"GM_SCORE_EXIT_RANK": "0.30", "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "2"}),
    ("sl06", {"GM_STOP_LOSS_PCT": "0.06"}),
    ("tp18", {"GM_TAKE_PROFIT_PCT": "0.18"}),
    ("sells2_score99", {"GM_MAX_DAILY_SELLS": "2", "GM_SCORE_EXIT_ENTRY_RATIO": "0.99", "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "2"}),
]

VARIANTS: list[dict] = []
for signal_name in ["turn1_b118", "mv8w_b130", "avgmv_low75_mid108"]:
    for exit_name, extra in EXIT_VARIANTS:
        VARIANTS.append(_variant(f"{signal_name}_{exit_name}", signal_name=signal_name, extra_env=extra))
    VARIANTS.append(_variant(f"{signal_name}_strict", signal_name=signal_name, env_profile="strict"))
    VARIANTS.append(_variant(f"{signal_name}_base", signal_name=signal_name, env_profile="base"))
    VARIANTS.append(
        _variant(
            f"{signal_name}_dd_hard",
            signal_name=signal_name,
            extra_env={
                "GM_EQUITY_DD_SOFT_TRIGGER": "0.07",
                "GM_EQUITY_DD_HARD_TRIGGER": "0.14",
                "GM_EQUITY_DD_SOFT_SCALE": "0.78",
                "GM_EQUITY_DD_HARD_SCALE": "0.50",
            },
        )
    )


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


def _signal_stats(signal_file: Path) -> dict:
    day_sums: dict[str, float] = {}
    count = 0
    with signal_file.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            count += 1
            day = str(row.get("buy_date") or row.get("signal_date"))
            day_sums[day] = day_sums.get(day, 0.0) + (_to_float(row.get("target_pct"), 0.0) or 0.0)
    return {
        "signal_count": count,
        "buy_days": len(day_sums),
        "avg_day_target_sum": sum(day_sums.values()) / len(day_sums) if day_sums else None,
        "min_day_target_sum": min(day_sums.values()) if day_sums else None,
        "max_day_target_sum": max(day_sums.values()) if day_sums else None,
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
    (REPORT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for index, cfg in enumerate(VARIANTS, start=1):
        signal_file = SIGNALS[str(cfg["signal_name"])]
        log_file = REPORT_DIR / "logs" / f"{cfg['name']}.log"
        signal_stats = _signal_stats(signal_file)
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
        print(f"[{index}/{len(VARIANTS)}] {cfg['name']} annual={row.get('annual')} sharpe={row.get('sharpe')} avg={row.get('avg_invested_pct')}")
    _write_rows(REPORT_DIR / "summary_by_objective.csv", sorted(results, key=_objective, reverse=True))
    _write_rows(REPORT_DIR / "summary_by_sharpe.csv", sorted(results, key=lambda row: _metric(row, "sharpe"), reverse=True))
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
