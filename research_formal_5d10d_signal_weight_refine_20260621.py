from __future__ import annotations

import ast
import csv
import datetime as datetime_module
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
    / "strategy_agent_model_application_20260621"
    / "formal_5d10d_pool_exec_cross_refine_20260621"
)
REPORT_DIR = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_5d10d_signal_weight_refine_20260621"
)
BASE_SIGNAL = SOURCE_DIR / "signals" / "base_h7_tp1092.csv"
SCORE_DB = SOURCE_DIR / "pool_exec_scores.db"
SCORE_TABLE = "base_h7_tp1092_prank_10d90_5d10_plim2_pmv150000p0_frank_10d60_5d40_fmv200000p0_atrnone"
MARKET_DB = DATA_DIR / "STOCK_DAILY_DATA.db"
STRATEGY_DIR = Path(r"D:\work\dfcf\juejin\strategy\ce327750-634e-11f1-b8d7-10ffe0295517")
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")
BACKTEST_START = "2024-06-05 09:00:00"
BACKTEST_END = "2026-06-30 15:30:00"


VARIANTS = [
    {"name": "equal_base", "mode": "equal", "max_target_pct": 0.1092},
    {"name": "primary_boost_mild", "mode": "primary_boost", "primary_pct": 0.112, "fallback_pct": 0.1073, "max_target_pct": 0.112},
    {"name": "primary_boost_strong", "mode": "primary_boost", "primary_pct": 0.115, "fallback_pct": 0.1050, "max_target_pct": 0.115},
    {"name": "fallback_boost_mild", "mode": "primary_boost", "primary_pct": 0.1080, "fallback_pct": 0.1100, "max_target_pct": 0.110},
    {"name": "rank_slope_mild", "mode": "rank_map", "rank_pct": {1: 0.112, 2: 0.111, 3: 0.109, 4: 0.107, 5: 0.105}, "max_target_pct": 0.112},
    {"name": "rank_slope_strong", "mode": "rank_map", "rank_pct": {1: 0.115, 2: 0.112, 3: 0.108, 4: 0.104, 5: 0.100}, "max_target_pct": 0.115},
    {"name": "top2_boost_tail_cut", "mode": "rank_map", "rank_pct": {1: 0.114, 2: 0.113, 3: 0.108, 4: 0.105, 5: 0.100}, "max_target_pct": 0.114},
    {"name": "top1_boost_flat_tail", "mode": "rank_map", "rank_pct": {1: 0.116, 2: 0.109, 3: 0.108, 4: 0.107, 5: 0.106}, "max_target_pct": 0.116},
    {"name": "tail_cut_keep_top", "mode": "rank_map", "rank_pct": {1: 0.111, 2: 0.111, 3: 0.110, 4: 0.107, 5: 0.096}, "max_target_pct": 0.111},
    {"name": "low_vol_cash_buffer", "mode": "rank_map", "rank_pct": {1: 0.108, 2: 0.108, 3: 0.108, 4: 0.108, 5: 0.108}, "max_target_pct": 0.108},
    {
        "name": "equity_dd_loose",
        "mode": "equal",
        "max_target_pct": 0.1092,
        "extra_env": {
            "GM_EQUITY_DD_RISK_MODE": "1",
            "GM_EQUITY_DD_SOFT_TRIGGER": "0.15",
            "GM_EQUITY_DD_HARD_TRIGGER": "0.25",
            "GM_EQUITY_DD_RECOVER_TRIGGER": "0.05",
            "GM_EQUITY_DD_SOFT_SCALE": "0.95",
            "GM_EQUITY_DD_HARD_SCALE": "0.85",
        },
    },
    {
        "name": "equity_dd_mild",
        "mode": "equal",
        "max_target_pct": 0.1092,
        "extra_env": {
            "GM_EQUITY_DD_RISK_MODE": "1",
            "GM_EQUITY_DD_SOFT_TRIGGER": "0.10",
            "GM_EQUITY_DD_HARD_TRIGGER": "0.18",
            "GM_EQUITY_DD_RECOVER_TRIGGER": "0.04",
            "GM_EQUITY_DD_SOFT_SCALE": "0.90",
            "GM_EQUITY_DD_HARD_SCALE": "0.75",
        },
    },
    {
        "name": "equity_dd_resize_loose",
        "mode": "equal",
        "max_target_pct": 0.1092,
        "extra_env": {
            "GM_EQUITY_DD_RISK_MODE": "1",
            "GM_EQUITY_DD_RESIZE_EXISTING": "1",
            "GM_EQUITY_DD_SOFT_TRIGGER": "0.15",
            "GM_EQUITY_DD_HARD_TRIGGER": "0.25",
            "GM_EQUITY_DD_RECOVER_TRIGGER": "0.05",
            "GM_EQUITY_DD_SOFT_SCALE": "0.95",
            "GM_EQUITY_DD_HARD_SCALE": "0.85",
        },
    },
    {
        "name": "index_risk_loose",
        "mode": "equal",
        "max_target_pct": 0.1092,
        "extra_env": {
            "GM_INDEX_RISK_EXIT_MODE": "1",
            "GM_INDEX_RISK_CC_THRESHOLD": "-0.035",
            "GM_INDEX_RISK_INTRADAY_THRESHOLD": "-0.035",
            "GM_INDEX_RISK_BUY_SCALE": "0.85",
        },
    },
    {
        "name": "index_risk_mild",
        "mode": "equal",
        "max_target_pct": 0.1092,
        "extra_env": {
            "GM_INDEX_RISK_EXIT_MODE": "1",
            "GM_INDEX_RISK_CC_THRESHOLD": "-0.025",
            "GM_INDEX_RISK_INTRADAY_THRESHOLD": "-0.025",
            "GM_INDEX_RISK_BUY_SCALE": "0.75",
        },
    },
    {
        "name": "combined_loose",
        "mode": "equal",
        "max_target_pct": 0.1092,
        "extra_env": {
            "GM_EQUITY_DD_RISK_MODE": "1",
            "GM_EQUITY_DD_SOFT_TRIGGER": "0.15",
            "GM_EQUITY_DD_HARD_TRIGGER": "0.25",
            "GM_EQUITY_DD_RECOVER_TRIGGER": "0.05",
            "GM_EQUITY_DD_SOFT_SCALE": "0.95",
            "GM_EQUITY_DD_HARD_SCALE": "0.85",
            "GM_INDEX_RISK_EXIT_MODE": "1",
            "GM_INDEX_RISK_CC_THRESHOLD": "-0.035",
            "GM_INDEX_RISK_INTRADAY_THRESHOLD": "-0.035",
            "GM_INDEX_RISK_BUY_SCALE": "0.85",
        },
    },
]


def _to_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _target_for_row(row: dict, variant: dict) -> float:
    mode = variant["mode"]
    if mode == "equal":
        return float(variant["max_target_pct"])
    if mode == "primary_boost":
        pred = _to_float(row.get("pred_prob"), 0.0) or 0.0
        return float(variant["primary_pct"] if pred >= 2.0 else variant["fallback_pct"])
    if mode == "rank_map":
        rank = int(float(row.get("rank") or 5))
        return float(variant["rank_pct"].get(rank, variant["rank_pct"].get(5)))
    raise ValueError(mode)


def _write_variant_signal(variant: dict, signal_file: Path) -> None:
    with BASE_SIGNAL.open("r", encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
        fieldnames = list(rows[0].keys()) if rows else []
    signal_file.parent.mkdir(parents=True, exist_ok=True)
    with signal_file.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row["target_pct"] = f"{_target_for_row(row, variant):.5f}"
            row["holding_days"] = "7"
            row["score_exit_entry_ratio"] = "1.0"
            row["min_holding_days_before_score_exit"] = "3"
            writer.writerow(row)


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
    with signal_file.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    targets = [_to_float(row.get("target_pct")) for row in rows]
    targets = [value for value in targets if value is not None]
    return {
        "signal_count": len(rows),
        "buy_days": len({row.get("buy_date") for row in rows if row.get("buy_date")}),
        "avg_signal_target_pct": sum(targets) / len(targets) if targets else None,
        "max_signal_target_pct": max(targets) if targets else None,
        "min_signal_target_pct": min(targets) if targets else None,
    }


def _run_backtest(variant: dict, signal_file: Path, log_file: Path) -> int:
    if log_file.exists() and _extract_indicator(log_file):
        return 0
    env = os.environ.copy()
    env.update(
        {
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
            "GM_SYNC_POSITIONS": "0",
            "GM_CASH_BUFFER": "0.995",
            "GM_VERBOSE_TRADES": "0",
            "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.0",
        }
    )
    env.update({str(k): str(v) for k, v in (variant.get("extra_env") or {}).items()})
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
        "10",
        "--holding-days",
        "7",
        "--max-holding-days",
        "7",
        "--target-position-pct",
        f"{float(variant['max_target_pct']):.5f}",
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


def _sort_by_annual(row: dict) -> tuple[float, float, float]:
    return (
        float(row.get("annual") or -999.0),
        float(row.get("sharpe") or -999.0),
        float(row.get("avg_invested_pct") or -999.0),
    )


def _sort_by_sharpe(row: dict) -> tuple[float, float, float]:
    return (
        float(row.get("sharpe") or -999.0),
        float(row.get("annual") or -999.0),
        float(row.get("avg_invested_pct") or -999.0),
    )


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    (REPORT_DIR / "signals").mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    results = []
    for index, variant in enumerate(VARIANTS, start=1):
        signal_file = REPORT_DIR / "signals" / f"{variant['name']}.csv"
        log_file = REPORT_DIR / "logs" / f"{variant['name']}.log"
        if not signal_file.exists():
            _write_variant_signal(variant, signal_file)
        returncode = _run_backtest(variant, signal_file, log_file)
        indicator = _extract_indicator(log_file) or {}
        result = {
            "name": variant["name"],
            "mode": variant["mode"],
            "max_target_pct": variant["max_target_pct"],
            "extra_env": ";".join(f"{k}={v}" for k, v in sorted((variant.get("extra_env") or {}).items())),
            "returncode": returncode,
            "signal_file": str(signal_file),
            "log_file": str(log_file),
            "annual": indicator.get("pnl_ratio_annual"),
            "sharpe": indicator.get("sharp_ratio"),
            "max_drawdown": indicator.get("max_drawdown"),
            "win_ratio": indicator.get("win_ratio"),
            "open_count": indicator.get("open_count"),
            "close_count": indicator.get("close_count"),
            **_signal_stats(signal_file),
            **_exposure_stats(log_file),
        }
        results.append(result)
        _write_rows(REPORT_DIR / "summary.csv", results)
        _write_rows(REPORT_DIR / "summary_by_annual.csv", sorted(results, key=_sort_by_annual, reverse=True))
        _write_rows(REPORT_DIR / "summary_by_sharpe.csv", sorted(results, key=_sort_by_sharpe, reverse=True))
        print(
            f"[{index}/{len(VARIANTS)}] {variant['name']} annual={result.get('annual')} sharpe={result.get('sharpe')} avg={result.get('avg_invested_pct')}",
            flush=True,
        )
    qualified = [
        row
        for row in sorted(results, key=_sort_by_sharpe, reverse=True)
        if float(row.get("annual") or -999.0) >= 2.0
        and float(row.get("sharpe") or -999.0) >= 3.0
        and float(row.get("avg_invested_pct") or -999.0) >= 0.80
    ]
    _write_rows(REPORT_DIR / "summary_qualified_target.csv", qualified)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
