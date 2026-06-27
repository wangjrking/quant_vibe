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
REPORT_DIR = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_5d10d_risk_exit_refine_20260621"
)
STRATEGY_DIR = Path(r"D:\work\dfcf\juejin\strategy\ce327750-634e-11f1-b8d7-10ffe0295517")
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")
MARKET_DB = DATA_DIR / "STOCK_DAILY_DATA.db"
SIGNAL_FILE = (
    ROOT
    / "quant"
    / "main"
    / "strategy_library"
    / "exploration"
    / "l5_formal_5d10d_candidate_20260621"
    / "signals"
    / "candidate_signals.csv"
)
SCORE_DB = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_5d10d_tiered_score_refine_20260621"
    / "tiered_scores.db"
)
SCORE_TABLE = "tier_p90_f50_plim2_pmv150000p0_fmv200000p0"
BACKTEST_START = "2024-06-05 09:00:00"
BACKTEST_END = "2026-06-30 15:30:00"


VARIANTS = [
    {"name": "baseline_sl08", "stop_loss": 0.08, "take_profit": None, "light_stop": None, "score_exit_ratio": 0.95, "score_exit_rank": None},
    {"name": "sl05", "stop_loss": 0.05, "take_profit": None, "light_stop": None, "score_exit_ratio": 0.95, "score_exit_rank": None},
    {"name": "sl06", "stop_loss": 0.06, "take_profit": None, "light_stop": None, "score_exit_ratio": 0.95, "score_exit_rank": None},
    {"name": "sl10", "stop_loss": 0.10, "take_profit": None, "light_stop": None, "score_exit_ratio": 0.95, "score_exit_rank": None},
    {"name": "slnone", "stop_loss": None, "take_profit": None, "light_stop": None, "score_exit_ratio": 0.95, "score_exit_rank": None},
    {"name": "sl08_tp12", "stop_loss": 0.08, "take_profit": 0.12, "light_stop": None, "score_exit_ratio": 0.95, "score_exit_rank": None},
    {"name": "sl08_tp18", "stop_loss": 0.08, "take_profit": 0.18, "light_stop": None, "score_exit_ratio": 0.95, "score_exit_rank": None},
    {"name": "sl06_tp12", "stop_loss": 0.06, "take_profit": 0.12, "light_stop": None, "score_exit_ratio": 0.95, "score_exit_rank": None},
    {"name": "sl06_tp18", "stop_loss": 0.06, "take_profit": 0.18, "light_stop": None, "score_exit_ratio": 0.95, "score_exit_rank": None},
    {"name": "light03", "stop_loss": 0.08, "take_profit": None, "light_stop": 0.03, "min_light_hold": 1, "score_exit_ratio": 0.95, "score_exit_rank": None},
    {"name": "light04", "stop_loss": 0.08, "take_profit": None, "light_stop": 0.04, "min_light_hold": 1, "score_exit_ratio": 0.95, "score_exit_rank": None},
    {"name": "light05", "stop_loss": 0.08, "take_profit": None, "light_stop": 0.05, "min_light_hold": 1, "score_exit_ratio": 0.95, "score_exit_rank": None},
    {"name": "ser090", "stop_loss": 0.08, "take_profit": None, "light_stop": None, "score_exit_ratio": 0.90, "score_exit_rank": None},
    {"name": "ser085", "stop_loss": 0.08, "take_profit": None, "light_stop": None, "score_exit_ratio": 0.85, "score_exit_rank": None},
    {"name": "rank060", "stop_loss": 0.08, "take_profit": None, "light_stop": None, "score_exit_ratio": 0.95, "score_exit_rank": 0.60},
    {"name": "rank070", "stop_loss": 0.08, "take_profit": None, "light_stop": None, "score_exit_ratio": 0.95, "score_exit_rank": 0.70},
    {"name": "rank080", "stop_loss": 0.08, "take_profit": None, "light_stop": None, "score_exit_ratio": 0.95, "score_exit_rank": 0.80},
    {"name": "sl06_rank070", "stop_loss": 0.06, "take_profit": None, "light_stop": None, "score_exit_ratio": 0.95, "score_exit_rank": 0.70},
    {"name": "sl06_light04", "stop_loss": 0.06, "take_profit": None, "light_stop": 0.04, "min_light_hold": 1, "score_exit_ratio": 0.95, "score_exit_rank": None},
    {"name": "sl06_tp12_rank070", "stop_loss": 0.06, "take_profit": 0.12, "light_stop": None, "score_exit_ratio": 0.95, "score_exit_rank": 0.70},
]


def _value(value) -> str:
    return "none" if value is None else str(value)


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


def _run_backtest(variant: dict, log_file: Path) -> int:
    if log_file.exists() and _extract_indicator(log_file):
        return 0
    env = os.environ.copy()
    env.update(
        {
            "GM_OPEN_DAILY_SCORE_EXIT": "1",
            "GM_SCORE_EXIT_ENTRY_RATIO": _value(variant["score_exit_ratio"]),
            "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "2",
            "GM_MAX_DAILY_SELLS": "1",
            "GM_STOP_LOSS_PCT": _value(variant["stop_loss"]),
            "GM_TAKE_PROFIT_PCT": _value(variant["take_profit"]),
            "GM_LIGHT_STOP_LOSS_PCT": _value(variant["light_stop"]),
            "GM_MIN_HOLDING_DAYS_BEFORE_LIGHT_STOP": str(variant.get("min_light_hold", 0)),
            "GM_SCORE_EXIT_RANK": _value(variant["score_exit_rank"]),
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
    command = [
        str(JUEJIN_PYTHON),
        str(MAIN / "run_juejin_signal_backtest.py"),
        "--strategy-dir",
        str(STRATEGY_DIR),
        "--signal-file",
        str(SIGNAL_FILE),
        "--log-file",
        str(log_file),
        "--max-positions",
        "10",
        "--holding-days",
        "5",
        "--max-holding-days",
        "5",
        "--target-position-pct",
        "0.098",
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


def _sort_key(row: dict) -> tuple[float, float, float]:
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
    (REPORT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    results = []
    for index, variant in enumerate(VARIANTS, start=1):
        log_file = REPORT_DIR / "logs" / f"{variant['name']}.log"
        returncode = _run_backtest(variant, log_file)
        indicator = _extract_indicator(log_file) or {}
        result = {
            **variant,
            "returncode": returncode,
            "signal_file": str(SIGNAL_FILE),
            "score_db": str(SCORE_DB),
            "score_table": SCORE_TABLE,
            "log_file": str(log_file),
            "annual": indicator.get("pnl_ratio_annual"),
            "sharpe": indicator.get("sharp_ratio"),
            "max_drawdown": indicator.get("max_drawdown"),
            "win_ratio": indicator.get("win_ratio"),
            "open_count": indicator.get("open_count"),
            "close_count": indicator.get("close_count"),
            **_exposure_stats(log_file),
        }
        results.append(result)
        _write_rows(REPORT_DIR / "summary.csv", results)
        _write_rows(REPORT_DIR / "summary_by_sharpe.csv", sorted(results, key=_sort_key, reverse=True))
        print(
            f"[{index}/{len(VARIANTS)}] {variant['name']} annual={result.get('annual')} sharpe={result.get('sharpe')} avg={result.get('avg_invested_pct')}",
            flush=True,
        )
    qualified = [
        row
        for row in sorted(results, key=_sort_key, reverse=True)
        if float(row.get("annual") or -999) >= 2.0
        and float(row.get("sharpe") or -999) >= 3.0
        and float(row.get("avg_invested_pct") or -999) >= 0.80
    ]
    _write_rows(REPORT_DIR / "summary_qualified_target.csv", qualified)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
