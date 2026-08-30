from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import os
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA = ROOT / "quant" / "data_file"
REPORT_DIR = DATA / "reports" / "strategy_agent_latest_l4_top3_frequency_grid_20260630"
STRATEGY_DIR = DATA / "reports" / "strategy_agent_dynamic_slippage_70w_20260628" / "code_snapshot_dynamic_slippage"
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")
MARKET_DB = DATA / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
SCORE_DB = REPORT_DIR / "scores" / "grid_scores.duckdb"
SCORE_TABLE = "score_w72_23_05_amt150_mv30_latest_l4_full"
SIGNAL_DIR = REPORT_DIR / "repro_frequency_refine_signals"
OUT_LOG_DIR = REPORT_DIR / "repro_dd_refine_logs"
OUT_CSV = REPORT_DIR / "reproducible_dd_refine_20260630.csv"
OUT_JSON = REPORT_DIR / "reproducible_dd_refine_20260630.json"


SIGNALS = [
    {"signal": "top3_equal34", "target_cap": 0.34},
    {"signal": "top3_equal30", "target_cap": 0.30},
    {"signal": "scale1p45_cap39", "target_cap": 0.39},
    {"signal": "scale1p30_cap35", "target_cap": 0.35},
    {"signal": "gap_nochase_scale1p45", "target_cap": 0.39},
]


DD_MODES = [
    {"name": "dd_off", "enabled": "0", "soft": "0.08", "hard": "0.14", "recover": "0.04", "soft_scale": "0.80", "hard_scale": "0.60"},
    {"name": "dd_mid", "enabled": "1", "soft": "0.07", "hard": "0.12", "recover": "0.035", "soft_scale": "0.72", "hard_scale": "0.50"},
    {"name": "dd_strict", "enabled": "1", "soft": "0.06", "hard": "0.10", "recover": "0.03", "soft_scale": "0.65", "hard_scale": "0.45"},
    {"name": "dd_hard", "enabled": "1", "soft": "0.05", "hard": "0.08", "recover": "0.025", "soft_scale": "0.55", "hard_scale": "0.35"},
]


SELL_RULE = {
    "holding_days": 2,
    "max_holding_days": 3,
    "score_exit": 0.98,
    "score_continue": 0.99,
    "min_score_exit_days": 1,
    "day_drop_ratio": 0.995,
    "max_daily_sells": 1,
}


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fields} for row in rows])


def _extract_indicator(log_file: Path) -> dict[str, Any] | None:
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


def _run(signal_case: dict[str, Any], dd_mode: dict[str, Any]) -> dict[str, Any]:
    signal_name = str(signal_case["signal"])
    signal_file = SIGNAL_DIR / f"{signal_name}.csv"
    if not signal_file.exists():
        raise FileNotFoundError(signal_file)
    name = f"{signal_name}__{dd_mode['name']}"
    log_file = OUT_LOG_DIR / f"{name}.log"
    indicator = _extract_indicator(log_file)
    returncode = 0
    if indicator is None:
        env = os.environ.copy()
        env.update(
            {
                "GM_OPEN_DAILY_SCORE_EXIT": "1",
                "GM_MAX_DAILY_SELLS": str(SELL_RULE["max_daily_sells"]),
                "GM_LIGHT_STOP_LOSS_PCT": "none",
                "GM_LOG_EXPOSURE": "1",
                "GM_SKIP_OPEN_LIMIT_UP_BUY": "1",
                "GM_SYNC_POSITIONS": "1",
                "GM_CASH_BUFFER": "0.99",
                "GM_VERBOSE_TRADES": "1",
                "GM_FORCE_SELL_MARKET_ORDER": "0",
                "GM_FORCE_BUY_MARKET_ORDER": "0",
                "GM_INTRADAY_RISK_MODE": "0",
                "GM_INTRADAY_REPLACE_BUY": "0",
                "GM_EQUITY_DD_RISK_MODE": dd_mode["enabled"],
                "GM_EQUITY_DD_RESIZE_EXISTING": "0",
                "GM_EQUITY_DD_SOFT_TRIGGER": dd_mode["soft"],
                "GM_EQUITY_DD_HARD_TRIGGER": dd_mode["hard"],
                "GM_EQUITY_DD_RECOVER_TRIGGER": dd_mode["recover"],
                "GM_EQUITY_DD_SOFT_SCALE": dd_mode["soft_scale"],
                "GM_EQUITY_DD_HARD_SCALE": dd_mode["hard_scale"],
                "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": str(SELL_RULE["min_score_exit_days"]),
                "GM_SCORE_EXIT_ENTRY_RATIO": str(SELL_RULE["score_exit"]),
                "GM_SCORE_CONTINUE_ENTRY_RATIO": str(SELL_RULE["score_continue"]),
                "GM_SCORE_STOP_LOSS_DAY_DROP_RATIO": str(SELL_RULE["day_drop_ratio"]),
                "GM_ADAPTIVE_LIQUIDITY_SLIPPAGE": "1",
                "GM_ADAPTIVE_SLIPPAGE_ORDER_VALUE": "700000",
                "GM_ADAPTIVE_SLIPPAGE_BASE": "0.0015",
                "GM_ADAPTIVE_SLIPPAGE_IMPACT_COEF": "0.0200",
                "GM_ADAPTIVE_SLIPPAGE_CAP": "0.0065",
                "GM_ADAPTIVE_SLIPPAGE_AMOUNT_UNIT": "1000",
                "GM_ADAPTIVE_BUY_SLIPPAGE_MULT": "1.0",
                "GM_ADAPTIVE_SELL_SLIPPAGE_MULT": "0.0",
                "GM_RESIZE_HELD_ON_SIGNAL": "0",
                "GM_DEFER_EXITS_WITHOUT_BUY_SIGNAL": "0",
                "GM_INDEX_RISK_EXIT_MODE": "0",
                "GM_BREADTH_RISK_EXIT_MODE": "0",
            }
        )
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
            "5",
            "--holding-days",
            str(SELL_RULE["holding_days"]),
            "--max-holding-days",
            str(SELL_RULE["max_holding_days"]),
            "--target-position-pct",
            str(float(signal_case["target_cap"])),
            "--score-db",
            str(SCORE_DB),
            "--score-table",
            SCORE_TABLE,
            "--market-db",
            str(MARKET_DB),
            "--backtest-start",
            "2022-06-07 09:00:00",
            "--backtest-end",
            "2026-06-29 15:30:00",
            "--backtest-adjust",
            "none",
            "--backtest-initial-cash",
            "600000",
            "--backtest-slippage-ratio",
            "0",
            "--stop-loss-pct",
            "0.05",
            "--take-profit-pct",
            "0.08",
        ]
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, cwd=str(MAIN), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        returncode = proc.returncode
        indicator = _extract_indicator(log_file)
    return {
        "name": name,
        "signal": signal_name,
        "dd_mode": dd_mode["name"],
        "returncode": returncode,
        "annual": indicator.get("pnl_ratio_annual") if indicator else None,
        "pnl_ratio": indicator.get("pnl_ratio") if indicator else None,
        "sharpe": indicator.get("sharp_ratio") if indicator else None,
        "max_drawdown": indicator.get("max_drawdown") if indicator else None,
        "win_ratio": indicator.get("win_ratio") if indicator else None,
        "open_count": indicator.get("open_count") if indicator else None,
        "close_count": indicator.get("close_count") if indicator else None,
        "signal_file": str(signal_file),
        "log_file": str(log_file),
        "note": "research-only dd refine; no production parameter change",
    }


def main() -> None:
    results: list[dict[str, Any]] = []
    for signal_case in SIGNALS:
        for dd_mode in DD_MODES:
            row = _run(signal_case, dd_mode)
            results.append(row)
            _write_rows(OUT_CSV, results)
            OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

