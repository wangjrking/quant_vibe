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

OUT_SIGNAL_DIR = REPORT_DIR / "formal_score_frequency_followup_signals"
OUT_LOG_DIR = REPORT_DIR / "formal_score_frequency_followup_logs"
OUT_CSV = REPORT_DIR / "formal_score_frequency_followup_20260630.csv"
OUT_JSON = REPORT_DIR / "formal_score_frequency_followup_20260630.json"


BASES = {
    "dyn_mild_09_12_15": REPORT_DIR / "dynamic_target_signals" / "dyn_mild_09_12_15.csv",
    "dyn_mild_bonus_17": REPORT_DIR / "dynamic_target_signals" / "dyn_mild_bonus_17.csv",
    "top3_cool2d18": REPORT_DIR
    / "signals"
    / "w72_23_05_amt150_mv30_top3_pos25_cool2d18_h2m3_e097_c098_daydrop99.csv",
}


CASES: list[dict[str, Any]] = [
    {
        "name": "dyn_mild_formal_h2m3_ms1",
        "base": "dyn_mild_09_12_15",
        "max_positions": 5,
        "target_position_pct": 0.15,
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.97,
        "score_continue": 0.98,
        "max_daily_sells": 1,
        "open_daily_score_exit": 1,
    },
    {
        "name": "dyn_mild_formal_h2m3_ms0",
        "base": "dyn_mild_09_12_15",
        "max_positions": 5,
        "target_position_pct": 0.15,
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.97,
        "score_continue": 0.98,
        "max_daily_sells": 0,
        "open_daily_score_exit": 1,
    },
    {
        "name": "dyn_mild_formal_h1m2_ms1",
        "base": "dyn_mild_09_12_15",
        "max_positions": 5,
        "target_position_pct": 0.15,
        "holding_days": 1,
        "max_holding_days": 2,
        "score_exit": 0.99,
        "score_continue": 0.995,
        "max_daily_sells": 1,
        "open_daily_score_exit": 1,
    },
    {
        "name": "dyn_bonus17_formal_h2m3_ms1",
        "base": "dyn_mild_bonus_17",
        "max_positions": 5,
        "target_position_pct": 0.17,
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.97,
        "score_continue": 0.98,
        "max_daily_sells": 1,
        "open_daily_score_exit": 1,
    },
    {
        "name": "top3_cool2d18_formal_h2m3_ms1",
        "base": "top3_cool2d18",
        "max_positions": 3,
        "target_position_pct": 0.25,
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.97,
        "score_continue": 0.98,
        "max_daily_sells": 1,
        "open_daily_score_exit": 1,
    },
    {
        "name": "top3_cool2d18_formal_h2m3_ms0",
        "base": "top3_cool2d18",
        "max_positions": 3,
        "target_position_pct": 0.25,
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.97,
        "score_continue": 0.98,
        "max_daily_sells": 0,
        "open_daily_score_exit": 1,
    },
    {
        "name": "top3_cool2d18_formal_h1m2_ms1",
        "base": "top3_cool2d18",
        "max_positions": 3,
        "target_position_pct": 0.25,
        "holding_days": 1,
        "max_holding_days": 2,
        "score_exit": 0.99,
        "score_continue": 0.995,
        "max_daily_sells": 1,
        "open_daily_score_exit": 1,
    },
    {
        "name": "dyn_mild_no_scoreexit_h2m3_ms1",
        "base": "dyn_mild_09_12_15",
        "max_positions": 5,
        "target_position_pct": 0.15,
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.97,
        "score_continue": 0.98,
        "max_daily_sells": 1,
        "open_daily_score_exit": 0,
    },
    {
        "name": "dyn_mild_no_scoreexit_h1m2_ms1",
        "base": "dyn_mild_09_12_15",
        "max_positions": 5,
        "target_position_pct": 0.15,
        "holding_days": 1,
        "max_holding_days": 2,
        "score_exit": 0.99,
        "score_continue": 0.995,
        "max_daily_sells": 1,
        "open_daily_score_exit": 0,
    },
    {
        "name": "top3_cool2d18_no_scoreexit_h2m3_ms1",
        "base": "top3_cool2d18",
        "max_positions": 3,
        "target_position_pct": 0.25,
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.97,
        "score_continue": 0.98,
        "max_daily_sells": 1,
        "open_daily_score_exit": 0,
    },
    {
        "name": "top3_cool2d18_no_scoreexit_h1m2_ms1",
        "base": "top3_cool2d18",
        "max_positions": 3,
        "target_position_pct": 0.25,
        "holding_days": 1,
        "max_holding_days": 2,
        "score_exit": 0.99,
        "score_continue": 0.995,
        "max_daily_sells": 1,
        "open_daily_score_exit": 0,
    },
]


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


def _make_signal(case: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    base_path = BASES[str(case["base"])]
    rows = list(csv.DictReader(base_path.open("r", encoding="utf-8-sig", newline="")))
    for row in rows:
        row["holding_days"] = str(int(case["holding_days"]))
        row["max_holding_days"] = str(int(case["max_holding_days"]))
        row["score_exit_entry_ratio"] = f"{float(case['score_exit']):.5f}"
        row["score_continue_entry_ratio"] = f"{float(case['score_continue']):.5f}"
        row["min_holding_days_before_score_exit"] = "1"
        row["strategy_variant"] = str(case["name"])
        row["filter_name"] = str(case["name"])
        row["dynamic_hold_name"] = str(case["name"])
    output = OUT_SIGNAL_DIR / f"{case['name']}.csv"
    _write_rows(output, rows)
    return output, {
        "signal_rows": len(rows),
        "signal_days": len({row.get("signal_date") for row in rows}),
    }


def _run(case: dict[str, Any]) -> dict[str, Any]:
    signal_file, signal_meta = _make_signal(case)
    log_file = OUT_LOG_DIR / f"{case['name']}.log"
    indicator = _extract_indicator(log_file)
    returncode = 0
    if indicator is None:
        env = os.environ.copy()
        env.update(
            {
                "GM_OPEN_DAILY_SCORE_EXIT": str(int(case.get("open_daily_score_exit", 1))),
                "GM_MAX_DAILY_SELLS": str(int(case["max_daily_sells"])),
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
                "GM_EQUITY_DD_RISK_MODE": "1",
                "GM_EQUITY_DD_RESIZE_EXISTING": "0",
                "GM_EQUITY_DD_SOFT_TRIGGER": "0.08",
                "GM_EQUITY_DD_HARD_TRIGGER": "0.14",
                "GM_EQUITY_DD_RECOVER_TRIGGER": "0.04",
                "GM_EQUITY_DD_SOFT_SCALE": "0.80",
                "GM_EQUITY_DD_HARD_SCALE": "0.60",
                "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "1",
                "GM_SCORE_EXIT_ENTRY_RATIO": str(float(case["score_exit"])),
                "GM_SCORE_CONTINUE_ENTRY_RATIO": str(float(case["score_continue"])),
                "GM_SCORE_STOP_LOSS_DAY_DROP_RATIO": "0.99",
                "GM_ADAPTIVE_LIQUIDITY_SLIPPAGE": "1",
                "GM_ADAPTIVE_SLIPPAGE_ORDER_VALUE": "700000",
                "GM_ADAPTIVE_SLIPPAGE_BASE": "0.0015",
                "GM_ADAPTIVE_SLIPPAGE_IMPACT_COEF": "0.0200",
                "GM_ADAPTIVE_SLIPPAGE_CAP": "0.0065",
                "GM_ADAPTIVE_SLIPPAGE_AMOUNT_UNIT": "1000",
                "GM_ADAPTIVE_BUY_SLIPPAGE_MULT": "1.0",
                "GM_ADAPTIVE_SELL_SLIPPAGE_MULT": "0.0",
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
            str(int(case["max_positions"])),
            "--holding-days",
            str(int(case["holding_days"])),
            "--max-holding-days",
            str(int(case["max_holding_days"])),
            "--target-position-pct",
            str(float(case["target_position_pct"])),
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
        **case,
        **signal_meta,
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
        "score_db": str(SCORE_DB),
        "score_table": SCORE_TABLE,
        "open_daily_score_exit": int(case.get("open_daily_score_exit", 1)),
        "intraday_risk": 0,
        "note": "research-only formal-score frequency follow-up; production unchanged",
    }


def main() -> None:
    for name, path in BASES.items():
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path}")
    results: list[dict[str, Any]] = []
    for case in CASES:
        row = _run(case)
        results.append(row)
        _write_rows(OUT_CSV, results)
        OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

