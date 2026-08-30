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
REPORT_DIR = DATA / "reports" / "strategy_agent_dynamic_slippage_70w_20260628"
STRATEGY_DIR = REPORT_DIR / "code_snapshot_dynamic_slippage"
BASE_REPORT = DATA / "reports" / "strategy_agent_tune_four_year_formal_current_rules_20260628"
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")
MARKET_DB = DATA / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
SCORE_DB = BASE_REPORT / "scores" / "grid_scores.duckdb"

CASES = [
    ("w76_19_05_hold4m6_pos80", "w76_19_05_amt90_mv20__hold4m6_c096_e094_ddtight_pos80", 0.80),
    ("w76_19_05_hold4m6_pos85", "w76_19_05_amt90_mv20__hold4m6_c096_e094_ddtight_pos85", 0.85),
    ("w77_18_05_hold4m6_pos80", "w77_18_05_amt90_mv20__hold4m6_c096_e094_ddtight_pos80", 0.80),
    ("w77_18_05_hold4m6_pos85", "w77_18_05_amt90_mv20__hold4m6_c096_e094_ddtight_pos85", 0.85),
]

SLICES = [
    ("full", "2024-06-05 09:00:00", "2026-06-26 15:30:00"),
    ("recent120", "2025-12-24 09:00:00", "2026-06-26 15:30:00"),
    ("recent60", "2026-03-25 09:00:00", "2026-06-26 15:30:00"),
    ("ytd2026", "2026-01-05 09:00:00", "2026-06-26 15:30:00"),
]

BASE_ENV = {
    "GM_OPEN_DAILY_SCORE_EXIT": "1",
    "GM_MAX_DAILY_SELLS": "1",
    "GM_LIGHT_STOP_LOSS_PCT": "none",
    "GM_LOG_EXPOSURE": "1",
    "GM_SKIP_OPEN_LIMIT_UP_BUY": "1",
    "GM_SYNC_POSITIONS": "1",
    "GM_CASH_BUFFER": "0.99",
    "GM_VERBOSE_TRADES": "1",
    "GM_FORCE_SELL_MARKET_ORDER": "0",
    "GM_FORCE_BUY_MARKET_ORDER": "0",
    "GM_INTRADAY_RISK_MODE": "1",
    "GM_INTRADAY_REPLACE_BUY": "0",
    "GM_INTRADAY_RISK_TIMES": "10:00:00,11:00:00,14:30:00",
    "GM_EQUITY_DD_RISK_MODE": "1",
    "GM_EQUITY_DD_RESIZE_EXISTING": "0",
    "GM_EQUITY_DD_SOFT_TRIGGER": "0.06",
    "GM_EQUITY_DD_HARD_TRIGGER": "0.10",
    "GM_EQUITY_DD_RECOVER_TRIGGER": "0.03",
    "GM_EQUITY_DD_SOFT_SCALE": "0.65",
    "GM_EQUITY_DD_HARD_SCALE": "0.45",
    "GM_SCORE_EXIT_ENTRY_RATIO": "0.94",
    "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "1",
    "GM_SCORE_CONTINUE_ENTRY_RATIO": "0.96",
    "GM_ADAPTIVE_LIQUIDITY_SLIPPAGE": "1",
    "GM_ADAPTIVE_SLIPPAGE_ORDER_VALUE": "700000",
    "GM_ADAPTIVE_SLIPPAGE_BASE": "0.0015",
    "GM_ADAPTIVE_SLIPPAGE_IMPACT_COEF": "0.0200",
    "GM_ADAPTIVE_SLIPPAGE_CAP": "0.0065",
    "GM_ADAPTIVE_SLIPPAGE_AMOUNT_UNIT": "1000",
    "GM_ADAPTIVE_BUY_SLIPPAGE_MULT": "1.0",
    "GM_ADAPTIVE_SELL_SLIPPAGE_MULT": "0.0",
}


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


def _run(case_name: str, stem: str, target_pct: float, tag: str, start: str, end: str) -> dict[str, Any]:
    signal_file = BASE_REPORT / "signals" / f"{stem}.csv"
    score_table = f"score_{stem}"
    log_file = REPORT_DIR / "logs" / f"{case_name}__adaptive70w_buyonly__{tag}.log"
    indicator = _extract_indicator(log_file)
    if indicator is None:
        env = os.environ.copy()
        env.update(BASE_ENV)
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
            "1",
            "--holding-days",
            "4",
            "--max-holding-days",
            "6",
            "--target-position-pct",
            f"{target_pct:.2f}",
            "--score-db",
            str(SCORE_DB),
            "--score-table",
            score_table,
            "--market-db",
            str(MARKET_DB),
            "--backtest-start",
            start,
            "--backtest-end",
            end,
            "--backtest-adjust",
            "none",
            "--backtest-initial-cash",
            "600000",
            "--backtest-slippage-ratio",
            "0",
            "--stop-loss-pct",
            "0.05",
            "--take-profit-pct",
            "0.09",
        ]
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, cwd=str(MAIN), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        indicator = _extract_indicator(log_file)
        if proc.returncode != 0 or indicator is None:
            return {"case": case_name, "slice": tag, "returncode": proc.returncode, "log_file": str(log_file)}
    return {
        "case": case_name,
        "slice": tag,
        "annual": indicator.get("pnl_ratio_annual"),
        "pnl_ratio": indicator.get("pnl_ratio"),
        "sharpe": indicator.get("sharp_ratio"),
        "max_drawdown": indicator.get("max_drawdown"),
        "win_ratio": indicator.get("win_ratio"),
        "open_count": indicator.get("open_count"),
        "close_count": indicator.get("close_count"),
        "log_file": str(log_file),
        "returncode": 0,
    }


def main() -> None:
    detail: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for case_name, stem, target_pct in CASES:
        rows = [_run(case_name, stem, target_pct, tag, start, end) for tag, start, end in SLICES]
        detail.extend(rows)
        out = {
            "case": case_name,
            "signal_file": str(BASE_REPORT / "signals" / f"{stem}.csv"),
            "score_table": f"score_{stem}",
            "target_pct": target_pct,
        }
        for row in rows:
            prefix = row["slice"]
            for key in ["annual", "pnl_ratio", "sharpe", "max_drawdown", "win_ratio", "open_count"]:
                out[f"{prefix}_{key}"] = row.get(key)
        summary.append(out)
        print(json.dumps(out, ensure_ascii=False), flush=True)
        _write_rows(REPORT_DIR / "adaptive70w_buyonly_midweights_detail.csv", detail)
        _write_rows(REPORT_DIR / "adaptive70w_buyonly_midweights_summary.csv", summary)


if __name__ == "__main__":
    main()

