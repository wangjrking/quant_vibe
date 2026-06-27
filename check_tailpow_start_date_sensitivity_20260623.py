from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_ROOT = DATA_DIR / "reports" / "strategy_agent_goal_high_annual_20260623"
OUT_DIR = REPORT_ROOT / "current_formal_top1keep_tailpow_start_sensitivity"
MARKET_DB = DATA_DIR / "STOCK_DAILY_DATA.db"
STRATEGY_DIR = Path(r"D:\work\dfcf\juejin\strategy\ce327750-634e-11f1-b8d7-10ffe0295517")
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")

VARIANTS = [
    {
        "name": "best_g1p38_o0p942",
        "signal_file": REPORT_ROOT / "current_formal_top1keep_tailpow_peak" / "signals" / "tp_peak_g1p38_o0p942.csv",
        "score_db": REPORT_ROOT / "current_formal_top1keep_tailpow_peak" / "scores_top1keep_tailpow_peak.db",
        "score_table": "score_tp_peak_g1p38_o0p942",
        "score_exit_ratio": "0.942",
    },
    {
        "name": "neighbor_g1p40_o0p946",
        "signal_file": REPORT_ROOT / "current_formal_top1keep_tailpow_peak" / "signals" / "tp_peak_g1p4_o0p946.csv",
        "score_db": REPORT_ROOT / "current_formal_top1keep_tailpow_peak" / "scores_top1keep_tailpow_peak.db",
        "score_table": "score_tp_peak_g1p4_o0p946",
        "score_exit_ratio": "0.946",
    },
]

STARTS = [
    ("start_20240605", "2024-06-05 09:00:00"),
    ("start_20240902", "2024-09-02 09:00:00"),
    ("start_20241202", "2024-12-02 09:00:00"),
    ("start_20250102", "2025-01-02 09:00:00"),
    ("start_20250401", "2025-04-01 09:00:00"),
    ("start_20250701", "2025-07-01 09:00:00"),
    ("start_20251008", "2025-10-08 09:00:00"),
    ("start_20260105", "2026-01-05 09:00:00"),
]

BACKTEST_END = "2026-06-23 15:30:00"

BASE_ENV = {
    "GM_OPEN_DAILY_SCORE_EXIT": "1",
    "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "2",
    "GM_MAX_DAILY_SELLS": "4",
    "GM_STOP_LOSS_PCT": "0.08",
    "GM_TAKE_PROFIT_PCT": "none",
    "GM_LIGHT_STOP_LOSS_PCT": "none",
    "GM_LOG_EXPOSURE": "1",
    "GM_SKIP_OPEN_LIMIT_UP_BUY": "1",
    "GM_SYNC_POSITIONS": "1",
    "GM_CASH_BUFFER": "0.99",
    "GM_VERBOSE_TRADES": "0",
    "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.02",
    "GM_EQUITY_DD_RISK_MODE": "0",
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


def _run_case(variant: dict, start_name: str, start: str) -> dict:
    log_file = OUT_DIR / "logs" / f"{variant['name']}_{start_name}.log"
    indicator = _extract_indicator(log_file)
    if indicator is None:
        env = os.environ.copy()
        env.update(BASE_ENV)
        env["GM_SCORE_EXIT_ENTRY_RATIO"] = variant["score_exit_ratio"]
        command = [
            str(JUEJIN_PYTHON),
            str(MAIN / "run_juejin_signal_backtest.py"),
            "--strategy-dir",
            str(STRATEGY_DIR),
            "--signal-file",
            str(variant["signal_file"]),
            "--log-file",
            str(log_file),
            "--max-positions",
            "4",
            "--holding-days",
            "5",
            "--max-holding-days",
            "5",
            "--target-position-pct",
            "0.5",
            "--score-db",
            str(variant["score_db"]),
            "--score-table",
            str(variant["score_table"]),
            "--market-db",
            str(MARKET_DB),
            "--backtest-start",
            start,
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
        indicator = _extract_indicator(log_file)
        returncode = proc.returncode
    else:
        returncode = 0
    return {
        "variant": variant["name"],
        "start_name": start_name,
        "backtest_start": start,
        "backtest_end": BACKTEST_END,
        "returncode": returncode,
        "annual": indicator.get("pnl_ratio_annual") if indicator else None,
        "pnl_ratio": indicator.get("pnl_ratio") if indicator else None,
        "sharpe": indicator.get("sharp_ratio") if indicator else None,
        "max_drawdown": indicator.get("max_drawdown") if indicator else None,
        "win_ratio": indicator.get("win_ratio") if indicator else None,
        "open_count": indicator.get("open_count") if indicator else None,
        "close_count": indicator.get("close_count") if indicator else None,
        "log_file": str(log_file),
    }


def main() -> int:
    rows = []
    for variant in VARIANTS:
        for start_name, start in STARTS:
            row = _run_case(variant, start_name, start)
            rows.append(row)
            print(
                f"{row['variant']} {row['start_name']} annual={row['annual']} "
                f"pnl={row['pnl_ratio']} sharpe={row['sharpe']} maxdd={row['max_drawdown']}",
                flush=True,
            )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with (OUT_DIR / "start_sensitivity_summary.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (OUT_DIR / "start_sensitivity_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
