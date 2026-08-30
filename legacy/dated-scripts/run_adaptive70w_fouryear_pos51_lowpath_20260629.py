from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import os
import subprocess
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA = ROOT / "quant" / "data_file"
SOURCE_DIR = DATA / "reports" / "strategy_agent_adaptive70w_fouryear_pos52_baseline_posgrid_20260629"
OUT_DIR = DATA / "reports" / "strategy_agent_adaptive70w_fouryear_pos51_lowpath_20260629"
STRATEGY_DIR = DATA / "reports" / "strategy_agent_dynamic_slippage_70w_20260628" / "code_snapshot_dynamic_slippage"
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")
MARKET_DB = DATA / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
SCORE_DB = DATA / "reports" / "strategy_agent_adaptive70w_fouryear_fullwindow_20260629" / "scores" / "grid_scores.duckdb"

CASE_KEY = "w72_23_05_amt150_mv30__hold3m5_c097_e096_ddtight_pos65__amt150000_mv300000_pc150_pos52_cool2d18"
SIGNAL_FILE = SOURCE_DIR / "signals" / f"{CASE_KEY}__pos51.csv"
SCORE_TABLE = f"score_{CASE_KEY}"
BACKTEST_END = "2026-06-26 15:30:00"

ANCHOR_STARTS = {
    "20240102": [
        "2023-12-27 09:00:00",
        "2023-12-28 09:00:00",
        "2023-12-29 09:00:00",
        "2024-01-02 09:00:00",
        "2024-01-03 09:00:00",
        "2024-01-04 09:00:00",
        "2024-01-05 09:00:00",
    ],
    "20250102": [
        "2024-12-27 09:00:00",
        "2024-12-30 09:00:00",
        "2024-12-31 09:00:00",
        "2025-01-02 09:00:00",
        "2025-01-03 09:00:00",
        "2025-01-06 09:00:00",
        "2025-01-07 09:00:00",
    ],
    "20250701": [
        "2025-06-26 09:00:00",
        "2025-06-27 09:00:00",
        "2025-06-30 09:00:00",
        "2025-07-01 09:00:00",
        "2025-07-02 09:00:00",
        "2025-07-03 09:00:00",
        "2025-07-04 09:00:00",
    ],
    "20251009": [
        "2025-09-26 09:00:00",
        "2025-09-29 09:00:00",
        "2025-09-30 09:00:00",
        "2025-10-09 09:00:00",
        "2025-10-10 09:00:00",
        "2025-10-13 09:00:00",
        "2025-10-14 09:00:00",
    ],
    "20260105": [
        "2025-12-29 09:00:00",
        "2025-12-30 09:00:00",
        "2025-12-31 09:00:00",
        "2026-01-05 09:00:00",
        "2026-01-06 09:00:00",
        "2026-01-07 09:00:00",
        "2026-01-08 09:00:00",
    ],
}

DD_ENV = {
    "GM_EQUITY_DD_RISK_MODE": "1",
    "GM_EQUITY_DD_RESIZE_EXISTING": "0",
    "GM_EQUITY_DD_SOFT_TRIGGER": "0.07",
    "GM_EQUITY_DD_HARD_TRIGGER": "0.12",
    "GM_EQUITY_DD_RECOVER_TRIGGER": "0.035",
    "GM_EQUITY_DD_SOFT_SCALE": "0.75",
    "GM_EQUITY_DD_HARD_SCALE": "0.55",
    "GM_EQUITY_DD_STRICT_WHEN_DRAWDOWN": "1",
    "GM_EQUITY_DD_STRICT_TRIGGER": "0.08",
    "GM_EQUITY_DD_STRICT_SOFT_TRIGGER": "0.06",
    "GM_EQUITY_DD_STRICT_HARD_TRIGGER": "0.10",
    "GM_EQUITY_DD_STRICT_RECOVER_TRIGGER": "0.03",
    "GM_EQUITY_DD_STRICT_SOFT_SCALE": "0.65",
    "GM_EQUITY_DD_STRICT_HARD_SCALE": "0.45",
}

BASE_ENV = {
    "GM_OPEN_DAILY_SCORE_EXIT": "1",
    "GM_MAX_DAILY_SELLS": "0",
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
    "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "1",
    "GM_SCORE_CONTINUE_ENTRY_RATIO": "0.97",
    "GM_SCORE_EXIT_ENTRY_RATIO": "0.96",
    "GM_SCORE_STOP_LOSS_DAY_DROP_RATIO": "none",
    "GM_ADAPTIVE_LIQUIDITY_SLIPPAGE": "1",
    "GM_ADAPTIVE_SLIPPAGE_ORDER_VALUE": "700000",
    "GM_ADAPTIVE_SLIPPAGE_BASE": "0.0015",
    "GM_ADAPTIVE_SLIPPAGE_IMPACT_COEF": "0.0200",
    "GM_ADAPTIVE_SLIPPAGE_CAP": "0.0065",
    "GM_ADAPTIVE_SLIPPAGE_AMOUNT_UNIT": "1000",
    "GM_ADAPTIVE_BUY_SLIPPAGE_MULT": "1.0",
    "GM_ADAPTIVE_SELL_SLIPPAGE_MULT": "0.25",
    **DD_ENV,
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


def _run(anchor: str, offset: int, start: str) -> dict[str, Any]:
    tag = f"{anchor}_{offset:+d}".replace("+", "p").replace("-", "m")
    log_file = OUT_DIR / "logs" / f"{CASE_KEY}__{tag}.log"
    indicator = _extract_indicator(log_file)
    returncode = 0
    if indicator is None:
        env = os.environ.copy()
        env.update(BASE_ENV)
        cmd = [
            str(JUEJIN_PYTHON),
            str(MAIN / "run_juejin_signal_backtest.py"),
            "--strategy-dir",
            str(STRATEGY_DIR),
            "--signal-file",
            str(SIGNAL_FILE),
            "--log-file",
            str(log_file),
            "--max-positions",
            "1",
            "--holding-days",
            "3",
            "--max-holding-days",
            "5",
            "--target-position-pct",
            "0.51",
            "--score-db",
            str(SCORE_DB),
            "--score-table",
            SCORE_TABLE,
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
        "anchor": anchor,
        "offset": offset,
        "start_date": start.split(" ", 1)[0],
        "start_ts": start,
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


def main() -> None:
    if not SIGNAL_FILE.exists():
        raise FileNotFoundError(f"signal file not found: {SIGNAL_FILE}")
    detail: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for anchor, starts in ANCHOR_STARTS.items():
        anchor_rows = []
        for idx, start in enumerate(starts):
            row = _run(anchor, idx - 3, start)
            detail.append(row)
            anchor_rows.append(row)
            _write_rows(OUT_DIR / "lowpath_detail.csv", detail)
        annuals = [float(row["annual"]) for row in anchor_rows if row["annual"] is not None]
        sharpes = [float(row["sharpe"]) for row in anchor_rows if row["sharpe"] is not None]
        drawdowns = [float(row["max_drawdown"]) for row in anchor_rows if row["max_drawdown"] is not None]
        out = {
            "anchor": anchor,
            "starts": len(anchor_rows),
            "annual_min": min(annuals) if annuals else None,
            "annual_median": median(annuals) if annuals else None,
            "annual_max": max(annuals) if annuals else None,
            "sharpe_min": min(sharpes) if sharpes else None,
            "max_drawdown_max": max(drawdowns) if drawdowns else None,
            "pass_gt_0_all_starts": all(value > 0 for value in annuals) if annuals else False,
        }
        summary.append(out)
        _write_rows(OUT_DIR / "lowpath_summary.csv", summary)
        print(json.dumps(out, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

