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
DATA = ROOT / "quant" / "data_file"
REPORT_DIR = DATA / "reports" / "strategy_agent_amt8w_execution_rescue_20260625"
BASE_SIGNAL = DATA / "reports" / "strategy_agent_prod_filter_narrow_grid_20260625" / "signals" / "prod_filter_amt8w_mv20w.csv"
STRATEGY_DIR = MAIN / "strategy_library" / "production" / "prod_dynamic_h2m3_pos8975_scale7555_v20260625" / "code_snapshot"
SCORE_DB = DATA / "reports" / "strategy_agent_goal_high_annual_20260623" / "top1_no_delist_rerun_20260624" / "diversification_tune_20260624" / "scores_diversification.db"
SCORE_TABLE = "score_div_top1_w90_5d10_h5_e099"
MARKET_DB = DATA / "STOCK_DAILY_DATA.db"
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")

BASE_ENV = {
    "GM_OPEN_DAILY_SCORE_EXIT": "1",
    "GM_MAX_DAILY_SELLS": "1",
    "GM_LIGHT_STOP_LOSS_PCT": "none",
    "GM_LOG_EXPOSURE": "1",
    "GM_SKIP_OPEN_LIMIT_UP_BUY": "1",
    "GM_SYNC_POSITIONS": "1",
    "GM_CASH_BUFFER": "0.99",
    "GM_VERBOSE_TRADES": "0",
    "GM_FORCE_SELL_MARKET_ORDER": "1",
    "GM_INTRADAY_RISK_MODE": "1",
    "GM_INTRADAY_REPLACE_BUY": "0",
    "GM_INTRADAY_RISK_TIMES": "10:00:00,11:00:00,14:30:00",
    "GM_EQUITY_DD_RISK_MODE": "1",
    "GM_EQUITY_DD_SOFT_TRIGGER": "0.07",
    "GM_EQUITY_DD_RECOVER_TRIGGER": "0.03",
    "GM_EQUITY_DD_SOFT_SCALE": "0.75",
    "GM_EQUITY_DD_HARD_SCALE": "0.55",
    "GM_EQUITY_DD_RESIZE_EXISTING": "0",
    "GM_SCORE_EXIT_ENTRY_RATIO": "0.97",
    "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "1",
}

CASES = [
    {"name": "amt8w_base", "continue_ratio": 0.975, "dd_hard": 0.115},
    {"name": "amt8w_c098", "continue_ratio": 0.98, "dd_hard": 0.115},
    {"name": "amt8w_c097", "continue_ratio": 0.97, "dd_hard": 0.115},
    {"name": "amt8w_dh12", "continue_ratio": 0.975, "dd_hard": 0.12},
]

SLICES = [
    ("full", "2024-06-05 09:00:00", "2026-06-23 15:30:00"),
    ("recent120", "2025-12-24 09:00:00", "2026-06-23 15:30:00"),
    ("recent60", "2026-03-24 09:00:00", "2026-06-23 15:30:00"),
    ("late20250701", "2025-07-01 09:00:00", "2026-06-23 15:30:00"),
    ("late20251009", "2025-10-09 09:00:00", "2026-06-23 15:30:00"),
    ("late20260105", "2026-01-05 09:00:00", "2026-06-23 15:30:00"),
]


def _read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


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
        writer.writerows([{key: row.get(key, "") for key in fields} for row in rows])


def _signal_for_case(case: dict) -> Path:
    path = REPORT_DIR / "signals" / f"{case['name']}.csv"
    rows = _read_rows(BASE_SIGNAL)
    for row in rows:
        row["score_continue_entry_ratio"] = f"{case['continue_ratio']:.5f}"
        row["strategy_variant"] = case["name"]
    _write_rows(path, rows)
    return path


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


def _run(case: dict, signal_file: Path, slice_name: str, start: str, end: str) -> dict:
    log_file = REPORT_DIR / "logs" / f"{slice_name}_{case['name']}.log"
    env = os.environ.copy()
    env.update(BASE_ENV)
    env["GM_SCORE_CONTINUE_ENTRY_RATIO"] = str(case["continue_ratio"])
    env["GM_EQUITY_DD_HARD_TRIGGER"] = str(case["dd_hard"])
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
        "1",
        "--holding-days",
        "2",
        "--max-holding-days",
        "3",
        "--target-position-pct",
        "0.8975",
        "--score-db",
        str(SCORE_DB),
        "--score-table",
        SCORE_TABLE,
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
        "0.0015",
        "--stop-loss-pct",
        "0.06",
        "--take-profit-pct",
        "0.07",
    ]
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as log:
        proc = subprocess.run(command, cwd=str(MAIN), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    indicator = _extract_indicator(log_file)
    return {
        "case_name": case["name"],
        "slice": slice_name,
        "start": start,
        "returncode": proc.returncode,
        "annual": indicator.get("pnl_ratio_annual") if indicator else None,
        "pnl_ratio": indicator.get("pnl_ratio") if indicator else None,
        "sharpe": indicator.get("sharp_ratio") if indicator else None,
        "max_drawdown": indicator.get("max_drawdown") if indicator else None,
        "win_ratio": indicator.get("win_ratio") if indicator else None,
        "open_count": indicator.get("open_count") if indicator else None,
        "continue_ratio": case["continue_ratio"],
        "dd_hard": case["dd_hard"],
        "log_file": str(log_file),
        "signal_file": str(signal_file),
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    summary: list[dict] = []
    for case in CASES:
        signal_file = _signal_for_case(case)
        case_rows = []
        for slice_name, start, end in SLICES:
            row = _run(case, signal_file, slice_name, start, end)
            rows.append(row)
            case_rows.append(row)
        by_slice = {row["slice"]: row for row in case_rows}
        summary.append(
            {
                "case_name": case["name"],
                "continue_ratio": case["continue_ratio"],
                "dd_hard": case["dd_hard"],
                "full_annual": by_slice["full"]["annual"],
                "full_sharpe": by_slice["full"]["sharpe"],
                "full_max_drawdown": by_slice["full"]["max_drawdown"],
                "recent120_annual": by_slice["recent120"]["annual"],
                "recent60_annual": by_slice["recent60"]["annual"],
                "late20250701_annual": by_slice["late20250701"]["annual"],
                "late20251009_annual": by_slice["late20251009"]["annual"],
                "late20260105_annual": by_slice["late20260105"]["annual"],
            }
        )
    _write_rows(REPORT_DIR / "detail.csv", rows)
    _write_rows(REPORT_DIR / "summary.csv", summary)
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report_dir": str(REPORT_DIR), "cases": len(summary)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
