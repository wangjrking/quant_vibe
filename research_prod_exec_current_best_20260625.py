from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import os
import re
import subprocess
from pathlib import Path


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA = ROOT / "quant" / "data_file"
REPORT_DIR = DATA / "reports" / "strategy_agent_prod_exec_current_best_20260625"
STRATEGY_DIR = MAIN / "strategy_library" / "production" / "prod_dynamic_h2m3_pos8975_scale7555_v20260625" / "code_snapshot"
SIGNAL_ROOT = (
    DATA
    / "reports"
    / "strategy_agent_goal_high_annual_20260623"
    / "top1_no_delist_rerun_20260624"
    / "diversification_tune_20260624"
    / "strict_sync_liquidity_neighborhood_20260624"
    / "formal_horizon_entry_confirmation_20260624"
    / "dynamic_h2_m3_sl06_dd0712_pos89_c975_high_pos_boundary_20260625"
    / "signals"
)
SCORE_DB = (
    DATA
    / "reports"
    / "strategy_agent_goal_high_annual_20260623"
    / "top1_no_delist_rerun_20260624"
    / "diversification_tune_20260624"
    / "scores_diversification.db"
)
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
    "GM_EQUITY_DD_RESIZE_EXISTING": "0",
}

CASES = [
    {
        "name": "pos90_base",
        "signal_file": SIGNAL_ROOT / "pos90_scale7454.csv",
        "dd_soft": 0.07,
        "dd_hard": 0.115,
        "dd_soft_scale": 0.74,
        "dd_hard_scale": 0.54,
        "target_position_pct": 0.90,
        "extra_env": {},
    },
    {
        "name": "pos90_replace1",
        "signal_file": SIGNAL_ROOT / "pos90_scale7454.csv",
        "dd_soft": 0.07,
        "dd_hard": 0.115,
        "dd_soft_scale": 0.74,
        "dd_hard_scale": 0.54,
        "target_position_pct": 0.90,
        "extra_env": {"GM_INTRADAY_REPLACE_BUY": "1", "GM_INTRADAY_REPLACE_MAX_BUYS": "1"},
    },
    {
        "name": "pos90_replace2",
        "signal_file": SIGNAL_ROOT / "pos90_scale7454.csv",
        "dd_soft": 0.07,
        "dd_hard": 0.115,
        "dd_soft_scale": 0.74,
        "dd_hard_scale": 0.54,
        "target_position_pct": 0.90,
        "extra_env": {"GM_INTRADAY_REPLACE_BUY": "1", "GM_INTRADAY_REPLACE_MAX_BUYS": "2"},
    },
    {
        "name": "pos90_resize105",
        "signal_file": SIGNAL_ROOT / "pos90_scale7454.csv",
        "dd_soft": 0.07,
        "dd_hard": 0.115,
        "dd_soft_scale": 0.74,
        "dd_hard_scale": 0.54,
        "target_position_pct": 0.90,
        "extra_env": {
            "GM_RESIZE_HELD_ON_SIGNAL": "1",
            "GM_RESIZE_HELD_TARGET_MULT": "1.05",
            "GM_RESIZE_HELD_MIN_DELTA_PCT": "0.015",
        },
    },
    {
        "name": "pos8975_base",
        "signal_file": SIGNAL_ROOT / "pos8975_scale7555.csv",
        "dd_soft": 0.07,
        "dd_hard": 0.115,
        "dd_soft_scale": 0.75,
        "dd_hard_scale": 0.55,
        "target_position_pct": 0.8975,
        "extra_env": {},
    },
    {
        "name": "pos8975_replace1",
        "signal_file": SIGNAL_ROOT / "pos8975_scale7555.csv",
        "dd_soft": 0.07,
        "dd_hard": 0.115,
        "dd_soft_scale": 0.75,
        "dd_hard_scale": 0.55,
        "target_position_pct": 0.8975,
        "extra_env": {"GM_INTRADAY_REPLACE_BUY": "1", "GM_INTRADAY_REPLACE_MAX_BUYS": "1"},
    },
    {
        "name": "pos8975_replace2",
        "signal_file": SIGNAL_ROOT / "pos8975_scale7555.csv",
        "dd_soft": 0.07,
        "dd_hard": 0.115,
        "dd_soft_scale": 0.75,
        "dd_hard_scale": 0.55,
        "target_position_pct": 0.8975,
        "extra_env": {"GM_INTRADAY_REPLACE_BUY": "1", "GM_INTRADAY_REPLACE_MAX_BUYS": "2"},
    },
]

SLICES = [
    ("full", "2024-06-05 09:00:00", "2026-06-23 15:30:00"),
    ("recent120", "2025-12-24 09:00:00", "2026-06-23 15:30:00"),
    ("recent60", "2026-03-24 09:00:00", "2026-06-23 15:30:00"),
    ("late20250701", "2025-07-01 09:00:00", "2026-06-23 15:30:00"),
    ("late20251009", "2025-10-09 09:00:00", "2026-06-23 15:30:00"),
    ("late20260105", "2026-01-05 09:00:00", "2026-06-23 15:30:00"),
]


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
    pattern = re.compile(r"EXPOSURE .* invested_pct=([0-9.]+).*active_positions=(\d+)")
    values: list[float] = []
    actives: list[int] = []
    if not log_file.exists():
        return {"avg_invested_pct": None, "max_active_positions": None, "exposure_points": 0}
    for match in pattern.finditer(log_file.read_text(encoding="utf-8", errors="ignore")):
        values.append(float(match.group(1)))
        actives.append(int(match.group(2)))
    if not values:
        return {"avg_invested_pct": None, "max_active_positions": None, "exposure_points": 0}
    return {
        "avg_invested_pct": sum(values) / len(values),
        "max_active_positions": max(actives) if actives else None,
        "exposure_points": len(values),
    }


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


def _run_case(case: dict, slice_name: str, start: str, end: str) -> dict:
    env = os.environ.copy()
    env.update(BASE_ENV)
    env["GM_EQUITY_DD_SOFT_TRIGGER"] = str(case["dd_soft"])
    env["GM_EQUITY_DD_HARD_TRIGGER"] = str(case["dd_hard"])
    env["GM_EQUITY_DD_RECOVER_TRIGGER"] = "0.03"
    env["GM_EQUITY_DD_SOFT_SCALE"] = str(case["dd_soft_scale"])
    env["GM_EQUITY_DD_HARD_SCALE"] = str(case["dd_hard_scale"])
    env.update(case["extra_env"])
    log_file = REPORT_DIR / "logs" / f"{slice_name}_{case['name']}.log"
    command = [
        str(JUEJIN_PYTHON),
        str(MAIN / "run_juejin_signal_backtest.py"),
        "--strategy-dir",
        str(STRATEGY_DIR),
        "--signal-file",
        str(case["signal_file"]),
        "--log-file",
        str(log_file),
        "--max-positions",
        "1",
        "--holding-days",
        "2",
        "--max-holding-days",
        "3",
        "--target-position-pct",
        str(case["target_position_pct"]),
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
    ]
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as log:
        proc = subprocess.run(command, cwd=str(MAIN), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    indicator = _extract_indicator(log_file)
    exposure = _exposure_stats(log_file)
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
        "close_count": indicator.get("close_count") if indicator else None,
        "avg_invested_pct": exposure["avg_invested_pct"],
        "max_active_positions": exposure["max_active_positions"],
        "extra_env": json.dumps(case["extra_env"], ensure_ascii=False, sort_keys=True),
        "signal_file": str(case["signal_file"]),
        "log_file": str(log_file),
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    detail_rows: list[dict] = []
    summary_rows: list[dict] = []
    for case in CASES:
        case_rows: list[dict] = []
        for slice_name, start, end in SLICES:
            row = _run_case(case, slice_name, start, end)
            detail_rows.append(row)
            case_rows.append(row)
            print(
                json.dumps(
                    {
                        "case": case["name"],
                        "slice": slice_name,
                        "annual": row["annual"],
                        "sharpe": row["sharpe"],
                        "max_drawdown": row["max_drawdown"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        by_slice = {row["slice"]: row for row in case_rows}
        summary_rows.append(
            {
                "case_name": case["name"],
                "signal_file": str(case["signal_file"]),
                "extra_env": json.dumps(case["extra_env"], ensure_ascii=False, sort_keys=True),
                "full_annual": by_slice["full"]["annual"],
                "full_sharpe": by_slice["full"]["sharpe"],
                "full_max_drawdown": by_slice["full"]["max_drawdown"],
                "recent120_annual": by_slice["recent120"]["annual"],
                "recent60_annual": by_slice["recent60"]["annual"],
                "late20250701_annual": by_slice["late20250701"]["annual"],
                "late20251009_annual": by_slice["late20251009"]["annual"],
                "late20260105_annual": by_slice["late20260105"]["annual"],
                "avg_invested_pct_full": by_slice["full"]["avg_invested_pct"],
            }
        )
    _write_rows(REPORT_DIR / "detail.csv", detail_rows)
    _write_rows(REPORT_DIR / "summary.csv", summary_rows)
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report_dir": str(REPORT_DIR), "cases": len(summary_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
