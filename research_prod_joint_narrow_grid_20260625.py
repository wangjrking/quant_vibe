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
REPORT_DIR = DATA / "reports" / "strategy_agent_prod_joint_narrow_grid_20260625"
STRATEGY_DIR = MAIN / "strategy_library" / "production" / "prod_dynamic_h2m3_pos8975_scale7555_v20260625" / "code_snapshot"
SIGNAL_FILE = DATA / "reports" / "strategy_agent_goal_high_annual_20260623" / "top1_no_delist_rerun_20260624" / "diversification_tune_20260624" / "strict_sync_liquidity_neighborhood_20260624" / "formal_horizon_entry_confirmation_20260624" / "dynamic_h2_m3_sl06_dd0712_pos89_c975_high_pos_boundary_20260625" / "signals" / "pos8975_scale7555.csv"
SCORE_DB = DATA / "reports" / "strategy_agent_goal_high_annual_20260623" / "top1_no_delist_rerun_20260624" / "diversification_tune_20260624" / "scores_diversification.db"
SCORE_TABLE = "score_div_top1_w90_5d10_h5_e099"
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
    "GM_SCORE_EXIT_ENTRY_RATIO": "0.97",
    "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "1",
}

CASES = [
    {"name": "baseline_prod", "target_pct": 0.8975, "continue_ratio": 0.975, "stop_loss": 0.06, "take_profit": 0.07, "dd_soft": 0.07, "dd_hard": 0.12, "dd_soft_scale": 0.75, "dd_hard_scale": 0.55},
    {"name": "dh115", "target_pct": 0.8975, "continue_ratio": 0.975, "stop_loss": 0.06, "take_profit": 0.07, "dd_soft": 0.07, "dd_hard": 0.115, "dd_soft_scale": 0.75, "dd_hard_scale": 0.55},
    {"name": "dh115_c096", "target_pct": 0.8975, "continue_ratio": 0.96, "stop_loss": 0.06, "take_profit": 0.07, "dd_soft": 0.07, "dd_hard": 0.115, "dd_soft_scale": 0.75, "dd_hard_scale": 0.55},
    {"name": "dh115_c096_sl05", "target_pct": 0.8975, "continue_ratio": 0.96, "stop_loss": 0.05, "take_profit": 0.07, "dd_soft": 0.07, "dd_hard": 0.115, "dd_soft_scale": 0.75, "dd_hard_scale": 0.55},
    {"name": "dh115_c097_sl05", "target_pct": 0.8975, "continue_ratio": 0.97, "stop_loss": 0.05, "take_profit": 0.07, "dd_soft": 0.07, "dd_hard": 0.115, "dd_soft_scale": 0.75, "dd_hard_scale": 0.55},
    {"name": "dh115_c096_pos90", "target_pct": 0.90, "continue_ratio": 0.96, "stop_loss": 0.06, "take_profit": 0.07, "dd_soft": 0.07, "dd_hard": 0.115, "dd_soft_scale": 0.75, "dd_hard_scale": 0.55},
]

SLICES = [
    ("full", "2024-06-05 09:00:00", "2026-06-23 15:30:00"),
    ("recent120", "2025-12-24 09:00:00", "2026-06-23 15:30:00"),
    ("recent60", "2026-03-24 09:00:00", "2026-06-23 15:30:00"),
    ("ytd2026", "2026-01-05 09:00:00", "2026-06-23 15:30:00"),
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


def _run_case(case: dict, slice_name: str, start: str, end: str) -> dict:
    env = os.environ.copy()
    env.update(BASE_ENV)
    env["GM_SCORE_CONTINUE_ENTRY_RATIO"] = str(case["continue_ratio"])
    env["GM_STOP_LOSS_PCT"] = str(case["stop_loss"])
    env["GM_TAKE_PROFIT_PCT"] = str(case["take_profit"])
    env["GM_EQUITY_DD_SOFT_TRIGGER"] = str(case["dd_soft"])
    env["GM_EQUITY_DD_HARD_TRIGGER"] = str(case["dd_hard"])
    env["GM_EQUITY_DD_RECOVER_TRIGGER"] = "0.03"
    env["GM_EQUITY_DD_SOFT_SCALE"] = str(case["dd_soft_scale"])
    env["GM_EQUITY_DD_HARD_SCALE"] = str(case["dd_hard_scale"])
    log_file = REPORT_DIR / "logs" / f"{slice_name}_{case['name']}.log"
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
        "1",
        "--holding-days",
        "2",
        "--max-holding-days",
        "3",
        "--target-position-pct",
        str(case["target_pct"]),
        "--score-db",
        str(SCORE_DB),
        "--score-table",
        SCORE_TABLE,
        "--market-db",
        str(DATA / "STOCK_DAILY_DATA.db"),
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
        str(case["stop_loss"]),
        "--take-profit-pct",
        str(case["take_profit"]),
    ]
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as log:
        proc = subprocess.run(command, cwd=str(MAIN), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    indicator = _extract_indicator(log_file)
    exposure = _exposure_stats(log_file)
    return {
        "tag": f"{slice_name}_{case['name']}",
        "case_name": case["name"],
        "slice": slice_name,
        "start": start,
        "end": end,
        "returncode": proc.returncode,
        "annual": indicator.get("pnl_ratio_annual") if indicator else None,
        "pnl_ratio": indicator.get("pnl_ratio") if indicator else None,
        "sharpe": indicator.get("sharp_ratio") if indicator else None,
        "max_drawdown": indicator.get("max_drawdown") if indicator else None,
        "win_ratio": indicator.get("win_ratio") if indicator else None,
        "open_count": indicator.get("open_count") if indicator else None,
        "close_count": indicator.get("close_count") if indicator else None,
        "avg_invested_pct": exposure["avg_invested_pct"],
        "target_pct": case["target_pct"],
        "score_continue_entry_ratio": case["continue_ratio"],
        "stop_loss": case["stop_loss"],
        "take_profit": case["take_profit"],
        "dd_soft": case["dd_soft"],
        "dd_hard": case["dd_hard"],
        "dd_soft_scale": case["dd_soft_scale"],
        "dd_hard_scale": case["dd_hard_scale"],
        "log_file": str(log_file),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for case in CASES:
        for slice_name, start, end in SLICES:
            rows.append(_run_case(case, slice_name, start, end))
    _write_csv(REPORT_DIR / "full_grid.csv", rows)
    summary = []
    for case in CASES:
        case_rows = {row["slice"]: row for row in rows if row["case_name"] == case["name"]}
        full = case_rows["full"]
        summary.append(
            {
                "case_name": case["name"],
                "target_pct": case["target_pct"],
                "continue_ratio": case["continue_ratio"],
                "stop_loss": case["stop_loss"],
                "take_profit": case["take_profit"],
                "dd_hard": case["dd_hard"],
                "full_annual": full["annual"],
                "full_sharpe": full["sharpe"],
                "full_max_drawdown": full["max_drawdown"],
                "recent120_annual": case_rows["recent120"]["annual"],
                "recent60_annual": case_rows["recent60"]["annual"],
                "ytd_annual": case_rows["ytd2026"]["annual"],
            }
        )
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report_dir": str(REPORT_DIR), "cases": len(summary)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
