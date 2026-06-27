from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import math
import os
import re
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA_DIR = ROOT / "quant" / "data_file"
SOURCE_DIR = (
    DATA_DIR
    / "reports"
    / "strategy_agent_goal_high_annual_20260623"
    / "top1_no_delist_rerun_20260624"
    / "diversification_tune_20260624"
)
REPORT_DIR = SOURCE_DIR / "strict_sync_risk_overlay_20260624"
MARKET_DB = DATA_DIR / "STOCK_DAILY_DATA.db"
SCORE_DB = SOURCE_DIR / "scores_diversification.db"
STRATEGY_DIR = SOURCE_DIR / "strict_sync_strategy_snapshot_20260624"
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")
BACKTEST_END = "2026-06-23 15:30:00"

STARTS = [
    ("full_20240605", "2024-06-05 09:00:00"),
    ("late_20250701", "2025-07-01 09:00:00"),
    ("late_20251009", "2025-10-09 09:00:00"),
    ("late_20260105", "2026-01-05 09:00:00"),
]

BASE_ENV = {
    "GM_OPEN_DAILY_SCORE_EXIT": "1",
    "GM_MAX_DAILY_SELLS": "3",
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

SOURCE_CASES = [
    {
        "source": "div_top1_w90_5d10_h5_e099",
        "holding_days": 5,
        "exit_ratio": 0.99,
        "min_hold": 1,
    },
    {
        "source": "div_top1_w90_5d10_h5_e098",
        "holding_days": 5,
        "exit_ratio": 0.98,
        "min_hold": 1,
    },
    {
        "source": "div_top1_w89_5d11_h5_e099",
        "holding_days": 5,
        "exit_ratio": 0.99,
        "min_hold": 1,
    },
    {
        "source": "div_top1_w90_5d10_h5_e098_mh2",
        "holding_days": 5,
        "exit_ratio": 0.98,
        "min_hold": 2,
    },
]

TARGET_PCTS = [0.60, 0.75, 0.90, 0.99]
STOP_LOSSES = [0.05, 0.08, 0.12]


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


def _read_signal(source: str) -> list[dict]:
    path = SOURCE_DIR / "signals" / f"{source}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _case_name(source: str, target_pct: float, stop_loss: float) -> str:
    return f"risk_{source}_pos{int(target_pct * 100):02d}_sl{int(stop_loss * 100):02d}"


def _write_signal(case: dict) -> Path:
    rows = _read_signal(case["source"])
    for row in rows:
        row["target_pct"] = f"{float(case['target_pct']):.5f}"
        row["holding_days"] = int(case["holding_days"])
        row["max_holding_days"] = int(case["holding_days"])
        row["score_exit_entry_ratio"] = f"{float(case['exit_ratio']):.5f}"
        row["min_holding_days_before_score_exit"] = int(case["min_hold"])
    path = REPORT_DIR / "signals" / f"{case['name']}.csv"
    _write_rows(path, rows)
    return path


def _score_table(source: str) -> str:
    return "score_" + re.sub(r"[^A-Za-z0-9_]+", "_", source)


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


def _run_case(case: dict, signal_file: Path, start_name: str, start: str) -> dict:
    log_file = REPORT_DIR / "logs" / f"{case['name']}_{start_name}.log"
    indicator = _extract_indicator(log_file)
    returncode = 0
    if indicator is None:
        env = os.environ.copy()
        env.update(BASE_ENV)
        env["GM_STOP_LOSS_PCT"] = str(float(case["stop_loss_pct"]))
        env["GM_SCORE_EXIT_ENTRY_RATIO"] = str(float(case["exit_ratio"]))
        env["GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT"] = str(int(case["min_hold"]))
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
            str(int(case["holding_days"])),
            "--max-holding-days",
            str(int(case["holding_days"])),
            "--target-position-pct",
            str(float(case["target_pct"])),
            "--score-db",
            str(SCORE_DB),
            "--score-table",
            _score_table(case["source"]),
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
        returncode = proc.returncode
        indicator = _extract_indicator(log_file)
    return {
        "name": case["name"],
        "source": case["source"],
        "target_pct": case["target_pct"],
        "stop_loss_pct": case["stop_loss_pct"],
        "start_name": start_name,
        "returncode": returncode,
        "annual": indicator.get("pnl_ratio_annual") if indicator else None,
        "pnl_ratio": indicator.get("pnl_ratio") if indicator else None,
        "sharpe": indicator.get("sharp_ratio") if indicator else None,
        "max_drawdown": indicator.get("max_drawdown") if indicator else None,
        "win_ratio": indicator.get("win_ratio") if indicator else None,
        "open_count": indicator.get("open_count") if indicator else None,
        "close_count": indicator.get("close_count") if indicator else None,
        "config_json": json.dumps(case, ensure_ascii=False, sort_keys=True),
        "signal_file": str(signal_file),
        "log_file": str(log_file),
        **_exposure_stats(log_file),
    }


def _metric(row: dict, key: str) -> float:
    try:
        value = float(row.get(key))
    except (TypeError, ValueError):
        return float("-inf")
    if math.isnan(value) or math.isinf(value):
        return float("-inf")
    return value


def _summarize(rows: list[dict]) -> list[dict]:
    out = []
    for name in sorted({row["name"] for row in rows}):
        items = [row for row in rows if row["name"] == name]
        full = next((row for row in items if row["start_name"] == "full_20240605"), None)
        if not full or full.get("annual") is None:
            continue
        late = [float(row["annual"]) for row in items if row["start_name"].startswith("late_") and row.get("annual") is not None]
        out.append(
            {
                "name": name,
                "source": full.get("source"),
                "target_pct": full.get("target_pct"),
                "stop_loss_pct": full.get("stop_loss_pct"),
                "full_annual": full.get("annual"),
                "full_sharpe": full.get("sharpe"),
                "full_max_drawdown": full.get("max_drawdown"),
                "full_open_count": full.get("open_count"),
                "full_close_count": full.get("close_count"),
                "late_min_annual": min(late) if late else None,
                "late_median_annual": float(pd.Series(late).median()) if late else None,
                "late_max_annual": max(late) if late else None,
                "avg_invested_pct": full.get("avg_invested_pct"),
                "ge80_ratio": full.get("ge80_ratio"),
                "max_active_positions": full.get("max_active_positions"),
                "score": (
                    float(full["annual"]) * 0.20
                    + (float(full.get("sharpe") or 0.0) * 0.70)
                    - (float(full.get("max_drawdown") or 0.0) * 2.5)
                    + ((min(late) if late else 0.0) * 0.35)
                    + ((float(pd.Series(late).median()) if late else 0.0) * 0.30)
                ),
            }
        )
    out.sort(key=lambda row: (row["score"], row["full_sharpe"]), reverse=True)
    return out


def main() -> int:
    if not STRATEGY_DIR.joinpath("main.py").exists():
        raise SystemExit(f"strict sync strategy main.py not found: {STRATEGY_DIR}")
    cases = []
    for source in SOURCE_CASES:
        for target_pct in TARGET_PCTS:
            for stop_loss in STOP_LOSSES:
                case = dict(source)
                case["target_pct"] = target_pct
                case["stop_loss_pct"] = stop_loss
                case["name"] = _case_name(source["source"], target_pct, stop_loss)
                cases.append(case)
    rows = []
    total = len(cases) * len(STARTS)
    done = 0
    for case in cases:
        signal_file = _write_signal(case)
        for start_name, start in STARTS:
            done += 1
            row = _run_case(case, signal_file, start_name, start)
            rows.append(row)
            _write_rows(REPORT_DIR / "cases.csv", rows)
            print(
                f"[{done}/{total}] {case['name']} {start_name} "
                f"annual={row['annual']} sharpe={row['sharpe']} maxdd={row['max_drawdown']}",
                flush=True,
            )
    summary = _summarize(rows)
    _write_rows(REPORT_DIR / "summary.csv", summary)
    _write_rows(REPORT_DIR / "summary_by_annual.csv", sorted(summary, key=lambda row: _metric(row, "full_annual"), reverse=True))
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
