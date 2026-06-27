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
OUT_DIR = REPORT_ROOT / "current_formal_top1keep_tailpow_startup_rules"
MARKET_DB = DATA_DIR / "STOCK_DAILY_DATA.db"
STRATEGY_DIR = Path(r"D:\work\dfcf\juejin\strategy\ce327750-634e-11f1-b8d7-10ffe0295517")
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")

BASE_SIGNAL = REPORT_ROOT / "current_formal_top1keep_tailpow_peak" / "signals" / "tp_peak_g1p38_o0p942.csv"
SCORE_DB = REPORT_ROOT / "current_formal_top1keep_tailpow_peak" / "scores_top1keep_tailpow_peak.db"
SCORE_TABLE = "score_tp_peak_g1p38_o0p942"
BACKTEST_END = "2026-06-23 15:30:00"

STARTS = [
    ("start_20250701", "2025-07-01 09:00:00", "20250701"),
    ("start_20251008", "2025-10-08 09:00:00", "20251008"),
    ("start_20260105", "2026-01-05 09:00:00", "20260105"),
]

RULES = [
    {"name": "base", "startup_days": 0, "top_k": 4, "skip_days": 0, "mode": "base"},
    {"name": "delay5", "startup_days": 40, "top_k": 4, "skip_days": 5, "mode": "base"},
    {"name": "delay10", "startup_days": 40, "top_k": 4, "skip_days": 10, "mode": "base"},
    {"name": "top1_20_t50", "startup_days": 20, "top_k": 1, "skip_days": 0, "mode": "fixed", "target": 0.50},
    {"name": "top1_40_t50", "startup_days": 40, "top_k": 1, "skip_days": 0, "mode": "fixed", "target": 0.50},
    {"name": "top2_20_t50", "startup_days": 20, "top_k": 2, "skip_days": 0, "mode": "fixed", "target": 0.50},
    {"name": "top2_40_t50", "startup_days": 40, "top_k": 2, "skip_days": 0, "mode": "fixed", "target": 0.50},
    {"name": "ramp40", "startup_days": 40, "top_k": 4, "skip_days": 0, "mode": "ramp"},
    {"name": "concentrate40", "startup_days": 40, "top_k": 4, "skip_days": 0, "mode": "concentrate"},
    {"name": "delay5_top2_40_t50", "startup_days": 40, "top_k": 2, "skip_days": 5, "mode": "fixed", "target": 0.50},
]

BASE_ENV = {
    "GM_OPEN_DAILY_SCORE_EXIT": "1",
    "GM_SCORE_EXIT_ENTRY_RATIO": "0.942",
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


def _parse_date(value: str) -> datetime_module.datetime:
    return datetime_module.datetime.strptime(value[:10], "%Y-%m-%d")


def _load_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader.fieldnames or []), list(reader)


def _write_rows(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fields} for row in rows])


def _startup_buy_dates(rows: list[dict[str, str]], start_yyyymmdd: str, count: int) -> list[str]:
    dates = sorted({str(row.get("buy_date") or "") for row in rows if str(row.get("buy_date") or "") >= start_yyyymmdd})
    return dates[:count]


def _adjust_target(row: dict[str, str], mode: str, index: int, startup_days: int) -> None:
    rank = int(float(row.get("rank") or 999999))
    if mode == "base":
        return
    if mode == "fixed":
        row["target_pct"] = f"{float(row.get('_target') or 0.50):.5f}"
        return
    if mode == "ramp":
        scale = min(1.0, max(0.35, (index + 1) / max(float(startup_days), 1.0)))
        current = float(row.get("target_pct") or 0.0)
        row["target_pct"] = f"{current * scale:.5f}"
        return
    if mode == "concentrate":
        targets = {1: 0.50, 2: 0.28, 3: 0.16, 4: 0.08}
        row["target_pct"] = f"{targets.get(rank, 0.0):.5f}"


def _build_signal_for_rule(fields: list[str], rows: list[dict[str, str]], start_tag: str, start_yyyymmdd: str, rule: dict) -> Path:
    startup_days = int(rule["startup_days"])
    startup_dates = _startup_buy_dates(rows, start_yyyymmdd, max(startup_days, int(rule["skip_days"])))
    startup_index = {date: index for index, date in enumerate(startup_dates)}
    skip_dates = set(startup_dates[: int(rule["skip_days"])])
    adjusted = []
    for raw in rows:
        row = dict(raw)
        buy_date = str(row.get("buy_date") or "")
        rank = int(float(row.get("rank") or 999999))
        if buy_date in skip_dates:
            continue
        if buy_date in startup_index and startup_index[buy_date] < startup_days:
            if rank > int(rule["top_k"]):
                continue
            if "target" in rule:
                row["_target"] = str(rule["target"])
            _adjust_target(row, str(rule["mode"]), startup_index[buy_date], startup_days)
        adjusted.append(row)
    signal_path = OUT_DIR / "signals" / f"{start_tag}_{rule['name']}.csv"
    _write_rows(signal_path, fields, adjusted)
    return signal_path


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


def _run_case(start_tag: str, backtest_start: str, rule: dict, signal_file: Path) -> dict:
    log_file = OUT_DIR / "logs" / f"{start_tag}_{rule['name']}.log"
    indicator = _extract_indicator(log_file)
    if indicator is None:
        env = os.environ.copy()
        env.update(BASE_ENV)
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
            "4",
            "--holding-days",
            "5",
            "--max-holding-days",
            "5",
            "--target-position-pct",
            "0.5",
            "--score-db",
            str(SCORE_DB),
            "--score-table",
            SCORE_TABLE,
            "--market-db",
            str(MARKET_DB),
            "--backtest-start",
            backtest_start,
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
        "start_tag": start_tag,
        "rule": rule["name"],
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
    }


def main() -> int:
    fields, rows = _load_rows(BASE_SIGNAL)
    results = []
    for start_tag, backtest_start, start_yyyymmdd in STARTS:
        for rule in RULES:
            signal_file = _build_signal_for_rule(fields, rows, start_tag, start_yyyymmdd, rule)
            result = _run_case(start_tag, backtest_start, rule, signal_file)
            results.append(result)
            print(
                f"{start_tag} {rule['name']} annual={result['annual']} "
                f"pnl={result['pnl_ratio']} sharpe={result['sharpe']} maxdd={result['max_drawdown']}",
                flush=True,
            )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = OUT_DIR / "startup_rules_summary.csv"
    with summary.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    (OUT_DIR / "startup_rules_summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
