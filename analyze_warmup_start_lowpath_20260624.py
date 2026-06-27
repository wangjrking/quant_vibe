from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import math
import os
import re
import sqlite3
import subprocess
from pathlib import Path


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
REPORT_DIR = SOURCE_DIR / "warmup_start_lowpath_20260624"
MARKET_DB = DATA_DIR / "STOCK_DAILY_DATA.db"
SCORE_DB = SOURCE_DIR / "scores_diversification.db"
STRATEGY_DIR = Path(r"D:\work\dfcf\juejin\strategy\ce327750-634e-11f1-b8d7-10ffe0295517")
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")
BACKTEST_END = "2026-06-23 15:30:00"

STRATEGIES = [
    {
        "name": "div_top1_w90_5d10_h5_e099",
        "target_pct": 0.99,
        "max_positions": 1,
        "holding_days": 5,
        "exit_ratio": 0.99,
        "min_hold": 1,
    },
    {
        "name": "div_top1_w90_5d10_h5_e099_pos93",
        "target_pct": 0.93,
        "max_positions": 1,
        "holding_days": 5,
        "exit_ratio": 0.99,
        "min_hold": 1,
    },
    {
        "name": "div_top1_w90_5d10_h5_e098",
        "target_pct": 0.99,
        "max_positions": 1,
        "holding_days": 5,
        "exit_ratio": 0.98,
        "min_hold": 1,
    },
]

ANCHORS = ["20250701", "20251009", "20260105"]
WARMUP_DAYS = [20, 40, 60]

BASE_ENV = {
    "GM_OPEN_DAILY_SCORE_EXIT": "1",
    "GM_MAX_DAILY_SELLS": "1",
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


def _market_dates() -> list[str]:
    conn = sqlite3.connect(MARKET_DB)
    try:
        return [str(row[0]) for row in conn.execute("SELECT DISTINCT trade_date FROM STOCK_DAILY_DATA ORDER BY trade_date")]
    finally:
        conn.close()


def _warmup_start(anchor: str, warmup_days: int, dates: list[str]) -> str:
    if anchor not in dates:
        raise RuntimeError(f"anchor trade date not found: {anchor}")
    index = dates.index(anchor)
    start_index = max(index - warmup_days, 0)
    return dates[start_index]


def _score_table(strategy: str) -> str:
    return "score_" + re.sub(r"[^A-Za-z0-9_]+", "_", strategy)


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


def _parse_nav(log_file: Path) -> list[dict]:
    pattern = re.compile(
        r"EXPOSURE\s+(?P<date>\d{8})\s+post_buy\s+invested_pct=(?P<invested>[0-9.]+)"
        r".*active_positions=(?P<active>\d+).*nav=(?P<nav>[0-9.]+)"
    )
    rows = []
    for line in log_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        rows.append(
            {
                "date": match.group("date"),
                "nav": float(match.group("nav")),
                "invested_pct": float(match.group("invested")),
                "active_positions": int(match.group("active")),
            }
        )
    rows.sort(key=lambda row: row["date"])
    return rows


def _max_drawdown(values: list[float]) -> float:
    peak = None
    out = 0.0
    for value in values:
        peak = value if peak is None else max(peak, value)
        if peak and peak > 0:
            out = max(out, 1.0 - value / peak)
    return out


def _nav_metrics(rows: list[dict]) -> dict:
    if len(rows) < 2:
        return {
            "period_days": len(rows),
            "annual": None,
            "pnl": None,
            "sharpe": None,
            "max_drawdown": None,
            "avg_invested_pct": None,
            "max_active_positions": None,
        }
    start = rows[0]["nav"]
    end = rows[-1]["nav"]
    returns = []
    for prev, curr in zip(rows, rows[1:]):
        if prev["nav"] > 0:
            returns.append(curr["nav"] / prev["nav"] - 1.0)
    mean = sum(returns) / len(returns) if returns else 0.0
    variance = sum((value - mean) ** 2 for value in returns) / (len(returns) - 1) if len(returns) > 1 else 0.0
    std = math.sqrt(variance)
    return {
        "period_days": len(rows),
        "annual": (end / start) ** (252.0 / max(len(rows) - 1, 1)) - 1.0 if start > 0 else None,
        "pnl": end / start - 1.0 if start > 0 else None,
        "sharpe": mean / std * math.sqrt(252.0) if std > 0 else None,
        "max_drawdown": _max_drawdown([row["nav"] for row in rows]),
        "avg_invested_pct": sum(row["invested_pct"] for row in rows) / len(rows),
        "max_active_positions": max(row["active_positions"] for row in rows),
    }


def _run_case(strategy: dict, anchor: str, warmup_days: int, start_date: str) -> dict:
    name = strategy["name"]
    log_file = REPORT_DIR / "logs" / f"{name}_anchor{anchor}_warm{warmup_days}.log"
    indicator = _extract_indicator(log_file)
    returncode = 0
    if indicator is None:
        env = os.environ.copy()
        env.update(BASE_ENV)
        env["GM_SCORE_EXIT_ENTRY_RATIO"] = str(float(strategy["exit_ratio"]))
        env["GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT"] = str(int(strategy["min_hold"]))
        command = [
            str(JUEJIN_PYTHON),
            str(MAIN / "run_juejin_signal_backtest.py"),
            "--strategy-dir",
            str(STRATEGY_DIR),
            "--signal-file",
            str(SOURCE_DIR / "signals" / f"{name}.csv"),
            "--log-file",
            str(log_file),
            "--max-positions",
            str(int(strategy["max_positions"])),
            "--holding-days",
            str(int(strategy["holding_days"])),
            "--max-holding-days",
            str(int(strategy["holding_days"])),
            "--target-position-pct",
            str(float(strategy["target_pct"])),
            "--score-db",
            str(SCORE_DB),
            "--score-table",
            _score_table(name),
            "--market-db",
            str(MARKET_DB),
            "--backtest-start",
            f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]} 09:00:00",
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
    nav = _parse_nav(log_file)
    sliced = [row for row in nav if row["date"] >= anchor]
    metrics = _nav_metrics(sliced)
    return {
        "strategy": name,
        "anchor": anchor,
        "warmup_days": warmup_days,
        "warmup_start": start_date,
        "returncode": returncode,
        "full_run_annual_from_warmup_start": indicator.get("pnl_ratio_annual") if indicator else None,
        "slice_metric_method": "juejin_log_nav_normalized_from_anchor",
        **metrics,
        "log_file": str(log_file),
    }


def _summarize(rows: list[dict]) -> list[dict]:
    out = []
    for strategy in sorted({row["strategy"] for row in rows}):
        for warmup_days in sorted({int(row["warmup_days"]) for row in rows if row["strategy"] == strategy}):
            items = [row for row in rows if row["strategy"] == strategy and int(row["warmup_days"]) == warmup_days]
            annuals = [float(row["annual"]) for row in items if row.get("annual") is not None]
            sharpes = [float(row["sharpe"]) for row in items if row.get("sharpe") is not None]
            drawdowns = [float(row["max_drawdown"]) for row in items if row.get("max_drawdown") is not None]
            invested = [float(row["avg_invested_pct"]) for row in items if row.get("avg_invested_pct") is not None]
            out.append(
                {
                    "strategy": strategy,
                    "warmup_days": warmup_days,
                    "anchor_count": len(items),
                    "annual_min": min(annuals) if annuals else None,
                    "annual_median": sorted(annuals)[len(annuals) // 2] if annuals else None,
                    "annual_max": max(annuals) if annuals else None,
                    "sharpe_min": min(sharpes) if sharpes else None,
                    "sharpe_median": sorted(sharpes)[len(sharpes) // 2] if sharpes else None,
                    "max_drawdown_max": max(drawdowns) if drawdowns else None,
                    "avg_invested_pct_median": sorted(invested)[len(invested) // 2] if invested else None,
                }
            )
    out.sort(key=lambda row: ((row["annual_min"] or -999), (row["annual_median"] or -999)), reverse=True)
    return out


def main() -> int:
    if not STRATEGY_DIR.joinpath("main.py").exists():
        raise SystemExit(f"juejin strategy main.py not found: {STRATEGY_DIR}")
    dates = _market_dates()
    rows = []
    total = len(STRATEGIES) * len(ANCHORS) * len(WARMUP_DAYS)
    done = 0
    for strategy in STRATEGIES:
        for anchor in ANCHORS:
            for warmup_days in WARMUP_DAYS:
                done += 1
                start_date = _warmup_start(anchor, warmup_days, dates)
                row = _run_case(strategy, anchor, warmup_days, start_date)
                rows.append(row)
                _write_rows(REPORT_DIR / "warmup_cases.csv", rows)
                print(
                    f"[{done}/{total}] {strategy['name']} anchor={anchor} warm={warmup_days} "
                    f"annual={row['annual']} sharpe={row['sharpe']} maxdd={row['max_drawdown']}",
                    flush=True,
                )
    summary = _summarize(rows)
    _write_rows(REPORT_DIR / "warmup_summary.csv", summary)
    (REPORT_DIR / "warmup_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
