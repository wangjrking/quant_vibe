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
from typing import Any


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA = ROOT / "quant" / "data_file"
MARKET_DB = DATA / "STOCK_DAILY_DATA.db"
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")
STRATEGY_DIR = (
    MAIN
    / "strategy_library"
    / "production"
    / "prod_dynamic_top1_amt9p5w_dd08_115_v20260625"
    / "code_snapshot"
)

CASE_NAME = "w84_09_07_amt90_mv20__exit_mid_h2_m3_c098_e0965_min1"
SIGNAL_FILE = (
    DATA
    / "reports"
    / "strategy_agent_latest_formal_prod_dynamic_top1_exit_only_0040_20260626"
    / "signals"
    / f"{CASE_NAME}.csv"
)
SCORE_DB = (
    DATA
    / "reports"
    / "strategy_agent_latest_formal_prod_dynamic_top1_exit_only_0040_20260626"
    / "scores"
    / "grid_scores.db"
)
SCORE_TABLE = f"score_{CASE_NAME}"
FULL_LOG = (
    DATA
    / "reports"
    / "strategy_agent_latest_formal_prod_dynamic_top1_exit_only_0040_20260626"
    / "logs"
    / f"{CASE_NAME}__full.log"
)
REPORT_DIR = DATA / "reports" / "strategy_agent_exit_mid_candidate_warmup60_0040_20260627"

ANCHORS = ["20250701", "20251009", "20260105"]
NEARBY_OFFSETS = [-3, -2, -1, 0, 1, 2, 3]
WARMUP_DAYS = 60
BACKTEST_END = "2026-06-25 15:30:00"

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
    "GM_EQUITY_DD_SOFT_TRIGGER": "0.085",
    "GM_EQUITY_DD_HARD_TRIGGER": "0.12",
    "GM_EQUITY_DD_RECOVER_TRIGGER": "0.03",
    "GM_EQUITY_DD_SOFT_SCALE": "0.78",
    "GM_EQUITY_DD_HARD_SCALE": "0.58",
    "GM_SCORE_EXIT_ENTRY_RATIO": "0.965",
    "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "1",
    "GM_SCORE_CONTINUE_ENTRY_RATIO": "0.98",
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


def _parse_nav(log_file: Path) -> list[dict[str, Any]]:
    pattern = re.compile(
        r"EXPOSURE\s+(?P<date>\d{8})\s+post_buy\s+invested_pct=(?P<invested>[0-9.]+)"
        r".*active_positions=(?P<active>\d+).*nav=(?P<nav>[0-9.]+)"
    )
    rows = []
    if not log_file.exists():
        return rows
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


def _max_drawdown(values: list[float]) -> float | None:
    if not values:
        return None
    peak = None
    out = 0.0
    for value in values:
        peak = value if peak is None else max(peak, value)
        if peak and peak > 0:
            out = max(out, 1.0 - value / peak)
    return out


def _nav_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) < 2:
        return {
            "annual": None,
            "pnl_ratio": None,
            "sharpe": None,
            "max_drawdown": None,
            "avg_invested_pct": None,
            "max_active_positions": None,
        }
    start = float(rows[0]["nav"])
    end = float(rows[-1]["nav"])
    returns = []
    for prev, curr in zip(rows, rows[1:]):
        if float(prev["nav"]) > 0:
            returns.append(float(curr["nav"]) / float(prev["nav"]) - 1.0)
    mean = sum(returns) / len(returns) if returns else 0.0
    variance = sum((value - mean) ** 2 for value in returns) / (len(returns) - 1) if len(returns) > 1 else 0.0
    std = math.sqrt(variance)
    return {
        "annual": (end / start) ** (252.0 / max(len(rows) - 1, 1)) - 1.0 if start > 0 else None,
        "pnl_ratio": end / start - 1.0 if start > 0 else None,
        "sharpe": mean / std * math.sqrt(252.0) if std > 0 else None,
        "max_drawdown": _max_drawdown([float(row["nav"]) for row in rows]),
        "avg_invested_pct": sum(float(row["invested_pct"]) for row in rows) / len(rows),
        "max_active_positions": max(int(row["active_positions"]) for row in rows),
    }


def _market_dates() -> list[str]:
    conn = sqlite3.connect(str(MARKET_DB))
    try:
        rows = conn.execute("SELECT DISTINCT trade_date FROM STOCK_DAILY_DATA ORDER BY trade_date").fetchall()
    finally:
        conn.close()
    return [str(row[0]) for row in rows]


def _nearby_starts(dates: list[str]) -> list[dict[str, Any]]:
    pos = {date: index for index, date in enumerate(dates)}
    out: list[dict[str, Any]] = []
    for anchor in ANCHORS:
        idx = pos[anchor]
        for offset in NEARBY_OFFSETS:
            j = idx + offset
            if j < 0 or j >= len(dates):
                continue
            out.append({"anchor": anchor, "offset": offset, "start_date": dates[j]})
    return out


def _warmup_start(start_date: str, dates: list[str]) -> str:
    idx = dates.index(start_date)
    return dates[max(idx - WARMUP_DAYS, 0)]


def _run_case(item: dict[str, Any], dates: list[str]) -> dict[str, Any]:
    warmup_start = _warmup_start(str(item["start_date"]), dates)
    tag = f"{item['anchor']}_o{int(item['offset']):+d}_{item['start_date']}".replace("+", "p").replace("-", "m")
    log_file = REPORT_DIR / "logs" / f"{CASE_NAME}_{tag}.log"
    indicator = _extract_indicator(log_file)
    returncode = 0
    if indicator is None:
        env = os.environ.copy()
        env.update(BASE_ENV)
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
            "0.9",
            "--score-db",
            str(SCORE_DB),
            "--score-table",
            SCORE_TABLE,
            "--market-db",
            str(MARKET_DB),
            "--backtest-start",
            f"{warmup_start[:4]}-{warmup_start[4:6]}-{warmup_start[6:]} 09:00:00",
            "--backtest-end",
            BACKTEST_END,
            "--backtest-adjust",
            "none",
            "--backtest-initial-cash",
            "600000",
            "--backtest-slippage-ratio",
            "0.0040",
            "--stop-loss-pct",
            "0.06",
            "--take-profit-pct",
            "0.07",
        ]
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w", encoding="utf-8") as log:
            proc = subprocess.run(command, cwd=str(MAIN), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        returncode = proc.returncode
        indicator = _extract_indicator(log_file)
    nav_rows = _parse_nav(log_file)
    sliced = [row for row in nav_rows if str(row["date"]) >= str(item["start_date"])]
    metrics = _nav_metrics(sliced)
    return {
        "anchor": item["anchor"],
        "offset": item["offset"],
        "start_date": item["start_date"],
        "warmup_start": warmup_start,
        "returncode": returncode,
        "annual": metrics["annual"],
        "pnl_ratio": metrics["pnl_ratio"],
        "sharpe": metrics["sharpe"],
        "max_drawdown": metrics["max_drawdown"],
        "avg_invested_pct": metrics["avg_invested_pct"],
        "max_active_positions": metrics["max_active_positions"],
        "full_run_annual_from_warmup_start": indicator.get("pnl_ratio_annual") if indicator else None,
        "log_file": str(log_file),
    }


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for anchor in ANCHORS:
        items = [row for row in rows if str(row["anchor"]) == anchor and row.get("annual") is not None]
        annuals = [float(row["annual"]) for row in items]
        sharpes = [float(row["sharpe"]) for row in items if row.get("sharpe") is not None]
        drawdowns = [float(row["max_drawdown"]) for row in items if row.get("max_drawdown") is not None]
        out.append(
            {
                "anchor": anchor,
                "starts": len(items),
                "annual_min": min(annuals) if annuals else None,
                "annual_median": sorted(annuals)[len(annuals) // 2] if annuals else None,
                "annual_max": max(annuals) if annuals else None,
                "sharpe_min": min(sharpes) if sharpes else None,
                "sharpe_median": sorted(sharpes)[len(sharpes) // 2] if sharpes else None,
                "max_drawdown_max": max(drawdowns) if drawdowns else None,
                "pass_gt_100pct_all_starts": bool(annuals) and min(annuals) >= 1.0,
            }
        )
    return out


def _full_metrics() -> dict[str, Any]:
    indicator = _extract_indicator(FULL_LOG)
    nav_rows = _parse_nav(FULL_LOG)
    nav = _nav_metrics(nav_rows)
    return {
        "annual_return": indicator.get("pnl_ratio_annual") if indicator else None,
        "pnl_ratio": indicator.get("pnl_ratio") if indicator else None,
        "sharpe": indicator.get("sharp_ratio") if indicator else None,
        "max_drawdown": indicator.get("max_drawdown") if indicator else None,
        "win_ratio": indicator.get("win_ratio") if indicator else None,
        "open_count": indicator.get("open_count") if indicator else None,
        "close_count": indicator.get("close_count") if indicator else None,
        "avg_invested_pct": nav.get("avg_invested_pct"),
        "max_active_positions": nav.get("max_active_positions"),
    }


def main() -> None:
    dates = _market_dates()
    starts = _nearby_starts(dates)
    detail_rows = []
    for item in starts:
        detail_rows.append(_run_case(item, dates))
        _write_rows(REPORT_DIR / "detail.csv", detail_rows)
        _write_rows(REPORT_DIR / "nearby_warmup60_summary.csv", _summarize(detail_rows))
        (REPORT_DIR / "summary.json").write_text(
            json.dumps(
                {
                    "candidate": CASE_NAME,
                    "slippage_ratio": 0.0040,
                    "full_metrics": _full_metrics(),
                    "warmup60_nearby_summary": _summarize(detail_rows),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    print(json.dumps({"report_dir": str(REPORT_DIR), "cases": len(detail_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
