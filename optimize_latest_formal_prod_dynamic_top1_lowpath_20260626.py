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

ANCHORS = ["20250701", "20251009", "20260105"]
BACKTEST_END = "2026-06-25 15:30:00"
REPORT_DIR = DATA / "reports" / "strategy_agent_latest_formal_prod_dynamic_top1_lowpath_20260626"

CANDIDATES = [
    {
        "name": "w84_09_07_amt90_mv20__exec_c097_pos90_dd12",
        "signal_file": DATA
        / "reports"
        / "strategy_agent_latest_formal_prod_dynamic_top1_opt_round3_20260626"
        / "signals"
        / "w84_09_07_amt90_mv20__exec_c097_pos90_dd12.csv",
        "score_db": DATA
        / "reports"
        / "strategy_agent_latest_formal_prod_dynamic_top1_opt_round3_20260626"
        / "scores"
        / "grid_scores.db",
        "score_table": "score_w84_09_07_amt90_mv20__exec_c097_pos90_dd12",
        "target_position_pct": 0.90,
        "holding_days": 2,
        "max_holding_days": 3,
        "score_continue_entry_ratio": 0.97,
        "score_exit_entry_ratio": 0.97,
        "min_holding_days_before_score_exit": 1,
        "stop_loss_pct": 0.06,
        "take_profit_pct": 0.07,
        "dd_soft_trigger": 0.085,
        "dd_hard_trigger": 0.12,
        "dd_recover_trigger": 0.03,
        "dd_soft_scale": 0.78,
        "dd_hard_scale": 0.58,
    },
    {
        "name": "w83_11_06_amt95_mv20__exec_c097_pos90",
        "signal_file": DATA
        / "reports"
        / "strategy_agent_latest_formal_prod_dynamic_top1_opt_round2_20260626"
        / "signals"
        / "w83_11_06_amt95_mv20__exec_c097_pos90.csv",
        "score_db": DATA
        / "reports"
        / "strategy_agent_latest_formal_prod_dynamic_top1_opt_round2_20260626"
        / "scores"
        / "grid_scores.db",
        "score_table": "score_w83_11_06_amt95_mv20__exec_c097_pos90",
        "target_position_pct": 0.90,
        "holding_days": 2,
        "max_holding_days": 3,
        "score_continue_entry_ratio": 0.97,
        "score_exit_entry_ratio": 0.97,
        "min_holding_days_before_score_exit": 1,
        "stop_loss_pct": 0.06,
        "take_profit_pct": 0.07,
        "dd_soft_trigger": 0.08,
        "dd_hard_trigger": 0.115,
        "dd_recover_trigger": 0.03,
        "dd_soft_scale": 0.75,
        "dd_hard_scale": 0.55,
    },
]

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

COLD_START_RULES = [
    {"name": "base", "startup_buy_days": 0, "target_scale": 1.00},
    {"name": "scale80_d10", "startup_buy_days": 10, "target_scale": 0.80},
    {"name": "scale70_d10", "startup_buy_days": 10, "target_scale": 0.70},
    {"name": "scale80_d20", "startup_buy_days": 20, "target_scale": 0.80},
    {"name": "scale70_d20", "startup_buy_days": 20, "target_scale": 0.70},
    {"name": "scale60_d20", "startup_buy_days": 20, "target_scale": 0.60},
]

WARMUP_DAYS = [20, 40, 60]


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
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
            "period_days": len(rows),
            "annual": None,
            "pnl": None,
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
        "period_days": len(rows),
        "annual": (end / start) ** (252.0 / max(len(rows) - 1, 1)) - 1.0 if start > 0 else None,
        "pnl": end / start - 1.0 if start > 0 else None,
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


def _warmup_start(anchor: str, warmup_days: int, dates: list[str]) -> str:
    idx = dates.index(anchor)
    return dates[max(idx - warmup_days, 0)]


def _apply_cold_start_rule(rows: list[dict[str, str]], anchor: str, rule: dict[str, Any]) -> list[dict[str, str]]:
    startup_days = int(rule["startup_buy_days"])
    scale = float(rule["target_scale"])
    if startup_days <= 0 or scale >= 0.999999:
        return [dict(row) for row in rows]
    buy_dates = sorted({str(row.get("buy_date") or "") for row in rows if str(row.get("buy_date") or "") >= anchor})
    selected = set(buy_dates[:startup_days])
    out: list[dict[str, str]] = []
    for raw in rows:
        row = dict(raw)
        buy_date = str(row.get("buy_date") or "")
        if buy_date in selected:
            current = float(row.get("target_pct") or 0.0)
            row["target_pct"] = f"{current * scale:.5f}"
        out.append(row)
    return out


def _run_backtest(
    candidate: dict[str, Any],
    signal_file: Path,
    log_file: Path,
    start_date: str,
) -> tuple[int, dict[str, Any] | None]:
    indicator = _extract_indicator(log_file)
    returncode = 0
    if indicator is None:
        env = os.environ.copy()
        env.update(BASE_ENV)
        env["GM_EQUITY_DD_SOFT_TRIGGER"] = str(candidate["dd_soft_trigger"])
        env["GM_EQUITY_DD_HARD_TRIGGER"] = str(candidate["dd_hard_trigger"])
        env["GM_EQUITY_DD_RECOVER_TRIGGER"] = str(candidate["dd_recover_trigger"])
        env["GM_EQUITY_DD_SOFT_SCALE"] = str(candidate["dd_soft_scale"])
        env["GM_EQUITY_DD_HARD_SCALE"] = str(candidate["dd_hard_scale"])
        env["GM_SCORE_EXIT_ENTRY_RATIO"] = str(candidate["score_exit_entry_ratio"])
        env["GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT"] = str(candidate["min_holding_days_before_score_exit"])
        env["GM_SCORE_CONTINUE_ENTRY_RATIO"] = str(candidate["score_continue_entry_ratio"])
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
            str(candidate["holding_days"]),
            "--max-holding-days",
            str(candidate["max_holding_days"]),
            "--target-position-pct",
            str(candidate["target_position_pct"]),
            "--score-db",
            str(candidate["score_db"]),
            "--score-table",
            str(candidate["score_table"]),
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
            "--stop-loss-pct",
            str(candidate["stop_loss_pct"]),
            "--take-profit-pct",
            str(candidate["take_profit_pct"]),
        ]
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w", encoding="utf-8") as log:
            proc = subprocess.run(command, cwd=str(MAIN), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        returncode = proc.returncode
        indicator = _extract_indicator(log_file)
    return returncode, indicator


def _cold_start_cases(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _read_rows(candidate["signal_file"])
    results: list[dict[str, Any]] = []
    for anchor in ANCHORS:
        for rule in COLD_START_RULES:
            variant_name = f"{candidate['name']}__cold__{anchor}__{rule['name']}"
            signal_path = REPORT_DIR / "signals" / f"{variant_name}.csv"
            log_path = REPORT_DIR / "logs" / f"{variant_name}.log"
            if not signal_path.exists():
                adjusted = _apply_cold_start_rule(rows, anchor, rule)
                _write_rows(signal_path, adjusted)
            returncode, indicator = _run_backtest(candidate, signal_path, log_path, anchor)
            results.append(
                {
                    "candidate": candidate["name"],
                    "mode": "cold_start_rule",
                    "anchor": anchor,
                    "variant": rule["name"],
                    "warmup_days": 0,
                    "startup_buy_days": rule["startup_buy_days"],
                    "target_scale": rule["target_scale"],
                    "returncode": returncode,
                    "annual": indicator.get("pnl_ratio_annual") if indicator else None,
                    "pnl_ratio": indicator.get("pnl_ratio") if indicator else None,
                    "sharpe": indicator.get("sharp_ratio") if indicator else None,
                    "max_drawdown": indicator.get("max_drawdown") if indicator else None,
                    "win_ratio": indicator.get("win_ratio") if indicator else None,
                    "open_count": indicator.get("open_count") if indicator else None,
                    "close_count": indicator.get("close_count") if indicator else None,
                    "signal_file": str(signal_path),
                    "log_file": str(log_path),
                }
            )
    return results


def _warmup_cases(candidate: dict[str, Any], market_dates: list[str]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for anchor in ANCHORS:
        for warmup_days in WARMUP_DAYS:
            start_date = _warmup_start(anchor, warmup_days, market_dates)
            variant_name = f"{candidate['name']}__warmup__{anchor}__w{warmup_days}"
            log_path = REPORT_DIR / "logs" / f"{variant_name}.log"
            returncode, indicator = _run_backtest(candidate, candidate["signal_file"], log_path, start_date)
            nav_rows = _parse_nav(log_path)
            sliced = [row for row in nav_rows if str(row["date"]) >= anchor]
            nav_metrics = _nav_metrics(sliced)
            results.append(
                {
                    "candidate": candidate["name"],
                    "mode": "warmup_state",
                    "anchor": anchor,
                    "variant": f"warmup_{warmup_days}",
                    "warmup_days": warmup_days,
                    "startup_buy_days": 0,
                    "target_scale": 1.0,
                    "returncode": returncode,
                    "annual": nav_metrics["annual"],
                    "pnl_ratio": nav_metrics["pnl"],
                    "sharpe": nav_metrics["sharpe"],
                    "max_drawdown": nav_metrics["max_drawdown"],
                    "win_ratio": indicator.get("win_ratio") if indicator else None,
                    "open_count": indicator.get("open_count") if indicator else None,
                    "close_count": indicator.get("close_count") if indicator else None,
                    "avg_invested_pct": nav_metrics["avg_invested_pct"],
                    "max_active_positions": nav_metrics["max_active_positions"],
                    "full_run_annual_from_warmup_start": indicator.get("pnl_ratio_annual") if indicator else None,
                    "signal_file": str(candidate["signal_file"]),
                    "log_file": str(log_path),
                }
            )
    return results


def _candidate_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["candidate"]), str(row["variant"]))
        grouped.setdefault(key, []).append(row)
    for (candidate, variant), items in sorted(grouped.items()):
        annuals = [float(row["annual"]) for row in items if row.get("annual") is not None]
        sharpes = [float(row["sharpe"]) for row in items if row.get("sharpe") is not None]
        drawdowns = [float(row["max_drawdown"]) for row in items if row.get("max_drawdown") is not None]
        out.append(
            {
                "candidate": candidate,
                "mode": items[0]["mode"],
                "variant": variant,
                "anchors": ",".join(str(row["anchor"]) for row in items),
                "min_annual": min(annuals) if annuals else None,
                "median_annual": sorted(annuals)[len(annuals) // 2] if annuals else None,
                "max_annual": max(annuals) if annuals else None,
                "min_sharpe": min(sharpes) if sharpes else None,
                "max_max_drawdown": max(drawdowns) if drawdowns else None,
                "pass_gt_100pct_all_anchors": bool(annuals) and min(annuals) >= 1.0,
                "row_count": len(items),
            }
        )
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        return
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


def main() -> int:
    market_dates = _market_dates()
    all_rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        all_rows.extend(_cold_start_cases(candidate))
        all_rows.extend(_warmup_cases(candidate, market_dates))
    summary_rows = _candidate_summary(all_rows)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(REPORT_DIR / "detail.csv", all_rows)
    _write_csv(REPORT_DIR / "summary.csv", summary_rows)
    (REPORT_DIR / "detail.json").write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
