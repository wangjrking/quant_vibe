from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import os
import sqlite3
import subprocess
from pathlib import Path
from typing import Any

from export_dynamic_top1_formal_signals import (
    _apply_filters,
    _build_candidates,
    _build_signal_row,
    _common_latest_trade_date,
    _is_limit_buy,
    _is_st_like,
    _load_buy_day_market_row,
    _load_strategy,
    _next_trade_date,
)


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
)
CODE_SNAPSHOT_DIR = STRATEGY_DIR / "code_snapshot"
REPORT_DIR = DATA / "reports" / "strategy_agent_latest_formal_prod_dynamic_top1_rebacktest_20260625"
SIGNAL_FILE = REPORT_DIR / "signals" / "prod_dynamic_top1_latest_formal_full_history.csv"
SCORE_DB = REPORT_DIR / "scores" / "latest_formal_blended_scores.db"
SCORE_TABLE = "score_prod_dynamic_top1_latest_formal_20260625"

TIME_SLICES = [
    ("full", "2024-06-05 09:00:00", "2026-06-25 15:30:00"),
    ("recent120", "2025-12-24 09:00:00", "2026-06-25 15:30:00"),
    ("recent60", "2026-03-24 09:00:00", "2026-06-25 15:30:00"),
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def _quote_ident(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _signal_dates(sources: dict[str, dict[str, Any]], latest_signal_date: str) -> list[str]:
    conn = sqlite3.connect(str(sources["10d"]["db_path"]))
    try:
        rows = conn.execute(
            f"""
            SELECT DISTINCT trade_date
            FROM {_quote_ident(sources['10d']['table'])}
            WHERE trade_date <= ?
            ORDER BY trade_date
            """,
            (str(latest_signal_date),),
        ).fetchall()
    finally:
        conn.close()
    return [str(row[0]) for row in rows]


def _rebuild_assets() -> dict[str, Any]:
    manifest, rules, sources = _load_strategy(STRATEGY_DIR)
    latest_signal_date = _common_latest_trade_date(sources)
    signal_dates = _signal_dates(sources, latest_signal_date)
    weights = rules["model_input"]["entry_weights"]

    signal_rows: list[dict[str, Any]] = []
    score_rows: list[tuple[str, str, float]] = []
    skip_days: list[dict[str, Any]] = []

    for signal_date in signal_dates:
        buy_date, latest_market_date = _next_trade_date(MARKET_DB, signal_date)
        daily_rows = _build_candidates(sources["3d"], sources["5d"], sources["10d"], MARKET_DB, signal_date)
        if not daily_rows:
            skip_days.append({"signal_date": signal_date, "reason": "no_joined_prediction_rows"})
            continue
        for row in daily_rows:
            entry_score = (
                float(weights["10d"]) * float(row["rank_10d"])
                + float(weights["5d"]) * float(row["rank_5d"])
                + float(weights["3d"]) * float(row["rank_3d"])
            )
            row["entry_score"] = entry_score
            score_rows.append((signal_date, str(row["stock_code"]), entry_score))

        filtered = _apply_filters(daily_rows, rules)
        if not filtered:
            skip_days.append({"signal_date": signal_date, "reason": "no_candidate_after_filters"})
            continue

        chosen: dict[str, Any] | None = None
        chosen_buy_day: dict[str, Any] | None = None
        skipped_candidates = 0
        for candidate in filtered:
            buy_day_row = _load_buy_day_market_row(MARKET_DB, buy_date, str(candidate["stock_code"]))
            if buy_day_row and (_is_st_like(buy_day_row) or _is_limit_buy(buy_day_row)):
                skipped_candidates += 1
                continue
            chosen = candidate
            chosen_buy_day = buy_day_row
            break
        if chosen is None:
            skip_days.append(
                {
                    "signal_date": signal_date,
                    "reason": "no_candidate_passed_buy_day_gate",
                    "candidate_count_after_filters": len(filtered),
                    "buy_day_rejected_candidates": skipped_candidates,
                }
            )
            continue

        signal_rows.append(
            _build_signal_row(
                chosen,
                manifest,
                rules,
                signal_date,
                buy_date,
                chosen_buy_day,
                latest_market_date,
            )
        )

    SIGNAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    SCORE_DB.parent.mkdir(parents=True, exist_ok=True)
    _write_rows(SIGNAL_FILE, signal_rows)

    if SCORE_DB.exists():
        SCORE_DB.unlink()
    conn = sqlite3.connect(str(SCORE_DB))
    try:
        conn.execute(f'DROP TABLE IF EXISTS "{SCORE_TABLE}"')
        conn.execute(
            f'''
            CREATE TABLE "{SCORE_TABLE}" (
                trade_date TEXT NOT NULL,
                stock_code TEXT NOT NULL,
                pred_prob REAL NOT NULL
            )
            '''
        )
        conn.executemany(
            f'INSERT INTO "{SCORE_TABLE}" (trade_date, stock_code, pred_prob) VALUES (?, ?, ?)',
            score_rows,
        )
        conn.execute(
            f'CREATE INDEX "idx_{SCORE_TABLE}_date_code" ON "{SCORE_TABLE}" (trade_date, stock_code)'
        )
        conn.commit()
    finally:
        conn.close()

    summary = {
        "strategy_id": manifest["strategy_id"],
        "latest_signal_date_common_to_3d_5d_10d": latest_signal_date,
        "signal_rows": len(signal_rows),
        "score_rows": len(score_rows),
        "signal_file": str(SIGNAL_FILE),
        "score_db": str(SCORE_DB),
        "score_table": SCORE_TABLE,
        "skip_days": skip_days[:50],
        "source_tables": {
            key: {
                "db_path": str(value["db_path"]),
                "table": value["table"],
            }
            for key, value in sources.items()
        },
    }
    _write_json(REPORT_DIR / "rebuild_summary.json", summary)
    return {"manifest": manifest, "rules": rules, "sources": sources, "summary": summary}


def _run_backtest(rules: dict[str, Any], tag: str, start: str, end: str) -> dict[str, Any]:
    log_file = REPORT_DIR / "logs" / f"{tag}.log"
    env = os.environ.copy()
    env.update(BASE_ENV)
    env["GM_EQUITY_DD_SOFT_TRIGGER"] = str(rules["risk_rule"]["dd_soft_trigger"])
    env["GM_EQUITY_DD_HARD_TRIGGER"] = str(rules["risk_rule"]["dd_hard_trigger"])
    env["GM_EQUITY_DD_RECOVER_TRIGGER"] = str(rules["risk_rule"]["dd_recover_trigger"])
    env["GM_EQUITY_DD_SOFT_SCALE"] = str(rules["risk_rule"]["dd_soft_scale"])
    env["GM_EQUITY_DD_HARD_SCALE"] = str(rules["risk_rule"]["dd_hard_scale"])
    env["GM_SCORE_EXIT_ENTRY_RATIO"] = str(rules["holding_rule"]["score_exit_entry_ratio"])
    env["GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT"] = str(rules["holding_rule"]["min_holding_days_before_score_exit"])
    env["GM_SCORE_CONTINUE_ENTRY_RATIO"] = str(rules["holding_rule"]["score_continue_entry_ratio"])

    command = [
        str(JUEJIN_PYTHON),
        str(MAIN / "run_juejin_signal_backtest.py"),
        "--strategy-dir",
        str(CODE_SNAPSHOT_DIR),
        "--signal-file",
        str(SIGNAL_FILE),
        "--log-file",
        str(log_file),
        "--max-positions",
        str(rules["position_rule"]["max_positions"]),
        "--holding-days",
        str(rules["holding_rule"]["holding_days"]),
        "--max-holding-days",
        str(rules["holding_rule"]["max_holding_days"]),
        "--target-position-pct",
        str(rules["position_rule"]["target_position_pct"]),
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
        str(rules["risk_rule"]["intraday_stop_loss_pct"]),
        "--take-profit-pct",
        str(rules["risk_rule"]["take_profit_pct"]),
    ]
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as log:
        proc = subprocess.run(command, cwd=str(MAIN), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    indicator = _extract_indicator(log_file)
    return {
        "tag": tag,
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
        "log_file": str(log_file),
    }


def main() -> None:
    rebuilt = _rebuild_assets()
    results = [_run_backtest(rebuilt["rules"], tag, start, end) for tag, start, end in TIME_SLICES]
    payload = {
        "rebuild_summary": rebuilt["summary"],
        "time_slices": results,
    }
    _write_json(REPORT_DIR / "backtest_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
