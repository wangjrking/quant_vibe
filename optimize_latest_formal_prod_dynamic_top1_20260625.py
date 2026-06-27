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
REPORT_DIR = DATA / "reports" / "strategy_agent_latest_formal_prod_dynamic_top1_opt_20260625"
SCORE_DB = REPORT_DIR / "scores" / "grid_scores.db"
BACKTEST_SLIPPAGE_RATIO = "0.0015"

TIME_SLICES = [
    ("full", "2024-06-05 09:00:00", "2026-06-25 15:30:00"),
    ("recent120", "2025-12-24 09:00:00", "2026-06-25 15:30:00"),
    ("recent60", "2026-03-24 09:00:00", "2026-06-25 15:30:00"),
    ("ytd2026", "2026-01-05 09:00:00", "2026-06-25 15:30:00"),
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

ENTRY_CASES = [
    {
        "case_name": "base_w78_12_10_amt95_mv20",
        "weight_name": "w78_5d12_3d10",
        "w10d": 0.78,
        "w5d": 0.12,
        "w3d": 0.10,
        "amount_min": 95000.0,
        "total_mv_min": 200000.0,
    },
    {
        "case_name": "w80_12_08_amt95_mv20",
        "weight_name": "w80_5d12_3d08",
        "w10d": 0.80,
        "w5d": 0.12,
        "w3d": 0.08,
        "amount_min": 95000.0,
        "total_mv_min": 200000.0,
    },
    {
        "case_name": "w82_10_08_amt95_mv20",
        "weight_name": "w82_5d10_3d08",
        "w10d": 0.82,
        "w5d": 0.10,
        "w3d": 0.08,
        "amount_min": 95000.0,
        "total_mv_min": 200000.0,
    },
    {
        "case_name": "w80_12_08_amt120_mv20",
        "weight_name": "w80_5d12_3d08",
        "w10d": 0.80,
        "w5d": 0.12,
        "w3d": 0.08,
        "amount_min": 120000.0,
        "total_mv_min": 200000.0,
    },
    {
        "case_name": "w78_10_12_amt95_mv20",
        "weight_name": "w78_5d10_3d12",
        "w10d": 0.78,
        "w5d": 0.10,
        "w3d": 0.12,
        "amount_min": 95000.0,
        "total_mv_min": 200000.0,
    },
]

EXEC_CASES = [
    {
        "case_name": "exec_base",
        "target_position_pct": 0.8975,
        "score_continue_entry_ratio": 0.975,
        "score_exit_entry_ratio": 0.97,
        "holding_days": 2,
        "max_holding_days": 3,
        "min_holding_days_before_score_exit": 1,
        "stop_loss_pct": 0.06,
        "take_profit_pct": 0.07,
        "dd_soft_trigger": 0.08,
        "dd_hard_trigger": 0.115,
        "dd_recover_trigger": 0.03,
        "dd_soft_scale": 0.75,
        "dd_hard_scale": 0.55,
    },
    {
        "case_name": "exec_c097_pos90",
        "target_position_pct": 0.90,
        "score_continue_entry_ratio": 0.97,
        "score_exit_entry_ratio": 0.97,
        "holding_days": 2,
        "max_holding_days": 3,
        "min_holding_days_before_score_exit": 1,
        "stop_loss_pct": 0.06,
        "take_profit_pct": 0.07,
        "dd_soft_trigger": 0.08,
        "dd_hard_trigger": 0.115,
        "dd_recover_trigger": 0.03,
        "dd_soft_scale": 0.75,
        "dd_hard_scale": 0.55,
    },
    {
        "case_name": "exec_c0975_pos94",
        "target_position_pct": 0.94,
        "score_continue_entry_ratio": 0.975,
        "score_exit_entry_ratio": 0.97,
        "holding_days": 2,
        "max_holding_days": 3,
        "min_holding_days_before_score_exit": 1,
        "stop_loss_pct": 0.06,
        "take_profit_pct": 0.07,
        "dd_soft_trigger": 0.08,
        "dd_hard_trigger": 0.115,
        "dd_recover_trigger": 0.03,
        "dd_soft_scale": 0.75,
        "dd_hard_scale": 0.55,
    },
    {
        "case_name": "exec_c097_dd13",
        "target_position_pct": 0.90,
        "score_continue_entry_ratio": 0.97,
        "score_exit_entry_ratio": 0.97,
        "holding_days": 2,
        "max_holding_days": 3,
        "min_holding_days_before_score_exit": 1,
        "stop_loss_pct": 0.06,
        "take_profit_pct": 0.07,
        "dd_soft_trigger": 0.08,
        "dd_hard_trigger": 0.13,
        "dd_recover_trigger": 0.03,
        "dd_soft_scale": 0.78,
        "dd_hard_scale": 0.60,
    },
]


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


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
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


def _build_case_rules(base_rules: dict[str, Any], entry_case: dict[str, Any], exec_case: dict[str, Any]) -> dict[str, Any]:
    rules = json.loads(json.dumps(base_rules, ensure_ascii=False))
    rules["model_input"]["weight_name"] = entry_case["weight_name"]
    rules["model_input"]["entry_weights"] = {
        "10d": entry_case["w10d"],
        "5d": entry_case["w5d"],
        "3d": entry_case["w3d"],
    }
    rules["selection_rule"]["filter_name"] = f"amt{int(entry_case['amount_min'] // 1000)}k_mv{int(entry_case['total_mv_min'] // 10000)}w"
    rules["selection_rule"]["amount_min"] = entry_case["amount_min"]
    rules["selection_rule"]["total_mv_min"] = entry_case["total_mv_min"]
    rules["position_rule"]["target_position_pct"] = exec_case["target_position_pct"]
    rules["holding_rule"]["score_continue_entry_ratio"] = exec_case["score_continue_entry_ratio"]
    rules["holding_rule"]["score_exit_entry_ratio"] = exec_case["score_exit_entry_ratio"]
    rules["holding_rule"]["holding_days"] = exec_case["holding_days"]
    rules["holding_rule"]["max_holding_days"] = exec_case["max_holding_days"]
    rules["holding_rule"]["min_holding_days_before_score_exit"] = exec_case["min_holding_days_before_score_exit"]
    rules["risk_rule"]["intraday_stop_loss_pct"] = exec_case["stop_loss_pct"]
    rules["risk_rule"]["take_profit_pct"] = exec_case["take_profit_pct"]
    rules["risk_rule"]["dd_soft_trigger"] = exec_case["dd_soft_trigger"]
    rules["risk_rule"]["dd_hard_trigger"] = exec_case["dd_hard_trigger"]
    rules["risk_rule"]["dd_recover_trigger"] = exec_case["dd_recover_trigger"]
    rules["risk_rule"]["dd_soft_scale"] = exec_case["dd_soft_scale"]
    rules["risk_rule"]["dd_hard_scale"] = exec_case["dd_hard_scale"]
    return rules


def _case_key(entry_case: dict[str, Any], exec_case: dict[str, Any]) -> str:
    return f"{entry_case['case_name']}__{exec_case['case_name']}"


def _build_signal_and_scores(
    manifest: dict[str, Any],
    sources: dict[str, dict[str, Any]],
    case_rules: dict[str, Any],
    case_key: str,
) -> dict[str, Any]:
    latest_signal_date = _common_latest_trade_date(sources)
    signal_dates = _signal_dates(sources, latest_signal_date)
    weights = case_rules["model_input"]["entry_weights"]
    signal_rows: list[dict[str, Any]] = []
    score_rows: list[tuple[str, str, float]] = []

    for signal_date in signal_dates:
        buy_date, latest_market_date = _next_trade_date(MARKET_DB, signal_date)
        daily_rows = _build_candidates(sources["3d"], sources["5d"], sources["10d"], MARKET_DB, signal_date)
        for row in daily_rows:
            entry_score = (
                float(weights["10d"]) * float(row["rank_10d"])
                + float(weights["5d"]) * float(row["rank_5d"])
                + float(weights["3d"]) * float(row["rank_3d"])
            )
            row["entry_score"] = entry_score
            score_rows.append((signal_date, str(row["stock_code"]), entry_score))

        filtered = _apply_filters(daily_rows, case_rules)
        if not filtered:
            continue

        chosen: dict[str, Any] | None = None
        chosen_buy_day: dict[str, Any] | None = None
        for candidate in filtered:
            buy_day_row = _load_buy_day_market_row(MARKET_DB, buy_date, str(candidate["stock_code"]))
            if buy_day_row and (_is_st_like(buy_day_row) or _is_limit_buy(buy_day_row)):
                continue
            chosen = candidate
            chosen_buy_day = buy_day_row
            break
        if chosen is None:
            continue

        row = _build_signal_row(
            chosen,
            manifest,
            case_rules,
            signal_date,
            buy_date,
            chosen_buy_day,
            latest_market_date,
        )
        row["strategy_variant"] = case_key
        signal_rows.append(row)

    signal_file = REPORT_DIR / "signals" / f"{case_key}.csv"
    score_table = f"score_{case_key}".replace("-", "_").replace(".", "_")
    _write_rows(signal_file, signal_rows)
    return {
        "signal_file": signal_file,
        "signal_rows": len(signal_rows),
        "score_rows": score_rows,
        "score_table": score_table,
        "latest_signal_date": latest_signal_date,
    }


def _append_score_table(score_table: str, score_rows: list[tuple[str, str, float]]) -> None:
    SCORE_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(SCORE_DB))
    try:
        conn.execute(
            f'''
            CREATE TABLE IF NOT EXISTS "{score_table}" (
                trade_date TEXT NOT NULL,
                stock_code TEXT NOT NULL,
                pred_prob REAL NOT NULL
            )
            '''
        )
        conn.execute(f'DELETE FROM "{score_table}"')
        conn.executemany(
            f'INSERT INTO "{score_table}" (trade_date, stock_code, pred_prob) VALUES (?, ?, ?)',
            score_rows,
        )
        conn.execute(
            f'CREATE INDEX IF NOT EXISTS "idx_{score_table}_date_code" ON "{score_table}" (trade_date, stock_code)'
        )
        conn.commit()
    finally:
        conn.close()


def _run_backtest(case_key: str, case_rules: dict[str, Any], signal_file: Path, score_table: str, tag: str, start: str, end: str) -> dict[str, Any]:
    log_file = REPORT_DIR / "logs" / f"{case_key}__{tag}.log"
    env = os.environ.copy()
    env.update(BASE_ENV)
    env["GM_EQUITY_DD_SOFT_TRIGGER"] = str(case_rules["risk_rule"]["dd_soft_trigger"])
    env["GM_EQUITY_DD_HARD_TRIGGER"] = str(case_rules["risk_rule"]["dd_hard_trigger"])
    env["GM_EQUITY_DD_RECOVER_TRIGGER"] = str(case_rules["risk_rule"]["dd_recover_trigger"])
    env["GM_EQUITY_DD_SOFT_SCALE"] = str(case_rules["risk_rule"]["dd_soft_scale"])
    env["GM_EQUITY_DD_HARD_SCALE"] = str(case_rules["risk_rule"]["dd_hard_scale"])
    env["GM_SCORE_EXIT_ENTRY_RATIO"] = str(case_rules["holding_rule"]["score_exit_entry_ratio"])
    env["GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT"] = str(case_rules["holding_rule"]["min_holding_days_before_score_exit"])
    env["GM_SCORE_CONTINUE_ENTRY_RATIO"] = str(case_rules["holding_rule"]["score_continue_entry_ratio"])

    command = [
        str(JUEJIN_PYTHON),
        str(MAIN / "run_juejin_signal_backtest.py"),
        "--strategy-dir",
        str(CODE_SNAPSHOT_DIR),
        "--signal-file",
        str(signal_file),
        "--log-file",
        str(log_file),
        "--max-positions",
        str(case_rules["position_rule"]["max_positions"]),
        "--holding-days",
        str(case_rules["holding_rule"]["holding_days"]),
        "--max-holding-days",
        str(case_rules["holding_rule"]["max_holding_days"]),
        "--target-position-pct",
        str(case_rules["position_rule"]["target_position_pct"]),
        "--score-db",
        str(SCORE_DB),
        "--score-table",
        score_table,
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
        str(BACKTEST_SLIPPAGE_RATIO),
        "--stop-loss-pct",
        str(case_rules["risk_rule"]["intraday_stop_loss_pct"]),
        "--take-profit-pct",
        str(case_rules["risk_rule"]["take_profit_pct"]),
    ]
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as log:
        proc = subprocess.run(command, cwd=str(MAIN), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    indicator = _extract_indicator(log_file)
    return {
        "case_key": case_key,
        "slice": tag,
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


def _summary_row(
    entry_case: dict[str, Any],
    exec_case: dict[str, Any],
    case_meta: dict[str, Any],
    slice_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_slice = {row["slice"]: row for row in slice_rows}
    full = by_slice["full"]
    recent120 = by_slice["recent120"]
    recent60 = by_slice["recent60"]
    ytd = by_slice["ytd2026"]
    return {
        "case_key": _case_key(entry_case, exec_case),
        "entry_case": entry_case["case_name"],
        "exec_case": exec_case["case_name"],
        "weight_name": entry_case["weight_name"],
        "w10d": entry_case["w10d"],
        "w5d": entry_case["w5d"],
        "w3d": entry_case["w3d"],
        "amount_min": entry_case["amount_min"],
        "total_mv_min": entry_case["total_mv_min"],
        "target_position_pct": exec_case["target_position_pct"],
        "score_continue_entry_ratio": exec_case["score_continue_entry_ratio"],
        "score_exit_entry_ratio": exec_case["score_exit_entry_ratio"],
        "stop_loss_pct": exec_case["stop_loss_pct"],
        "take_profit_pct": exec_case["take_profit_pct"],
        "dd_soft_trigger": exec_case["dd_soft_trigger"],
        "dd_hard_trigger": exec_case["dd_hard_trigger"],
        "dd_soft_scale": exec_case["dd_soft_scale"],
        "dd_hard_scale": exec_case["dd_hard_scale"],
        "signal_rows": case_meta["signal_rows"],
        "full_annual": full["annual"],
        "full_sharpe": full["sharpe"],
        "full_max_drawdown": full["max_drawdown"],
        "recent120_annual": recent120["annual"],
        "recent60_annual": recent60["annual"],
        "ytd2026_annual": ytd["annual"],
    }


def _sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    def _f(key: str) -> float:
        value = row.get(key)
        return float(value) if value not in (None, "") else float("-inf")

    dd = row.get("full_max_drawdown")
    dd_value = float(dd) if dd not in (None, "") else float("inf")
    return (
        _f("recent60_annual"),
        _f("recent120_annual"),
        _f("full_annual"),
        _f("full_sharpe"),
        -dd_value,
    )


def main() -> None:
    manifest, base_rules, sources = _load_strategy(STRATEGY_DIR)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    detail_path = REPORT_DIR / "detail.csv"
    summary_path = REPORT_DIR / "summary.csv"
    detail_rows: list[dict[str, Any]] = _read_rows(detail_path)
    summary_rows: list[dict[str, Any]] = _read_rows(summary_path)
    completed = {row["case_key"] for row in summary_rows if row.get("case_key")}
    if not completed and SCORE_DB.exists():
        SCORE_DB.unlink()

    for entry_case in ENTRY_CASES:
        for exec_case in EXEC_CASES:
            case_key = _case_key(entry_case, exec_case)
            if case_key in completed:
                continue
            case_rules = _build_case_rules(base_rules, entry_case, exec_case)
            case_meta = _build_signal_and_scores(manifest, sources, case_rules, case_key)
            _append_score_table(case_meta["score_table"], case_meta["score_rows"])
            slice_rows = [
                _run_backtest(case_key, case_rules, case_meta["signal_file"], case_meta["score_table"], tag, start, end)
                for tag, start, end in TIME_SLICES
            ]
            detail_rows.extend(slice_rows)
            summary_rows.append(_summary_row(entry_case, exec_case, case_meta, slice_rows))
            summary_rows.sort(key=_sort_key, reverse=True)
            _write_rows(detail_path, detail_rows)
            _write_rows(summary_path, summary_rows)
            _write_json(REPORT_DIR / "summary.json", summary_rows)

    summary_rows.sort(key=_sort_key, reverse=True)
    _write_rows(detail_path, detail_rows)
    _write_rows(summary_path, summary_rows)
    _write_json(REPORT_DIR / "summary.json", summary_rows)
    print(json.dumps({"report_dir": str(REPORT_DIR), "cases": len(summary_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
