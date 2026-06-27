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

import pandas as pd

from gm_signal_module import to_gm_symbol


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_ROOT = DATA_DIR / "reports" / "strategy_agent_goal_high_annual_20260623"
SOURCE_DIR = REPORT_ROOT / "current_formal_1d3d5d10d_fusion"
OUT_DIR = REPORT_ROOT / "tailpow_exit_1d3d_lowpath"
MARKET_DB = DATA_DIR / "STOCK_DAILY_DATA.db"
STRATEGY_DIR = Path(r"D:\work\dfcf\juejin\strategy\ce327750-634e-11f1-b8d7-10ffe0295517")
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")
BACKTEST_END = "2026-06-23 15:30:00"

STARTS = [
    ("full_20240605", "2024-06-05 09:00:00"),
    ("late_20250701", "2025-07-01 09:00:00"),
    ("late_20251009", "2025-10-09 09:00:00"),
    ("late_20260105", "2026-01-05 09:00:00"),
]

SELECTORS = [
    ("f_10d80_5d20", "formal_l5", {"10d": 0.80, "5d": 0.20}),
    ("f_10d85_5d15", "formal_l5", {"10d": 0.85, "5d": 0.15}),
    ("f_10d90_5d10", "formal_l5", {"10d": 0.90, "5d": 0.10}),
    ("f_10d75_5d25", "formal_l5", {"10d": 0.75, "5d": 0.25}),
    ("f_10d80_5d15_3d05", "formal_l5", {"10d": 0.80, "5d": 0.15, "3d": 0.05}),
    ("f_10d85_5d20_3dn05", "formal_l5", {"10d": 0.85, "5d": 0.20, "3d": -0.05}),
    ("r_10d70_5d20_1d10", "research_includes_1d_l4_only", {"10d": 0.70, "5d": 0.20, "1d": 0.10}),
    ("r_10d60_5d20_1d20", "research_includes_1d_l4_only", {"10d": 0.60, "5d": 0.20, "1d": 0.20}),
    ("r_10d60_5d10_3d10_1d20", "research_includes_1d_l4_only", {"10d": 0.60, "5d": 0.10, "3d": 0.10, "1d": 0.20}),
    ("r_10d50_5d10_3d20_1d20", "research_includes_1d_l4_only", {"10d": 0.50, "5d": 0.10, "3d": 0.20, "1d": 0.20}),
]

EXIT_SETTINGS = [
    ("g138_o942", 1.38, 0.942),
    ("g140_o946", 1.40, 0.946),
]

BASE_ENV = {
    "GM_OPEN_DAILY_SCORE_EXIT": "1",
    "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "2",
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


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", text)


def _to_float(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return value


def _is_missing(value) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    if not text or text.upper() in {"NONE", "NAN", "NULL"}:
        return True
    try:
        return math.isnan(float(text))
    except Exception:
        return False


def _is_st_like(row: dict | None) -> bool:
    if not row:
        return True
    name = str(row.get("name") or "").strip().upper()
    if name.startswith(("ST", "*ST")) or "退" in name or "閫€" in name:
        return True
    st_type = row.get("ST_TYPE", row.get("st_type"))
    if _is_missing(st_type):
        return False
    text = str(st_type).strip().upper()
    if text in {"0", "0.0", "FALSE", "NONE", "NAN"}:
        return False
    return True


def _is_limit_buy(row: dict | None) -> bool:
    if not row:
        return True
    limit_times = row.get("limit_times")
    if not _is_missing(limit_times) and (_to_float(limit_times, 0.0) or 0.0) > 0.0:
        return True
    pre_close = _to_float(row.get("pre_close"))
    open_price = _to_float(row.get("open"))
    if pre_close is None or open_price is None or pre_close <= 0 or open_price <= 0:
        return True
    code = str(row.get("stock_code") or "")
    limit_pct = 0.20 if code.startswith(("300", "301", "688")) else 0.10
    return open_price >= pre_close * (1.0 + limit_pct) * 0.995


def _trade_dates() -> list[str]:
    conn = sqlite3.connect(MARKET_DB)
    try:
        return [
            str(row[0])
            for row in conn.execute(
                "SELECT DISTINCT trade_date FROM STOCK_DAILY_DATA WHERE trade_date >= '20240604' ORDER BY trade_date"
            )
        ]
    finally:
        conn.close()


def _market_rows() -> dict[str, dict[str, dict]]:
    conn = sqlite3.connect(MARKET_DB)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT trade_date, stock_code, name, pre_close, open, close,
                   limit_times, ST_TYPE, ST_TYPE_name
            FROM STOCK_DAILY_DATA
            WHERE trade_date >= '20240604'
            """
        ).fetchall()
    finally:
        conn.close()
    out: dict[str, dict[str, dict]] = {}
    for row in rows:
        out.setdefault(str(row["trade_date"]), {})[str(row["stock_code"])] = dict(row)
    return out


def _tailpow(value: float, gamma: float, upper: float) -> float:
    x = max(0.0, min(1.0, float(value) / upper))
    return upper * (1.0 - ((1.0 - x) ** float(gamma)))


def _load_base() -> pd.DataFrame:
    path = SOURCE_DIR / "fusion_base.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    frame = frame.copy()
    frame["rank_10d_tail_base"] = frame["rank_10d"].astype(float)
    return frame


def _candidate_name(selector_name: str, exit_name: str) -> str:
    return _safe_name(f"{selector_name}_{exit_name}")


def _write_score_table(base: pd.DataFrame, name: str, gamma: float) -> tuple[Path, str]:
    score_db = OUT_DIR / "scores_tailpow_exit_1d3d_lowpath.db"
    table = f"score_{name}"
    upper = max(float(base["rank_10d_tail_base"].max()), 1.0)
    score_frame = base[["trade_date", "stock_code", "rank_10d_tail_base"]].copy()
    score_frame["pred_prob"] = score_frame["rank_10d_tail_base"].map(lambda value: _tailpow(float(value), gamma, upper))
    score_frame = score_frame[["trade_date", "stock_code", "pred_prob"]]
    score_db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(score_db)
    try:
        score_frame.to_sql(table, conn, if_exists="replace", index=False)
        conn.execute(f'CREATE INDEX IF NOT EXISTS "idx_{table}_date_code" ON "{table}" (trade_date, stock_code)')
        conn.execute(f'CREATE INDEX IF NOT EXISTS "idx_{table}_date_score" ON "{table}" (trade_date, pred_prob DESC)')
        conn.commit()
    finally:
        conn.close()
    return score_db, table


def _write_signals(
    base: pd.DataFrame,
    market: dict[str, dict[str, dict]],
    next_date: dict[str, str],
    name: str,
    weights: dict[str, float],
    gamma: float,
) -> Path:
    frame = base.copy()
    frame["entry_score"] = 0.0
    for label, weight in weights.items():
        frame["entry_score"] += float(weight) * frame[f"rank_{label}"].astype(float)
    frame["rank"] = frame.groupby("trade_date")["entry_score"].rank(method="first", ascending=False)
    selected = frame[frame["rank"] <= 4].copy()
    upper = max(float(frame["rank_10d_tail_base"].max()), 1.0)
    selected["pred_prob"] = selected["rank_10d_tail_base"].map(lambda value: _tailpow(float(value), gamma, upper))
    rows = []
    for row in selected.sort_values(["trade_date", "rank", "stock_code"]).to_dict("records"):
        signal_date = str(row["trade_date"])
        buy_date = next_date.get(signal_date)
        if not buy_date:
            continue
        stock_code = str(row["stock_code"])
        buy_market = market.get(buy_date, {}).get(stock_code)
        if _is_st_like(buy_market) or _is_limit_buy(buy_market):
            continue
        rank = int(row["rank"])
        signal = {
            "signal_date": signal_date,
            "buy_date": buy_date,
            "symbol": to_gm_symbol(stock_code),
            "stock_code": stock_code,
            "name": row.get("name"),
            "rank": rank,
            "pred_prob": row["pred_prob"],
            "entry_score": row["entry_score"],
            "pred_1d": row.get("pred_1d"),
            "pred_3d": row.get("pred_3d"),
            "pred_5d": row.get("pred_5d"),
            "pred_10d": row.get("pred_10d"),
            "rank_1d": row.get("rank_1d"),
            "rank_3d": row.get("rank_3d"),
            "rank_5d": row.get("rank_5d"),
            "rank_10d": row.get("rank_10d"),
            "target_pct": "0.33050",
            "holding_days": 5,
            "max_holding_days": 5,
        }
        if rank == 1:
            signal["score_exit_entry_ratio"] = "0.91000"
            signal["holding_days"] = 7
            signal["max_holding_days"] = 7
        rows.append(signal)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path = OUT_DIR / "signals" / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fields} for row in rows])
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


def _run_case(
    name: str,
    l5_scope: str,
    weights: dict[str, float],
    exit_name: str,
    exit_ratio: float,
    signal_file: Path,
    score_db: Path,
    score_table: str,
    start_name: str,
    start: str,
) -> dict:
    log_file = OUT_DIR / "logs" / f"{name}_{start_name}.log"
    indicator = _extract_indicator(log_file)
    if indicator is None:
        env = os.environ.copy()
        env.update(BASE_ENV)
        env["GM_MAX_DAILY_SELLS"] = "4"
        env["GM_SCORE_EXIT_ENTRY_RATIO"] = str(float(exit_ratio))
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
            str(score_db),
            "--score-table",
            score_table,
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
        indicator = _extract_indicator(log_file)
        returncode = proc.returncode
    else:
        returncode = 0
    return {
        "candidate": name,
        "l5_scope": l5_scope,
        "weights_json": json.dumps(weights, ensure_ascii=False, sort_keys=True),
        "exit_name": exit_name,
        "start_name": start_name,
        "returncode": returncode,
        "annual": indicator.get("pnl_ratio_annual") if indicator else None,
        "pnl_ratio": indicator.get("pnl_ratio") if indicator else None,
        "sharpe": indicator.get("sharp_ratio") if indicator else None,
        "max_drawdown": indicator.get("max_drawdown") if indicator else None,
        "win_ratio": indicator.get("win_ratio") if indicator else None,
        "open_count": indicator.get("open_count") if indicator else None,
        "close_count": indicator.get("close_count") if indicator else None,
        "signal_file": str(signal_file),
        "score_db": str(score_db),
        "score_table": score_table,
        "log_file": str(log_file),
    }


def _write_rows(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fields} for row in rows])


def _summarize(rows: list[dict]) -> list[dict]:
    summary = []
    for candidate in sorted({row["candidate"] for row in rows}):
        items = [row for row in rows if row["candidate"] == candidate]
        full = next(row for row in items if row["start_name"] == "full_20240605")
        late = [float(row["annual"]) for row in items if row["start_name"].startswith("late_") and row["annual"] is not None]
        summary.append(
            {
                "candidate": candidate,
                "l5_scope": full["l5_scope"],
                "weights_json": full["weights_json"],
                "exit_name": full["exit_name"],
                "full_annual": full["annual"],
                "full_sharpe": full["sharpe"],
                "full_max_drawdown": full["max_drawdown"],
                "late_min_annual": min(late),
                "late_median_annual": float(pd.Series(late).median()),
                "late_mean_annual": float(pd.Series(late).mean()),
                "late_max_annual": max(late),
                "score": min(late) * 0.35 + float(pd.Series(late).median()) * 0.25 + float(full["annual"]) * 0.40,
            }
        )
    summary.sort(key=lambda row: (row["score"], row["full_annual"]), reverse=True)
    return summary


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = _load_base()
    dates = _trade_dates()
    next_date = {dates[index]: dates[index + 1] for index in range(len(dates) - 1)}
    market = _market_rows()
    rows = []
    total = len(SELECTORS) * len(EXIT_SETTINGS) * len(STARTS)
    done = 0
    for selector_name, l5_scope, weights in SELECTORS:
        for exit_name, gamma, exit_ratio in EXIT_SETTINGS:
            name = _candidate_name(selector_name, exit_name)
            score_db, score_table = _write_score_table(base, name, gamma)
            signal_file = _write_signals(base, market, next_date, name, weights, gamma)
            for start_name, start in STARTS:
                done += 1
                row = _run_case(
                    name,
                    l5_scope,
                    weights,
                    exit_name,
                    exit_ratio,
                    signal_file,
                    score_db,
                    score_table,
                    start_name,
                    start,
                )
                rows.append(row)
                print(
                    f"[{done}/{total}] {name} {start_name} annual={row['annual']} "
                    f"sharpe={row['sharpe']} maxdd={row['max_drawdown']}",
                    flush=True,
                )
                _write_rows(OUT_DIR / "tailpow_lowpath_cases.csv", rows)
    summary = _summarize(rows)
    _write_rows(OUT_DIR / "tailpow_lowpath_summary.csv", summary)
    (OUT_DIR / "tailpow_lowpath_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
