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

import tune_top1_soft_rank_latest_formal_20260623 as top1


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA_DIR = ROOT / "quant" / "data_file"
SOURCE_DIR = (
    DATA_DIR
    / "reports"
    / "strategy_agent_goal_high_annual_20260623"
    / "top1_no_delist_rerun_20260624"
)
REPORT_DIR = SOURCE_DIR / "diversification_tune_20260624"
BASE_CACHE = SOURCE_DIR / "latest_formal_base_no_delist_v2.parquet"
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

BASE_ENV = {
    "GM_OPEN_DAILY_SCORE_EXIT": "1",
    "GM_MAX_DAILY_SELLS": "3",
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

CANDIDATES = [
    {
        "name": "div_top1_w90_5d10_h4_e098",
        "weights": {"10d": 0.90, "5d": 0.10},
        "targets": [0.99],
        "holding_days": 4,
        "exit_ratio": 0.98,
        "min_hold": 1,
    },
    {
        "name": "div_top1_w90_5d10_h2_e099",
        "weights": {"10d": 0.90, "5d": 0.10},
        "targets": [0.99],
        "holding_days": 2,
        "exit_ratio": 0.99,
        "min_hold": 1,
    },
    {
        "name": "div_top1_w90_5d10_h3_e099",
        "weights": {"10d": 0.90, "5d": 0.10},
        "targets": [0.99],
        "holding_days": 3,
        "exit_ratio": 0.99,
        "min_hold": 1,
    },
    {
        "name": "div_top1_w89_5d11_h3_e099",
        "weights": {"10d": 0.89, "5d": 0.11},
        "targets": [0.99],
        "holding_days": 3,
        "exit_ratio": 0.99,
        "min_hold": 1,
    },
    {
        "name": "div_top1_w90_5d10_h5_e098",
        "weights": {"10d": 0.90, "5d": 0.10},
        "targets": [0.99],
        "holding_days": 5,
        "exit_ratio": 0.98,
        "min_hold": 1,
    },
    {
        "name": "div_top1_w87_5d13_h5_e098",
        "weights": {"10d": 0.87, "5d": 0.13},
        "targets": [0.99],
        "holding_days": 5,
        "exit_ratio": 0.98,
        "min_hold": 1,
    },
    {
        "name": "div_top1_w88_5d12_h5_e098",
        "weights": {"10d": 0.88, "5d": 0.12},
        "targets": [0.99],
        "holding_days": 5,
        "exit_ratio": 0.98,
        "min_hold": 1,
    },
    {
        "name": "div_top1_w89_5d11_h5_e098",
        "weights": {"10d": 0.89, "5d": 0.11},
        "targets": [0.99],
        "holding_days": 5,
        "exit_ratio": 0.98,
        "min_hold": 1,
    },
    {
        "name": "div_top1_w91_5d09_h5_e098",
        "weights": {"10d": 0.91, "5d": 0.09},
        "targets": [0.99],
        "holding_days": 5,
        "exit_ratio": 0.98,
        "min_hold": 1,
    },
    {
        "name": "div_top1_w92_5d08_h5_e098",
        "weights": {"10d": 0.92, "5d": 0.08},
        "targets": [0.99],
        "holding_days": 5,
        "exit_ratio": 0.98,
        "min_hold": 1,
    },
    {
        "name": "div_top1_w93_5d07_h5_e098",
        "weights": {"10d": 0.93, "5d": 0.07},
        "targets": [0.99],
        "holding_days": 5,
        "exit_ratio": 0.98,
        "min_hold": 1,
    },
    {
        "name": "div_top1_w90_5d10_h5_e096",
        "weights": {"10d": 0.90, "5d": 0.10},
        "targets": [0.99],
        "holding_days": 5,
        "exit_ratio": 0.96,
        "min_hold": 1,
    },
    {
        "name": "div_top1_w90_5d10_h5_e099",
        "weights": {"10d": 0.90, "5d": 0.10},
        "targets": [0.99],
        "holding_days": 5,
        "exit_ratio": 0.99,
        "min_hold": 1,
    },
    {
        "name": "div_top1_w90_5d10_h5_e100",
        "weights": {"10d": 0.90, "5d": 0.10},
        "targets": [0.99],
        "holding_days": 5,
        "exit_ratio": 1.00,
        "min_hold": 1,
    },
    {
        "name": "div_top1_w90_5d10_h5_e101",
        "weights": {"10d": 0.90, "5d": 0.10},
        "targets": [0.99],
        "holding_days": 5,
        "exit_ratio": 1.01,
        "min_hold": 1,
    },
    {
        "name": "div_top1_w90_5d10_h5_e102",
        "weights": {"10d": 0.90, "5d": 0.10},
        "targets": [0.99],
        "holding_days": 5,
        "exit_ratio": 1.02,
        "min_hold": 1,
    },
    {
        "name": "div_top1_w89_5d11_h5_e099",
        "weights": {"10d": 0.89, "5d": 0.11},
        "targets": [0.99],
        "holding_days": 5,
        "exit_ratio": 0.99,
        "min_hold": 1,
    },
    {
        "name": "div_top1_w91_5d09_h5_e099",
        "weights": {"10d": 0.91, "5d": 0.09},
        "targets": [0.99],
        "holding_days": 5,
        "exit_ratio": 0.99,
        "min_hold": 1,
    },
    {
        "name": "div_top1_w90_5d10_h5_e099_pos95",
        "weights": {"10d": 0.90, "5d": 0.10},
        "targets": [0.95],
        "holding_days": 5,
        "exit_ratio": 0.99,
        "min_hold": 1,
    },
    {
        "name": "div_top1_w90_5d10_h5_e099_pos93",
        "weights": {"10d": 0.90, "5d": 0.10},
        "targets": [0.93],
        "holding_days": 5,
        "exit_ratio": 0.99,
        "min_hold": 1,
    },
    {
        "name": "div_top1_w90_5d10_h5_e099_pos94",
        "weights": {"10d": 0.90, "5d": 0.10},
        "targets": [0.94],
        "holding_days": 5,
        "exit_ratio": 0.99,
        "min_hold": 1,
    },
    {
        "name": "div_top1_w90_5d10_h5_e099_pos96",
        "weights": {"10d": 0.90, "5d": 0.10},
        "targets": [0.96],
        "holding_days": 5,
        "exit_ratio": 0.99,
        "min_hold": 1,
    },
    {
        "name": "div_top1_w90_5d10_h5_e099_pos97",
        "weights": {"10d": 0.90, "5d": 0.10},
        "targets": [0.97],
        "holding_days": 5,
        "exit_ratio": 0.99,
        "min_hold": 1,
    },
    {
        "name": "div_top1_w90_5d10_h5_e099_pos98",
        "weights": {"10d": 0.90, "5d": 0.10},
        "targets": [0.98],
        "holding_days": 5,
        "exit_ratio": 0.99,
        "min_hold": 1,
    },
    {
        "name": "div_top1_w90_5d10_h5_e099_pos90",
        "weights": {"10d": 0.90, "5d": 0.10},
        "targets": [0.90],
        "holding_days": 5,
        "exit_ratio": 0.99,
        "min_hold": 1,
    },
    {
        "name": "div_top1_w90_5d10_h5_e099_mh2",
        "weights": {"10d": 0.90, "5d": 0.10},
        "targets": [0.99],
        "holding_days": 5,
        "exit_ratio": 0.99,
        "min_hold": 2,
    },
    {
        "name": "div_top1_w90_5d10_h5_e098_mh2",
        "weights": {"10d": 0.90, "5d": 0.10},
        "targets": [0.99],
        "holding_days": 5,
        "exit_ratio": 0.98,
        "min_hold": 2,
    },
    {
        "name": "div_top1_w90_5d10_h6_e098",
        "weights": {"10d": 0.90, "5d": 0.10},
        "targets": [0.99],
        "holding_days": 6,
        "exit_ratio": 0.98,
        "min_hold": 1,
    },
    {
        "name": "div_top2_w90_5d10_55_43_h5_e098",
        "weights": {"10d": 0.90, "5d": 0.10},
        "targets": [0.55, 0.43],
        "holding_days": 5,
        "exit_ratio": 0.98,
        "min_hold": 1,
    },
    {
        "name": "div_top2_w90_5d10_50_49_h5_e098",
        "weights": {"10d": 0.90, "5d": 0.10},
        "targets": [0.50, 0.49],
        "holding_days": 5,
        "exit_ratio": 0.98,
        "min_hold": 1,
    },
    {
        "name": "div_top2_w88_5d12_55_43_h5_e098",
        "weights": {"10d": 0.88, "5d": 0.12},
        "targets": [0.55, 0.43],
        "holding_days": 5,
        "exit_ratio": 0.98,
        "min_hold": 1,
    },
    {
        "name": "div_top3_w90_5d10_40_32_27_h5_e098",
        "weights": {"10d": 0.90, "5d": 0.10},
        "targets": [0.40, 0.32, 0.27],
        "holding_days": 5,
        "exit_ratio": 0.98,
        "min_hold": 1,
    },
    {
        "name": "div_top3_w90_5d10_34_33_32_h5_e098",
        "weights": {"10d": 0.90, "5d": 0.10},
        "targets": [0.34, 0.33, 0.32],
        "holding_days": 5,
        "exit_ratio": 0.98,
        "min_hold": 1,
    },
]


def _sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value)


def _load_base() -> pd.DataFrame:
    if not BASE_CACHE.exists():
        raise FileNotFoundError(BASE_CACHE)
    frame = pd.read_parquet(BASE_CACHE)
    required = {"trade_date", "stock_code", "rank_5d", "rank_10d", "name", "close"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"base cache missing columns: {missing}")
    return frame


def _date_map(frame: pd.DataFrame) -> dict[str, str]:
    dates = sorted(str(value) for value in frame["trade_date"].dropna().unique())
    return {dates[index]: dates[index + 1] for index in range(len(dates) - 1)}


def _entry_score(frame: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    score = pd.Series(0.0, index=frame.index)
    for label, weight in weights.items():
        score += float(weight) * frame[f"rank_{label}"].astype(float)
    return score


def _score_value(row: dict) -> float:
    return top1._tailpow(float(row["rank_10d"]), 2.0, 1.0)


def _build_signals(base: pd.DataFrame, market: dict[str, dict[str, dict]], next_date: dict[str, str], cfg: dict) -> list[dict]:
    frame = base.copy()
    frame["entry_score"] = _entry_score(frame, cfg["weights"])
    frame = frame.sort_values(["trade_date", "entry_score", "stock_code"], ascending=[True, False, True])
    rows: list[dict] = []
    max_names = len(cfg["targets"])
    for signal_date, day in frame.groupby("trade_date", sort=True):
        buy_date = next_date.get(str(signal_date))
        if not buy_date:
            continue
        chosen = []
        for item in day.to_dict("records"):
            stock_code = str(item["stock_code"])
            buy_market = market.get(buy_date, {}).get(stock_code)
            if top1._is_st_like(buy_market) or top1._is_limit_buy(buy_market):
                continue
            chosen.append(item)
            if len(chosen) >= max_names:
                break
        for index, item in enumerate(chosen, start=1):
            stock_code = str(item["stock_code"])
            rows.append(
                {
                    "signal_date": str(signal_date),
                    "buy_date": buy_date,
                    "symbol": top1.to_gm_symbol(stock_code),
                    "stock_code": stock_code,
                    "name": item.get("name"),
                    "rank": index,
                    "pred_prob": _score_value(item),
                    "entry_score": item["entry_score"],
                    "pred_5d": item.get("pred_5d"),
                    "pred_10d": item.get("pred_10d"),
                    "rank_5d": item.get("rank_5d"),
                    "rank_10d": item.get("rank_10d"),
                    "amount": item.get("amount"),
                    "turnover_rate": item.get("turnover_rate"),
                    "total_mv": item.get("total_mv"),
                    "atr_qfq": item.get("atr_qfq"),
                    "target_pct": f"{float(cfg['targets'][index - 1]):.5f}",
                    "holding_days": int(cfg["holding_days"]),
                    "max_holding_days": int(cfg["holding_days"]),
                    "score_exit_entry_ratio": f"{float(cfg['exit_ratio']):.5f}",
                    "min_holding_days_before_score_exit": int(cfg["min_hold"]),
                }
            )
    return rows


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


def _write_score_table(base: pd.DataFrame, cfg: dict) -> tuple[Path, str]:
    path = REPORT_DIR / "scores_diversification.db"
    table = "score_" + _sanitize(cfg["name"])
    payload = base[["trade_date", "stock_code", "rank_10d"]].copy()
    payload["pred_prob"] = payload["rank_10d"].map(lambda value: top1._tailpow(float(value), 2.0, 1.0))
    payload = payload[["trade_date", "stock_code", "pred_prob"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        payload.to_sql(table, conn, if_exists="replace", index=False)
        conn.execute(f'CREATE INDEX IF NOT EXISTS "idx_{table}_date_code" ON "{table}" (trade_date, stock_code)')
        conn.commit()
    finally:
        conn.close()
    return path, table


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


def _signal_stats(rows: list[dict]) -> dict:
    day_sums: dict[str, float] = {}
    day_counts: dict[str, int] = {}
    for row in rows:
        day = str(row.get("buy_date") or "")
        day_sums[day] = day_sums.get(day, 0.0) + float(row.get("target_pct") or 0.0)
        day_counts[day] = day_counts.get(day, 0) + 1
    return {
        "signal_count": len(rows),
        "buy_days": len(day_sums),
        "avg_names_per_buy_day": sum(day_counts.values()) / len(day_counts) if day_counts else None,
        "shortage_days": sum(1 for value in day_counts.values() if value < max(day_counts.values(), default=0)),
        "avg_day_target_sum": sum(day_sums.values()) / len(day_sums) if day_sums else None,
        "min_day_target_sum": min(day_sums.values()) if day_sums else None,
        "max_day_target_sum": max(day_sums.values()) if day_sums else None,
        "min_signal_date": min((row["signal_date"] for row in rows), default=""),
        "max_signal_date": max((row["signal_date"] for row in rows), default=""),
    }


def _run_case(cfg: dict, signal_file: Path, score_db: Path, score_table: str, start_name: str, start: str) -> dict:
    log_file = REPORT_DIR / "logs" / f"{cfg['name']}_{start_name}.log"
    indicator = _extract_indicator(log_file)
    returncode = 0
    if indicator is None:
        env = os.environ.copy()
        env.update(BASE_ENV)
        env["GM_SCORE_EXIT_ENTRY_RATIO"] = str(float(cfg["exit_ratio"]))
        env["GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT"] = str(int(cfg["min_hold"]))
        env["GM_MAX_DAILY_SELLS"] = str(len(cfg["targets"]))
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
            str(len(cfg["targets"])),
            "--holding-days",
            str(int(cfg["holding_days"])),
            "--max-holding-days",
            str(int(cfg["holding_days"])),
            "--target-position-pct",
            str(max(float(value) for value in cfg["targets"])),
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
        returncode = proc.returncode
        indicator = _extract_indicator(log_file)
    return {
        "name": cfg["name"],
        "start_name": start_name,
        "returncode": returncode,
        "annual": indicator.get("pnl_ratio_annual") if indicator else None,
        "pnl_ratio": indicator.get("pnl_ratio") if indicator else None,
        "sharpe": indicator.get("sharp_ratio") if indicator else None,
        "max_drawdown": indicator.get("max_drawdown") if indicator else None,
        "win_ratio": indicator.get("win_ratio") if indicator else None,
        "open_count": indicator.get("open_count") if indicator else None,
        "close_count": indicator.get("close_count") if indicator else None,
        "config_json": json.dumps(cfg, ensure_ascii=False, sort_keys=True),
        "signal_file": str(signal_file),
        "score_db": str(score_db),
        "score_table": score_table,
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
                    float(full["annual"]) * 0.35
                    + (float(full.get("sharpe") or 0.0) * 0.50)
                    - (float(full.get("max_drawdown") or 0.0) * 2.0)
                    + ((min(late) if late else 0.0) * 0.20)
                    + ((float(pd.Series(late).median()) if late else 0.0) * 0.25)
                ),
            }
        )
    out.sort(key=lambda row: (row["score"], row["full_annual"]), reverse=True)
    return out


def main() -> int:
    if not STRATEGY_DIR.joinpath("main.py").exists():
        raise SystemExit(f"juejin strategy main.py not found: {STRATEGY_DIR}")
    base = _load_base()
    market = top1._market_rows()
    next_date = _date_map(base)
    all_rows: list[dict] = []
    signal_stats: list[dict] = []
    total = len(CANDIDATES) * len(STARTS)
    done = 0
    for cfg in CANDIDATES:
        signals = _build_signals(base, market, next_date, cfg)
        signal_file = REPORT_DIR / "signals" / f"{cfg['name']}.csv"
        _write_rows(signal_file, signals)
        signal_stats.append({"name": cfg["name"], **_signal_stats(signals)})
        score_db, score_table = _write_score_table(base, cfg)
        for start_name, start in STARTS:
            done += 1
            row = _run_case(cfg, signal_file, score_db, score_table, start_name, start)
            all_rows.append(row)
            _write_rows(REPORT_DIR / "cases.csv", all_rows)
            print(
                f"[{done}/{total}] {cfg['name']} {start_name} "
                f"annual={row['annual']} sharpe={row['sharpe']} maxdd={row['max_drawdown']}",
                flush=True,
            )
    _write_rows(REPORT_DIR / "signal_stats.csv", signal_stats)
    summary = _summarize(all_rows)
    _write_rows(REPORT_DIR / "summary.csv", summary)
    _write_rows(REPORT_DIR / "summary_by_annual.csv", sorted(summary, key=lambda row: _metric(row, "full_annual"), reverse=True))
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
