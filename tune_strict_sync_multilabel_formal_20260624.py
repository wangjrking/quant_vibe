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
PARENT_REPORT_DIR = SOURCE_DIR / "diversification_tune_20260624"
REPORT_DIR = PARENT_REPORT_DIR / "strict_sync_multilabel_formal_20260624"
BASE_CACHE = SOURCE_DIR / "latest_formal_base_no_delist_v2.parquet"
MARKET_DB = DATA_DIR / "STOCK_DAILY_DATA.db"
STRATEGY_DIR = PARENT_REPORT_DIR / "strict_sync_strategy_snapshot_20260624"
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

WEIGHT_SETS = [
    ("w10_90_5_10", {"10d": 0.90, "5d": 0.10}),
    ("w10_89_5_11", {"10d": 0.89, "5d": 0.11}),
    ("w10_87_5_13", {"10d": 0.87, "5d": 0.13}),
    ("w10_80_5_20", {"10d": 0.80, "5d": 0.20}),
    ("w10_80_5_10_3_10", {"10d": 0.80, "5d": 0.10, "3d": 0.10}),
    ("w10_70_5_15_3_15", {"10d": 0.70, "5d": 0.15, "3d": 0.15}),
    ("w10_60_5_20_3_20", {"10d": 0.60, "5d": 0.20, "3d": 0.20}),
    ("w10_75_5_10_3_15", {"10d": 0.75, "5d": 0.10, "3d": 0.15}),
    ("w10_70_5_20_1_10", {"10d": 0.70, "5d": 0.20, "1d": 0.10}),
    ("w10_65_5_15_3_15_1_05", {"10d": 0.65, "5d": 0.15, "3d": 0.15, "1d": 0.05}),
]

CASES: list[dict] = []
for weight_name, weights in WEIGHT_SETS:
    for exit_ratio in [0.98, 0.99]:
        CASES.append(
            {
                "name": f"ss_{weight_name}_h5_e{int(exit_ratio * 100):03d}",
                "weights": weights,
                "targets": [0.99],
                "holding_days": 5,
                "exit_ratio": exit_ratio,
                "min_hold": 1,
                "stop_loss_pct": 0.08,
            }
        )
for weight_name, weights in [
    ("w10_89_5_11", {"10d": 0.89, "5d": 0.11}),
    ("w10_87_5_13", {"10d": 0.87, "5d": 0.13}),
    ("w10_70_5_15_3_15", {"10d": 0.70, "5d": 0.15, "3d": 0.15}),
]:
    CASES.append(
        {
            "name": f"ss_{weight_name}_h5_e098_mh2",
            "weights": weights,
            "targets": [0.99],
            "holding_days": 5,
            "exit_ratio": 0.98,
            "min_hold": 2,
            "stop_loss_pct": 0.08,
        }
    )
for weight_name, weights in [
    ("w10_90_5_10", {"10d": 0.90, "5d": 0.10}),
    ("w10_89_5_11", {"10d": 0.89, "5d": 0.11}),
    ("w10_70_5_15_3_15", {"10d": 0.70, "5d": 0.15, "3d": 0.15}),
]:
    CASES.append(
        {
            "name": f"ss_{weight_name}_h4_e098",
            "weights": weights,
            "targets": [0.99],
            "holding_days": 4,
            "exit_ratio": 0.98,
            "min_hold": 1,
            "stop_loss_pct": 0.08,
        }
    )


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


def _load_base() -> pd.DataFrame:
    if not BASE_CACHE.exists():
        raise FileNotFoundError(BASE_CACHE)
    frame = pd.read_parquet(BASE_CACHE)
    required = {"trade_date", "stock_code", "rank_1d", "rank_3d", "rank_5d", "rank_10d", "name"}
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
                    "pred_prob": float(item["entry_score"]),
                    "entry_score": item["entry_score"],
                    "pred_1d": item.get("pred_1d"),
                    "pred_3d": item.get("pred_3d"),
                    "pred_5d": item.get("pred_5d"),
                    "pred_10d": item.get("pred_10d"),
                    "rank_1d": item.get("rank_1d"),
                    "rank_3d": item.get("rank_3d"),
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


def _write_score_table(base: pd.DataFrame, cfg: dict) -> tuple[Path, str]:
    path = REPORT_DIR / "scores_strict_multilabel.db"
    table = "score_" + re.sub(r"[^A-Za-z0-9_]+", "_", cfg["name"])
    payload = base[["trade_date", "stock_code"]].copy()
    payload["pred_prob"] = _entry_score(base, cfg["weights"])
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
        env["GM_STOP_LOSS_PCT"] = str(float(cfg["stop_loss_pct"]))
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
                    float(full["annual"]) * 0.25
                    + (float(full.get("sharpe") or 0.0) * 0.60)
                    - (float(full.get("max_drawdown") or 0.0) * 2.2)
                    + ((min(late) if late else 0.0) * 0.35)
                    + ((float(pd.Series(late).median()) if late else 0.0) * 0.25)
                ),
            }
        )
    out.sort(key=lambda row: (row["score"], row["full_annual"]), reverse=True)
    return out


def main() -> int:
    if not STRATEGY_DIR.joinpath("main.py").exists():
        raise SystemExit(f"strict sync strategy main.py not found: {STRATEGY_DIR}")
    base = _load_base()
    market = top1._market_rows()
    next_date = _date_map(base)
    all_rows: list[dict] = []
    signal_stats: list[dict] = []
    total = len(CASES) * len(STARTS)
    done = 0
    for cfg in CASES:
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
