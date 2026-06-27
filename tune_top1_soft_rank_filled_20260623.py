from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path

import pandas as pd

import tune_raw_tailpow_1d3d_lowpath_20260623 as raw
from gm_signal_module import to_gm_symbol


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_ROOT = DATA_DIR / "reports" / "strategy_agent_goal_high_annual_20260623"
OUT_DIR = REPORT_ROOT / "top1_soft_rank_filled_20260623"
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

CANDIDATES = [
    {"name": "fill_base_h5_g200_e098", "mode": "base", "holding_days": 5, "gamma": 2.0, "exit_ratio": 0.98, "min_hold": 1},
    {"name": "fill_base_h3_g200_e098", "mode": "base", "holding_days": 3, "gamma": 2.0, "exit_ratio": 0.98, "min_hold": 1},
    {"name": "fill_base_h1_g200_e098", "mode": "base", "holding_days": 1, "gamma": 2.0, "exit_ratio": 0.98, "min_hold": 1},
    {"name": "fill_base_h2_g200_e098", "mode": "base", "holding_days": 2, "gamma": 2.0, "exit_ratio": 0.98, "min_hold": 1},
    {"name": "fill_base_h7_g200_e098", "mode": "base", "holding_days": 7, "gamma": 2.0, "exit_ratio": 0.98, "min_hold": 1},
    {"name": "fill_base_h10_g200_e098", "mode": "base", "holding_days": 10, "gamma": 2.0, "exit_ratio": 0.98, "min_hold": 1},
    {"name": "fill_base_h5_g138_e094", "mode": "base", "holding_days": 5, "gamma": 1.38, "exit_ratio": 0.94, "min_hold": 1},
    {"name": "fill_base_h5_g138_e096", "mode": "base", "holding_days": 5, "gamma": 1.38, "exit_ratio": 0.96, "min_hold": 1},
    {"name": "fill_base_h7_g138_e094", "mode": "base", "holding_days": 7, "gamma": 1.38, "exit_ratio": 0.94, "min_hold": 1},
    {"name": "fill_base_h10_g138_e094", "mode": "base", "holding_days": 10, "gamma": 1.38, "exit_ratio": 0.94, "min_hold": 1},
    {"name": "fill_w70_h5_g200_e098", "mode": "w70", "holding_days": 5, "gamma": 2.0, "exit_ratio": 0.98, "min_hold": 1},
    {"name": "fill_w75_h5_g200_e098", "mode": "w75", "holding_days": 5, "gamma": 2.0, "exit_ratio": 0.98, "min_hold": 1},
    {"name": "fill_w85_h5_g200_e098", "mode": "w85", "holding_days": 5, "gamma": 2.0, "exit_ratio": 0.98, "min_hold": 1},
    {"name": "fill_w90_h5_g200_e098", "mode": "w90", "holding_days": 5, "gamma": 2.0, "exit_ratio": 0.98, "min_hold": 1},
    {"name": "fill_w70_h10_g138_e094", "mode": "w70", "holding_days": 10, "gamma": 1.38, "exit_ratio": 0.94, "min_hold": 1},
    {"name": "fill_w75_h10_g138_e094", "mode": "w75", "holding_days": 10, "gamma": 1.38, "exit_ratio": 0.94, "min_hold": 1},
    {"name": "fill_w85_h10_g138_e094", "mode": "w85", "holding_days": 10, "gamma": 1.38, "exit_ratio": 0.94, "min_hold": 1},
    {"name": "fill_w90_h10_g138_e094", "mode": "w90", "holding_days": 10, "gamma": 1.38, "exit_ratio": 0.94, "min_hold": 1},
    {"name": "soft_liq1_h5_g200_e098", "mode": "soft_liq1", "holding_days": 5, "gamma": 2.0, "exit_ratio": 0.98, "min_hold": 1},
    {"name": "soft_liq1_h3_g200_e098", "mode": "soft_liq1", "holding_days": 3, "gamma": 2.0, "exit_ratio": 0.98, "min_hold": 1},
    {"name": "soft_liq2_h5_g200_e098", "mode": "soft_liq2", "holding_days": 5, "gamma": 2.0, "exit_ratio": 0.98, "min_hold": 1},
    {"name": "soft_liq2_h3_g200_e098", "mode": "soft_liq2", "holding_days": 3, "gamma": 2.0, "exit_ratio": 0.98, "min_hold": 1},
    {"name": "soft_turn_h5_g200_e098", "mode": "soft_turn", "holding_days": 5, "gamma": 2.0, "exit_ratio": 0.98, "min_hold": 1},
    {"name": "soft_turn_h3_g200_e098", "mode": "soft_turn", "holding_days": 3, "gamma": 2.0, "exit_ratio": 0.98, "min_hold": 1},
]

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


def _base_score(frame: pd.DataFrame) -> pd.Series:
    return frame["rank_10d"].astype(float) * 0.80 + frame["rank_5d"].astype(float) * 0.20


def _adjusted_score(frame: pd.DataFrame, mode: str) -> pd.Series:
    score = _base_score(frame)
    amount = pd.to_numeric(frame["amount"], errors="coerce")
    turnover = pd.to_numeric(frame["turnover_rate"], errors="coerce")
    total_mv = pd.to_numeric(frame["total_mv"], errors="coerce")
    if mode == "base":
        return score
    if mode == "w70":
        return frame["rank_10d"].astype(float) * 0.70 + frame["rank_5d"].astype(float) * 0.30
    if mode == "w75":
        return frame["rank_10d"].astype(float) * 0.75 + frame["rank_5d"].astype(float) * 0.25
    if mode == "w85":
        return frame["rank_10d"].astype(float) * 0.85 + frame["rank_5d"].astype(float) * 0.15
    if mode == "w90":
        return frame["rank_10d"].astype(float) * 0.90 + frame["rank_5d"].astype(float) * 0.10
    if mode == "soft_liq1":
        score = score + ((amount >= 120000) * 0.0008)
        score = score + (((turnover >= 3.0) & (turnover <= 10.0)) * 0.0008)
        score = score + (((total_mv >= 200000) & (total_mv <= 5000000)) * 0.0004)
        score = score - ((turnover > 20.0) * 0.0006)
        return score
    if mode == "soft_liq2":
        score = score + ((amount >= 120000) * 0.0015)
        score = score + (((turnover >= 3.0) & (turnover <= 10.0)) * 0.0015)
        score = score + (((total_mv >= 200000) & (total_mv <= 5000000)) * 0.0008)
        score = score - ((turnover > 20.0) * 0.0010)
        score = score - ((total_mv < 150000) * 0.0008)
        return score
    if mode == "soft_turn":
        score = score + (((turnover >= 3.5) & (turnover <= 12.0)) * 0.0020)
        score = score - ((turnover > 25.0) * 0.0015)
        return score
    raise ValueError(f"unknown mode: {mode}")


def _write_signal(base: pd.DataFrame, market: dict[str, dict[str, dict]], next_date: dict[str, str], cfg: dict) -> Path:
    frame = base.copy()
    frame["base_entry_score"] = _base_score(frame)
    frame["entry_score"] = _adjusted_score(frame, str(cfg["mode"]))
    frame = frame.sort_values(["trade_date", "entry_score", "stock_code"], ascending=[True, False, True]).copy()
    upper = max(float(frame["rank_10d"].max()), 1.0)
    rows = []
    for signal_date, day in frame.groupby("trade_date", sort=True):
        buy_date = next_date.get(str(signal_date))
        if not buy_date:
            continue
        chosen = None
        for row in day.to_dict("records"):
            stock_code = str(row["stock_code"])
            buy_market = market.get(buy_date, {}).get(stock_code)
            if raw._is_st_like(buy_market) or raw._is_limit_buy(buy_market):
                continue
            chosen = row
            break
        if chosen is None:
            continue
        stock_code = str(chosen["stock_code"])
        rows.append(
            {
                "signal_date": str(signal_date),
                "buy_date": buy_date,
                "symbol": to_gm_symbol(stock_code),
                "stock_code": stock_code,
                "name": chosen.get("name"),
                "rank": 1,
                "pred_prob": raw._tailpow(float(chosen["rank_10d"]), float(cfg["gamma"]), upper),
                "entry_score": chosen["entry_score"],
                "base_entry_score": chosen["base_entry_score"],
                "pred_5d": chosen.get("pred_5d"),
                "pred_10d": chosen.get("pred_10d"),
                "rank_5d": chosen.get("rank_5d"),
                "rank_10d": chosen.get("rank_10d"),
                "amount": chosen.get("amount"),
                "turnover_rate": chosen.get("turnover_rate"),
                "total_mv": chosen.get("total_mv"),
                "atr_qfq": chosen.get("atr_qfq"),
                "target_pct": "0.99000",
                "holding_days": int(cfg["holding_days"]),
                "max_holding_days": int(cfg["holding_days"]),
                "score_exit_entry_ratio": f"{float(cfg['exit_ratio']):.5f}",
                "min_holding_days_before_score_exit": int(cfg["min_hold"]),
            }
        )
    fields = [
        "signal_date",
        "buy_date",
        "symbol",
        "stock_code",
        "name",
        "rank",
        "pred_prob",
        "entry_score",
        "base_entry_score",
        "pred_5d",
        "pred_10d",
        "rank_5d",
        "rank_10d",
        "amount",
        "turnover_rate",
        "total_mv",
        "atr_qfq",
        "target_pct",
        "holding_days",
        "max_holding_days",
        "score_exit_entry_ratio",
        "min_holding_days_before_score_exit",
    ]
    path = OUT_DIR / "signals" / f"{cfg['name']}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fields} for row in rows])
    return path


def _write_score_table(base: pd.DataFrame, cfg: dict) -> tuple[Path, str]:
    score_db = OUT_DIR / "scores_top1_soft_rank_filled.db"
    table = f"score_{cfg['name']}"
    upper = max(float(base["rank_10d"].max()), 1.0)
    payload = base[["trade_date", "stock_code", "rank_10d"]].copy()
    payload["pred_prob"] = payload["rank_10d"].map(lambda value: raw._tailpow(float(value), float(cfg["gamma"]), upper))
    payload = payload[["trade_date", "stock_code", "pred_prob"]]
    score_db.parent.mkdir(parents=True, exist_ok=True)
    import sqlite3

    conn = sqlite3.connect(score_db)
    try:
        payload.to_sql(table, conn, if_exists="replace", index=False)
        conn.execute(f'CREATE INDEX IF NOT EXISTS "idx_{table}_date_code" ON "{table}" (trade_date, stock_code)')
        conn.commit()
    finally:
        conn.close()
    return score_db, table


def _run_case(cfg: dict, signal_file: Path, score_db: Path, score_table: str, start_name: str, start: str) -> dict:
    log_file = OUT_DIR / "logs" / f"{cfg['name']}_{start_name}.log"
    indicator = raw._extract_indicator(log_file)
    if indicator is None:
        env = os.environ.copy()
        env.update(BASE_ENV)
        env["GM_SCORE_EXIT_ENTRY_RATIO"] = str(float(cfg["exit_ratio"]))
        env["GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT"] = str(int(cfg["min_hold"]))
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
            str(int(cfg["holding_days"])),
            "--max-holding-days",
            str(int(cfg["holding_days"])),
            "--target-position-pct",
            "1.0",
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
        indicator = raw._extract_indicator(log_file)
        returncode = proc.returncode
    else:
        returncode = 0
    return {
        "candidate": cfg["name"],
        "mode": cfg["mode"],
        "holding_days": cfg["holding_days"],
        "gamma": cfg["gamma"],
        "exit_ratio": cfg["exit_ratio"],
        "min_hold": cfg["min_hold"],
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
        if full["annual"] is None:
            continue
        late = [float(row["annual"]) for row in items if row["start_name"].startswith("late_") and row["annual"] is not None]
        if not late:
            continue
        summary.append(
            {
                "candidate": candidate,
                "mode": full["mode"],
                "holding_days": full["holding_days"],
                "gamma": full["gamma"],
                "exit_ratio": full["exit_ratio"],
                "min_hold": full["min_hold"],
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


def _signal_stats(signal_file: Path) -> dict:
    with signal_file.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    return {
        "signal_count": len(rows),
        "signal_days": len({row["signal_date"] for row in rows}),
        "buy_days": len({row["buy_date"] for row in rows}),
        "min_signal_date": min((row["signal_date"] for row in rows), default=""),
        "max_signal_date": max((row["signal_date"] for row in rows), default=""),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = raw._load_base()
    market = raw._market_rows()
    next_date = raw._date_map()
    rows = []
    stats = []
    total = len(CANDIDATES) * len(STARTS)
    done = 0
    for cfg in CANDIDATES:
        signal_file = _write_signal(base, market, next_date, cfg)
        stat = {"candidate": cfg["name"], **_signal_stats(signal_file)}
        stats.append(stat)
        score_db, score_table = _write_score_table(base, cfg)
        for start_name, start in STARTS:
            done += 1
            row = _run_case(cfg, signal_file, score_db, score_table, start_name, start)
            rows.append(row)
            print(
                f"[{done}/{total}] {cfg['name']} {start_name} annual={row['annual']} "
                f"sharpe={row['sharpe']} maxdd={row['max_drawdown']}",
                flush=True,
            )
            _write_rows(OUT_DIR / "soft_rank_cases.csv", rows)
    _write_rows(OUT_DIR / "soft_rank_signal_stats.csv", stats)
    summary = _summarize(rows)
    _write_rows(OUT_DIR / "soft_rank_summary.csv", summary)
    (OUT_DIR / "soft_rank_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
