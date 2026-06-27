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
OUT_DIR = REPORT_ROOT / "top1_filtered_enhanced_exit_20260623"
MARKET_DB = DATA_DIR / "STOCK_DAILY_DATA.db"
STRATEGY_DIR = Path(r"D:\work\dfcf\juejin\strategy\ce327750-634e-11f1-b003-10ffe0295517")
BACKTEST_STRATEGY_DIR = Path(r"D:\work\dfcf\juejin\strategy\ce327750-634e-11f1-b8d7-10ffe0295517")
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")
BACKTEST_END = "2026-06-23 15:30:00"

STARTS = [
    ("full_20240605", "2024-06-05 09:00:00"),
    ("late_20250701", "2025-07-01 09:00:00"),
    ("late_20251009", "2025-10-09 09:00:00"),
    ("late_20260105", "2026-01-05 09:00:00"),
]

FILTERS = [
    {"name": "liq_mid_strict", "mv_min": 300000, "mv_max": 3000000, "amount_min": 120000, "turn_min": 3.5, "turn_max": 7.0, "score_min": 1.0070},
    {"name": "liq_mid_wide_mv", "mv_min": 0, "mv_max": 3000000, "amount_min": 120000, "turn_min": 3.5, "turn_max": 7.0, "score_min": 1.0070},
    {"name": "liq_small_mid", "mv_min": 300000, "mv_max": 1500000, "amount_min": 120000, "turn_min": 3.5, "turn_max": 7.0, "score_min": 0.0},
    {"name": "liq_mid_amt80", "mv_min": 300000, "mv_max": 3000000, "amount_min": 80000, "turn_min": 3.5, "turn_max": 7.0, "score_min": 1.0070},
    {"name": "liq_mid_turn10", "mv_min": 200000, "mv_max": 3000000, "amount_min": 120000, "turn_min": 3.0, "turn_max": 10.0, "score_min": 1.0070},
    {"name": "liq_only", "mv_min": 0, "mv_max": None, "amount_min": 120000, "turn_min": 3.5, "turn_max": 7.0, "score_min": 0.0},
    {"name": "liq_only_score", "mv_min": 0, "mv_max": None, "amount_min": 120000, "turn_min": 3.5, "turn_max": 7.0, "score_min": 1.0070},
    {"name": "liq_mid_score_high", "mv_min": 300000, "mv_max": 3000000, "amount_min": 120000, "turn_min": 3.5, "turn_max": 7.0, "score_min": 1.0075},
]

EXIT_VARIANTS = [
    {"suffix": "h5_g200_e098", "holding_days": 5, "gamma": 2.0, "exit_ratio": 0.98, "min_hold": 1},
    {"suffix": "h3_g200_e098", "holding_days": 3, "gamma": 2.0, "exit_ratio": 0.98, "min_hold": 1},
    {"suffix": "h5_g138_e094", "holding_days": 5, "gamma": 1.38, "exit_ratio": 0.94, "min_hold": 1},
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


def _candidate_configs() -> list[dict]:
    out = []
    for filt in FILTERS:
        for exit_cfg in EXIT_VARIANTS:
            cfg = dict(filt)
            cfg.update(exit_cfg)
            cfg["name"] = f"{filt['name']}_{exit_cfg['suffix']}"
            out.append(cfg)
    return out


def _filter_mask(frame: pd.DataFrame, cfg: dict) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    mask &= pd.to_numeric(frame["total_mv"], errors="coerce") >= float(cfg["mv_min"])
    if cfg["mv_max"] is not None:
        mask &= pd.to_numeric(frame["total_mv"], errors="coerce") <= float(cfg["mv_max"])
    mask &= pd.to_numeric(frame["amount"], errors="coerce") >= float(cfg["amount_min"])
    mask &= pd.to_numeric(frame["turnover_rate"], errors="coerce") >= float(cfg["turn_min"])
    if cfg["turn_max"] is not None:
        mask &= pd.to_numeric(frame["turnover_rate"], errors="coerce") <= float(cfg["turn_max"])
    mask &= pd.to_numeric(frame["entry_score"], errors="coerce") >= float(cfg["score_min"])
    return mask


def _top1_frame(base: pd.DataFrame) -> pd.DataFrame:
    frame = base.copy()
    frame["entry_score"] = frame["rank_10d"].astype(float) * 0.80 + frame["rank_5d"].astype(float) * 0.20
    frame = frame.sort_values(["trade_date", "entry_score", "stock_code"], ascending=[True, False, True]).copy()
    frame["rn"] = frame.groupby("trade_date").cumcount() + 1
    return frame[frame["rn"] <= 1].copy()


def _write_signal(top1: pd.DataFrame, market: dict[str, dict[str, dict]], next_date: dict[str, str], cfg: dict) -> Path:
    selected = top1[_filter_mask(top1, cfg)].copy()
    upper = max(float(top1["rank_10d"].max()), 1.0)
    selected["pred_prob"] = selected["rank_10d"].map(lambda value: raw._tailpow(float(value), float(cfg["gamma"]), upper))
    rows = []
    for row in selected.sort_values(["trade_date", "stock_code"]).to_dict("records"):
        signal_date = str(row["trade_date"])
        buy_date = next_date.get(signal_date)
        if not buy_date:
            continue
        stock_code = str(row["stock_code"])
        buy_market = market.get(buy_date, {}).get(stock_code)
        if raw._is_st_like(buy_market) or raw._is_limit_buy(buy_market):
            continue
        rows.append(
            {
                "signal_date": signal_date,
                "buy_date": buy_date,
                "symbol": to_gm_symbol(stock_code),
                "stock_code": stock_code,
                "name": row.get("name"),
                "rank": 1,
                "pred_prob": row["pred_prob"],
                "entry_score": row["entry_score"],
                "pred_5d": row.get("pred_5d"),
                "pred_10d": row.get("pred_10d"),
                "rank_5d": row.get("rank_5d"),
                "rank_10d": row.get("rank_10d"),
                "amount": row.get("amount"),
                "turnover_rate": row.get("turnover_rate"),
                "total_mv": row.get("total_mv"),
                "atr_qfq": row.get("atr_qfq"),
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
    score_db = OUT_DIR / "scores_top1_filtered_enhanced_exit.db"
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
        conn.execute(f'CREATE INDEX IF NOT EXISTS "idx_{table}_date_score" ON "{table}" (trade_date, pred_prob DESC)')
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
            str(BACKTEST_STRATEGY_DIR),
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
        "holding_days": cfg["holding_days"],
        "gamma": cfg["gamma"],
        "exit_ratio": cfg["exit_ratio"],
        "min_hold": cfg["min_hold"],
        "mv_min": cfg["mv_min"],
        "mv_max": "" if cfg["mv_max"] is None else cfg["mv_max"],
        "amount_min": cfg["amount_min"],
        "turn_min": cfg["turn_min"],
        "turn_max": "" if cfg["turn_max"] is None else cfg["turn_max"],
        "score_min": cfg["score_min"],
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
                "config_json": json.dumps({key: full[key] for key in ["mv_min", "mv_max", "amount_min", "turn_min", "turn_max", "score_min"]}, ensure_ascii=False),
            }
        )
    summary.sort(key=lambda row: (row["score"], row["full_annual"]), reverse=True)
    return summary


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = raw._load_base()
    top1 = _top1_frame(base)
    market = raw._market_rows()
    next_date = raw._date_map()
    configs = _candidate_configs()
    rows = []
    total = len(configs) * len(STARTS)
    done = 0
    for cfg in configs:
        signal_file = _write_signal(top1, market, next_date, cfg)
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
            _write_rows(OUT_DIR / "top1_filtered_cases.csv", rows)
    summary = _summarize(rows)
    _write_rows(OUT_DIR / "top1_filtered_summary.csv", summary)
    (OUT_DIR / "top1_filtered_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
