from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import os
import re
import sqlite3
import subprocess
from pathlib import Path

import pandas as pd

from gm_signal_module import build_gm_signal_rows, load_market_rows_by_trade_date, write_gm_signals_csv
from selection_module import SelectionConfig


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = DATA_DIR / "reports" / "strategy_agent_goal_high_annual_20260623" / "formal_l4_grid"
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
MARKET_DB = DATA_DIR / "STOCK_DAILY_DATA.db"
STRATEGY_DIR = Path(r"D:\work\dfcf\juejin\strategy\ce327750-634e-11f1-b8d7-10ffe0295517")
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")

MANIFEST_5D = MAIN / "config" / "prediction_manifests" / "executable_5d_open_return_l4_formal_20260620.json"
MANIFEST_10D = MAIN / "config" / "prediction_manifests" / "executable_10d_open_return_l4_formal_20260617.json"
START_DATE = "20240604"
END_DATE = "20260622"
TARGET_ANNUAL = 25.672466012925824
MIN_EFFECTIVE_SIGNALS = 40
MIN_EFFECTIVE_BUY_DAYS = 30


def _load_manifest(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("approval_status") != "approved_for_l5":
        raise ValueError(f"Manifest is not approved_for_l5: {path}")
    return data


FORMAL_5D = _load_manifest(MANIFEST_5D)
FORMAL_10D = _load_manifest(MANIFEST_10D)
TABLE_5D = FORMAL_5D["table"]
TABLE_10D = FORMAL_10D["table"]


CONFIGS = [
    {
        "name": "top1_10d_q999_mv100_amt80_turn2_atr10",
        "score": "score_10d",
        "top_k": 1,
        "rank10d_min": 0.999,
        "rank5d_min": None,
        "max_total_mv": 1_000_000.0,
        "min_amount": 800_000.0,
        "min_turnover": 2.0,
        "max_atr": 0.10,
        "holding_days": 10,
        "max_holding_days": 10,
        "target_pct": 0.98,
    },
    {
        "name": "top1_10d_q9975_mv100_amt80_turn2_atr10_h10",
        "score": "score_10d",
        "top_k": 1,
        "rank10d_min": 0.9975,
        "rank5d_min": None,
        "max_total_mv": 1_000_000.0,
        "min_amount": 800_000.0,
        "min_turnover": 2.0,
        "max_atr": 0.10,
        "holding_days": 10,
        "max_holding_days": 10,
        "target_pct": 0.98,
    },
    {
        "name": "top1_10d_q9975_5d95_mv100_amt80_turn2_atr10_h10",
        "score": "score_10d",
        "top_k": 1,
        "rank10d_min": 0.9975,
        "rank5d_min": 0.95,
        "max_total_mv": 1_000_000.0,
        "min_amount": 800_000.0,
        "min_turnover": 2.0,
        "max_atr": 0.10,
        "holding_days": 10,
        "max_holding_days": 10,
        "target_pct": 0.98,
    },
    {
        "name": "top1_10d_q9975_5d90_mv200_amt50_turn1_atr12_h10",
        "score": "score_10d",
        "top_k": 1,
        "rank10d_min": 0.9975,
        "rank5d_min": 0.90,
        "max_total_mv": 2_000_000.0,
        "min_amount": 500_000.0,
        "min_turnover": 1.0,
        "max_atr": 0.12,
        "holding_days": 10,
        "max_holding_days": 10,
        "target_pct": 0.98,
    },
    {
        "name": "top1_fusion70_10d_q9975_mv100_amt80_turn2_atr10_h7",
        "score": "score_fusion70",
        "top_k": 1,
        "rank10d_min": 0.9975,
        "rank5d_min": None,
        "max_total_mv": 1_000_000.0,
        "min_amount": 800_000.0,
        "min_turnover": 2.0,
        "max_atr": 0.10,
        "holding_days": 7,
        "max_holding_days": 10,
        "target_pct": 0.98,
    },
    {
        "name": "top1_fusion50_10d_q9975_5d90_mv200_amt50_turn1_atr12_h7",
        "score": "score_fusion50",
        "top_k": 1,
        "rank10d_min": 0.9975,
        "rank5d_min": 0.90,
        "max_total_mv": 2_000_000.0,
        "min_amount": 500_000.0,
        "min_turnover": 1.0,
        "max_atr": 0.12,
        "holding_days": 7,
        "max_holding_days": 10,
        "target_pct": 0.98,
    },
    {
        "name": "top1_5d_q999_10d95_mv100_amt80_turn2_atr10_h5",
        "score": "score_5d",
        "top_k": 1,
        "rank10d_min": 0.95,
        "rank5d_min": 0.999,
        "max_total_mv": 1_000_000.0,
        "min_amount": 800_000.0,
        "min_turnover": 2.0,
        "max_atr": 0.10,
        "holding_days": 5,
        "max_holding_days": 5,
        "target_pct": 0.98,
    },
    {
        "name": "top1_5d_q9975_10d90_mv200_amt50_turn1_atr12_h5",
        "score": "score_5d",
        "top_k": 1,
        "rank10d_min": 0.90,
        "rank5d_min": 0.9975,
        "max_total_mv": 2_000_000.0,
        "min_amount": 500_000.0,
        "min_turnover": 1.0,
        "max_atr": 0.12,
        "holding_days": 5,
        "max_holding_days": 5,
        "target_pct": 0.98,
    },
    {
        "name": "top2_10d_q9975_mv200_amt50_turn1_atr12_h10",
        "score": "score_10d",
        "top_k": 2,
        "rank10d_min": 0.9975,
        "rank5d_min": None,
        "max_total_mv": 2_000_000.0,
        "min_amount": 500_000.0,
        "min_turnover": 1.0,
        "max_atr": 0.12,
        "holding_days": 10,
        "max_holding_days": 10,
        "target_pct": 0.98,
    },
    {
        "name": "top1_10d_q999_mv200_amt50_turn1_atr12_h12",
        "score": "score_10d",
        "top_k": 1,
        "rank10d_min": 0.999,
        "rank5d_min": None,
        "max_total_mv": 2_000_000.0,
        "min_amount": 500_000.0,
        "min_turnover": 1.0,
        "max_atr": 0.12,
        "holding_days": 12,
        "max_holding_days": 12,
        "target_pct": 0.98,
    },
    {
        "name": "top1_10d_q9975_mv500_amt30_turn05_noatr_h10",
        "score": "score_10d",
        "top_k": 1,
        "rank10d_min": 0.9975,
        "rank5d_min": None,
        "max_total_mv": 5_000_000.0,
        "min_amount": 300_000.0,
        "min_turnover": 0.5,
        "max_atr": None,
        "holding_days": 10,
        "max_holding_days": 10,
        "target_pct": 0.98,
    },
    {
        "name": "top1_10d_q9975_mv1000_amt30_turn05_noatr_h10",
        "score": "score_10d",
        "top_k": 1,
        "rank10d_min": 0.9975,
        "rank5d_min": None,
        "max_total_mv": 10_000_000.0,
        "min_amount": 300_000.0,
        "min_turnover": 0.5,
        "max_atr": None,
        "holding_days": 10,
        "max_holding_days": 10,
        "target_pct": 0.98,
    },
    {
        "name": "top1_10d_q995_mv500_amt30_turn05_noatr_h10",
        "score": "score_10d",
        "top_k": 1,
        "rank10d_min": 0.995,
        "rank5d_min": None,
        "max_total_mv": 5_000_000.0,
        "min_amount": 300_000.0,
        "min_turnover": 0.5,
        "max_atr": None,
        "holding_days": 10,
        "max_holding_days": 10,
        "target_pct": 0.98,
    },
    {
        "name": "top1_10d_q995_mv1000_amt50_turn1_noatr_h10",
        "score": "score_10d",
        "top_k": 1,
        "rank10d_min": 0.995,
        "rank5d_min": None,
        "max_total_mv": 10_000_000.0,
        "min_amount": 500_000.0,
        "min_turnover": 1.0,
        "max_atr": None,
        "holding_days": 10,
        "max_holding_days": 10,
        "target_pct": 0.98,
    },
    {
        "name": "top1_10d_q99_mv500_amt30_turn05_noatr_h10",
        "score": "score_10d",
        "top_k": 1,
        "rank10d_min": 0.99,
        "rank5d_min": None,
        "max_total_mv": 5_000_000.0,
        "min_amount": 300_000.0,
        "min_turnover": 0.5,
        "max_atr": None,
        "holding_days": 10,
        "max_holding_days": 10,
        "target_pct": 0.98,
    },
    {
        "name": "top2_10d_q9975_mv1000_amt30_turn05_noatr_h10",
        "score": "score_10d",
        "top_k": 2,
        "rank10d_min": 0.9975,
        "rank5d_min": None,
        "max_total_mv": 10_000_000.0,
        "min_amount": 300_000.0,
        "min_turnover": 0.5,
        "max_atr": None,
        "holding_days": 10,
        "max_holding_days": 10,
        "target_pct": 0.98,
    },
    {
        "name": "top1_fusion70_q995_5d90_mv1000_amt30_turn05_noatr_h7",
        "score": "score_fusion70",
        "top_k": 1,
        "rank10d_min": 0.995,
        "rank5d_min": 0.90,
        "max_total_mv": 10_000_000.0,
        "min_amount": 300_000.0,
        "min_turnover": 0.5,
        "max_atr": None,
        "holding_days": 7,
        "max_holding_days": 10,
        "target_pct": 0.98,
    },
    {
        "name": "top1_fusion50_q995_5d95_mv1000_amt30_turn05_noatr_h7",
        "score": "score_fusion50",
        "top_k": 1,
        "rank10d_min": 0.995,
        "rank5d_min": 0.95,
        "max_total_mv": 10_000_000.0,
        "min_amount": 300_000.0,
        "min_turnover": 0.5,
        "max_atr": None,
        "holding_days": 7,
        "max_holding_days": 10,
        "target_pct": 0.98,
    },
    {
        "name": "top1_5d_q9975_10d90_mv1000_amt30_turn05_noatr_h5",
        "score": "score_5d",
        "top_k": 1,
        "rank10d_min": 0.90,
        "rank5d_min": 0.9975,
        "max_total_mv": 10_000_000.0,
        "min_amount": 300_000.0,
        "min_turnover": 0.5,
        "max_atr": None,
        "holding_days": 5,
        "max_holding_days": 5,
        "target_pct": 0.98,
    },
    {
        "name": "top1_5d_q995_10d90_mv1000_amt30_turn05_noatr_h5",
        "score": "score_5d",
        "top_k": 1,
        "rank10d_min": 0.90,
        "rank5d_min": 0.995,
        "max_total_mv": 10_000_000.0,
        "min_amount": 300_000.0,
        "min_turnover": 0.5,
        "max_atr": None,
        "holding_days": 5,
        "max_holding_days": 5,
        "target_pct": 0.98,
    },
]

EXIT_VARIANTS = [
    {
        "suffix": "score095_mh2",
        "env": {"GM_OPEN_DAILY_SCORE_EXIT": "1", "GM_SCORE_EXIT_ENTRY_RATIO": "0.95", "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "2"},
    },
    {
        "suffix": "score098_mh2",
        "env": {"GM_OPEN_DAILY_SCORE_EXIT": "1", "GM_SCORE_EXIT_ENTRY_RATIO": "0.98", "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "2"},
    },
    {
        "suffix": "no_score_exit",
        "env": {"GM_OPEN_DAILY_SCORE_EXIT": "0"},
    },
]


def _connect_readonly(path: Path) -> sqlite3.Connection:
    uri = path.resolve().as_uri() + "?mode=ro"
    return sqlite3.connect(uri, uri=True, timeout=30)


def _load_base_frame() -> pd.DataFrame:
    with _connect_readonly(MODEL_DB) as conn:
        frame_5d = pd.read_sql_query(
            f"""
            SELECT trade_date, stock_code, pred_prob AS pred_5d
            FROM "{TABLE_5D}"
            WHERE trade_date >= ? AND trade_date <= ?
              AND pred_prob IS NOT NULL
            """,
            conn,
            params=(START_DATE, END_DATE),
        )
        frame_10d = pd.read_sql_query(
            f"""
            SELECT trade_date, stock_code, pred_prob AS pred_10d
            FROM "{TABLE_10D}"
            WHERE trade_date >= ? AND trade_date <= ?
              AND pred_prob IS NOT NULL
            """,
            conn,
            params=(START_DATE, END_DATE),
        )
    frame = frame_5d.merge(frame_10d, on=["trade_date", "stock_code"], how="inner")
    frame = frame[(frame["pred_5d"] >= 0.90) | (frame["pred_10d"] >= 0.90)].copy()
    for col in ["pred_5d", "pred_10d"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["rank_5d"] = frame["pred_5d"]
    frame["rank_10d"] = frame["pred_10d"]
    frame["score_5d"] = frame["pred_5d"]
    frame["score_10d"] = frame["pred_10d"]
    frame["score_fusion70"] = frame["pred_10d"] * 0.70 + frame["pred_5d"] * 0.30
    frame["score_fusion50"] = frame["pred_10d"] * 0.50 + frame["pred_5d"] * 0.50
    return frame


def _load_filtered_frame(cfg: dict) -> pd.DataFrame:
    where = [
        "f.trade_date >= ?",
        "f.trade_date <= ?",
        "f.pred_prob IS NOT NULL",
        "t.pred_prob IS NOT NULL",
    ]
    params: list[object] = [START_DATE, END_DATE]
    if cfg.get("rank10d_min") is not None:
        where.append("t.pred_prob >= ?")
        params.append(float(cfg["rank10d_min"]))
    if cfg.get("rank5d_min") is not None:
        where.append("f.pred_prob >= ?")
        params.append(float(cfg["rank5d_min"]))
    if cfg.get("max_total_mv") is not None:
        where.append("m.total_mv IS NOT NULL AND m.total_mv <= ?")
        params.append(float(cfg["max_total_mv"]))
    if cfg.get("min_amount") is not None:
        where.append("m.amount IS NOT NULL AND m.amount >= ?")
        params.append(float(cfg["min_amount"]))
    if cfg.get("min_turnover") is not None:
        where.append("m.turnover_rate IS NOT NULL AND m.turnover_rate >= ?")
        params.append(float(cfg["min_turnover"]))
    if cfg.get("max_atr") is not None:
        where.append("m.atr_qfq IS NOT NULL AND m.close > 0 AND (m.atr_qfq / m.close) <= ?")
        params.append(float(cfg["max_atr"]))
    where.extend(
        [
            "f.stock_code NOT LIKE '%.BJ'",
            "(m.name IS NULL OR (m.name NOT LIKE 'ST%' AND m.name NOT LIKE '*ST%' AND m.name NOT LIKE '%退市%'))",
            "(m.ST_TYPE IS NULL OR TRIM(CAST(m.ST_TYPE AS TEXT)) IN ('', '0', '0.0', 'None', 'NONE', 'nan', 'NaN'))",
            "(m.ST_TYPE_name IS NULL OR (m.ST_TYPE_name NOT LIKE '%风险警示%' AND m.ST_TYPE_name NOT LIKE '%退市风险%'))",
            "(m.limit_times IS NULL OR m.limit_times <= 0)",
        ]
    )
    with _connect_readonly(MODEL_DB) as conn:
        conn.execute(f"ATTACH DATABASE '{MARKET_DB.as_posix()}' AS market")
        frame = pd.read_sql_query(
            f"""
            SELECT
                f.trade_date,
                f.stock_code,
                f.pred_prob AS pred_5d,
                t.pred_prob AS pred_10d,
                f.pred_prob AS rank_5d,
                t.pred_prob AS rank_10d,
                m.name,
                m.open,
                m.close,
                m.pre_close,
                m.amount,
                m.turnover_rate,
                m.total_mv,
                m.atr_qfq,
                m.limit_times,
                m.ST_TYPE AS st_type,
                m.ST_TYPE_name AS st_type_name,
                NULL AS industry_encode
            FROM "{TABLE_5D}" f
            JOIN "{TABLE_10D}" t
              ON f.trade_date = t.trade_date
             AND f.stock_code = t.stock_code
            LEFT JOIN market.STOCK_DAILY_DATA m
              ON f.trade_date = m.trade_date
             AND f.stock_code = m.stock_code
            WHERE {" AND ".join(where)}
            """,
            conn,
            params=params,
        )
    for col in [
        "pred_5d",
        "pred_10d",
        "open",
        "close",
        "pre_close",
        "amount",
        "turnover_rate",
        "total_mv",
        "atr_qfq",
        "limit_times",
    ]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["score_5d"] = frame["rank_5d"]
    frame["score_10d"] = frame["rank_10d"]
    frame["score_fusion70"] = frame["rank_10d"] * 0.70 + frame["rank_5d"] * 0.30
    frame["score_fusion50"] = frame["rank_10d"] * 0.50 + frame["rank_5d"] * 0.50
    frame["atr_ratio"] = frame["atr_qfq"] / frame["close"]
    return frame


def _candidate_rows(frame: pd.DataFrame, cfg: dict) -> list[dict]:
    mask = pd.Series(True, index=frame.index)
    if cfg.get("rank10d_min") is not None:
        mask &= frame["rank_10d"] >= float(cfg["rank10d_min"])
    if cfg.get("rank5d_min") is not None:
        mask &= frame["rank_5d"] >= float(cfg["rank5d_min"])
    selected = frame[mask].copy()
    selected["pred_prob"] = selected[str(cfg["score"])]
    return selected.to_dict("records")


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
        return {"avg_invested_pct": None, "max_active_positions": None, "exposure_points": 0}
    for match in pattern.finditer(log_file.read_text(encoding="utf-8", errors="ignore")):
        values.append(float(match.group(1)))
        active.append(int(match.group(2)))
    return {
        "avg_invested_pct": (sum(values) / len(values)) if values else None,
        "max_active_positions": max(active) if active else None,
        "exposure_points": len(values),
    }


def _signal_stats(signal_file: Path) -> dict:
    if not signal_file.exists():
        return {
            "signal_count": 0,
            "buy_days": 0,
            "min_signal_date": None,
            "max_signal_date": None,
        }
    with signal_file.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    return {
        "signal_count": len(rows),
        "buy_days": len({row.get("buy_date") for row in rows if row.get("buy_date")}),
        "min_signal_date": min((row.get("signal_date") for row in rows if row.get("signal_date")), default=None),
        "max_signal_date": max((row.get("signal_date") for row in rows if row.get("signal_date")), default=None),
    }


def _empty_indicator_row(cfg: dict, exit_variant: dict, signal_file: Path, reason: str) -> dict:
    return {
        **cfg,
        "exit_variant": exit_variant["suffix"],
        "returncode": None,
        "skip_reason": reason,
        "signal_file": str(signal_file),
        "log_file": None,
        "annual": None,
        "sharpe": None,
        "max_drawdown": None,
        "pnl_ratio": None,
        "open_count": None,
        "close_count": None,
        "win_ratio": None,
        **_signal_stats(signal_file),
        "avg_invested_pct": None,
        "max_active_positions": None,
        "exposure_points": 0,
    }


def _run_backtest(cfg: dict, exit_variant: dict, signal_file: Path, log_file: Path) -> int:
    if log_file.exists() and _extract_indicator(log_file):
        return 0
    env = os.environ.copy()
    env.update(
        {
            "GM_MAX_DAILY_SELLS": "0",
            "GM_LOG_EXPOSURE": "1",
            "GM_SKIP_OPEN_LIMIT_UP_BUY": "1",
            "GM_SYNC_POSITIONS": "0",
            "GM_CASH_BUFFER": "0.98",
            "GM_VERBOSE_TRADES": "0",
            "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.0",
        }
    )
    env.update(exit_variant["env"])
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
        str(int(cfg["top_k"])),
        "--holding-days",
        str(int(cfg["holding_days"])),
        "--max-holding-days",
        str(int(cfg["max_holding_days"])),
        "--target-position-pct",
        str(float(cfg["target_pct"])),
        "--score-db",
        str(MODEL_DB),
        "--score-table",
        TABLE_10D if "10d" in str(cfg["score"]) or "fusion" in str(cfg["score"]) else TABLE_5D,
        "--market-db",
        str(MARKET_DB),
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
    return proc.returncode


def main() -> int:
    (REPORT_DIR / "signals").mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    market_rows = load_market_rows_by_trade_date(MARKET_DB, START_DATE, END_DATE)
    frame = _load_base_frame()
    results = []
    for cfg in CONFIGS:
        rows = _candidate_rows(frame, cfg)
        signal_file = REPORT_DIR / "signals" / f"{cfg['name']}.csv"
        signals = []
        if rows:
            signals = build_gm_signal_rows(
                    rows,
                    config=SelectionConfig(
                        top_k=int(cfg["top_k"]),
                        pred_col="pred_prob",
                        min_pred_prob=None,
                        min_pred_quantile=None,
                        max_atr_ratio=cfg.get("max_atr"),
                        min_amount=cfg.get("min_amount"),
                        min_turnover_rate=cfg.get("min_turnover"),
                        max_total_mv=cfg.get("max_total_mv"),
                        max_per_industry=999999,
                        exclude_bj=True,
                        exclude_st=True,
                        exclude_delisting=True,
                        exclude_current_limit=True,
                    ),
                    market_rows_by_trade_date=market_rows,
                    holding_days=int(cfg["holding_days"]),
                    max_positions=int(cfg["top_k"]),
                    weight_mode="equal",
                    target_total_pct=float(cfg["target_pct"]),
                )
        if signals:
            write_gm_signals_csv(signals, signal_file)
        signal_stats = _signal_stats(signal_file)
        if (
            int(signal_stats.get("signal_count") or 0) < MIN_EFFECTIVE_SIGNALS
            or int(signal_stats.get("buy_days") or 0) < MIN_EFFECTIVE_BUY_DAYS
        ):
            for exit_variant in EXIT_VARIANTS:
                row = _empty_indicator_row(cfg, exit_variant, signal_file, "low_signal_count")
                results.append(row)
                print(json.dumps(row, ensure_ascii=False, default=str), flush=True)
            continue
        for exit_variant in EXIT_VARIANTS:
            log_file = REPORT_DIR / "logs" / f"{cfg['name']}__{exit_variant['suffix']}.log"
            returncode = _run_backtest(cfg, exit_variant, signal_file, log_file)
            indicator = _extract_indicator(log_file) or {}
            row = {
                **cfg,
                "exit_variant": exit_variant["suffix"],
                "returncode": returncode,
                "skip_reason": None,
                "signal_file": str(signal_file),
                "log_file": str(log_file),
                "annual": indicator.get("pnl_ratio_annual"),
                "sharpe": indicator.get("sharp_ratio"),
                "max_drawdown": indicator.get("max_drawdown"),
                "pnl_ratio": indicator.get("pnl_ratio"),
                "open_count": indicator.get("open_count"),
                "close_count": indicator.get("close_count"),
                "win_ratio": indicator.get("win_ratio"),
                **signal_stats,
                **_exposure_stats(log_file),
            }
            results.append(row)
            print(json.dumps(row, ensure_ascii=False, default=str), flush=True)
    if results:
        fields = list(results[0].keys())
        for name, rows in {
            "summary.csv": results,
            "summary_by_annual.csv": sorted(results, key=lambda item: float(item.get("annual") or -999), reverse=True),
            "summary_target_over_2567.csv": [
                item for item in sorted(results, key=lambda row: float(row.get("annual") or -999), reverse=True)
                if float(item.get("annual") or -999) > TARGET_ANNUAL
                and int(item.get("signal_count") or 0) >= MIN_EFFECTIVE_SIGNALS
                and int(item.get("buy_days") or 0) >= MIN_EFFECTIVE_BUY_DAYS
            ],
        }.items():
            with (REPORT_DIR / name).open("w", encoding="utf-8-sig", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
