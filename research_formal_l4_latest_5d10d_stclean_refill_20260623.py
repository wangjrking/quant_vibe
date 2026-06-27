from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
from pathlib import Path

import export_formal_5d10d_l5_signals as exporter
from gm_signal_module import write_gm_signals_csv


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = (
    DATA_DIR
    / "reports"
    / "strategy_agent_goal_high_annual_20260623"
    / "latest_formal_5d10d_stclean_refill"
)
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
MARKET_DB = DATA_DIR / "STOCK_DAILY_DATA.db"
FUSION_DB = REPORT_DIR / "fusion_5d10d_latest_20260622.db"
SCORE_DB = REPORT_DIR / "latest_pool_scores.db"
TEMPLATE_STRATEGY_DIR = MAIN / "strategy_library" / "production" / "prod_formal_5d10d_gap_v20260622"
JUEJIN_STRATEGY_DIR = Path(r"D:\work\dfcf\juejin\strategy\ce327750-634e-11f1-b8d7-10ffe0295517")
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")

TABLE_5D = "stock_predict_data_model_agent_5d_gate6_balanced_bonus_20260622_executable_5d_open_return_research"
TABLE_10D = "stock_predict_data_model_agent_10d_topgate_fusion_20260622_executable_10d_open_return_research"
START_DATE = "20240604"
SIGNAL_END_DATE = "20260618"
BACKTEST_START = "2024-06-05 09:00:00"
BACKTEST_END = "2026-06-23 15:30:00"


BASE_ENV = {
    "GM_OPEN_DAILY_SCORE_EXIT": "1",
    "GM_SCORE_EXIT_ENTRY_RATIO": "1.0",
    "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "3",
    "GM_MAX_DAILY_SELLS": "1",
    "GM_STOP_LOSS_PCT": "0.08",
    "GM_TAKE_PROFIT_PCT": "none",
    "GM_LIGHT_STOP_LOSS_PCT": "none",
    "GM_SCORE_EXIT_RANK": "none",
    "GM_LOG_EXPOSURE": "1",
    "GM_SKIP_OPEN_LIMIT_UP_BUY": "1",
    "GM_SYNC_POSITIONS": "1",
    "GM_CASH_BUFFER": "0.99",
    "GM_VERBOSE_TRADES": "0",
    "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.02",
    "GM_EQUITY_DD_RISK_MODE": "1",
    "GM_EQUITY_DD_SOFT_TRIGGER": "0.10",
    "GM_EQUITY_DD_HARD_TRIGGER": "0.18",
    "GM_EQUITY_DD_RECOVER_TRIGGER": "0.05",
    "GM_EQUITY_DD_SOFT_SCALE": "0.85",
    "GM_EQUITY_DD_HARD_SCALE": "0.60",
}


def _quote_ident(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _rank_pct_sql(column: str) -> str:
    return f"""(
        CAST(RANK() OVER (PARTITION BY trade_date ORDER BY {column} ASC) AS REAL)
        + (CAST(COUNT(*) OVER (PARTITION BY trade_date, {column}) AS REAL) - 1.0) / 2.0
    ) / CAST(COUNT(*) OVER (PARTITION BY trade_date) AS REAL)"""


def _to_float(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return value


def _variant(
    name: str,
    *,
    primary_w10: float,
    fallback_w10: float,
    primary_limit: int,
    fallback_limit: int,
    primary_count_min: int,
    post_avg: float,
    rank_targets: dict[int, float],
    rank_max: int = 5,
    holding_days: int = 6,
    max_holding_days: int = 6,
    max_positions: int = 5,
    max_single: float = 0.30,
    primary_mv: float = 150000.0,
    primary_amount: float = 10000.0,
    primary_turnover: float = 0.3,
    fallback_mv: float = 200000.0,
    fallback_amount: float = 20000.0,
    fallback_turnover: float = 0.5,
    primary_max_amount: float | None = None,
    fallback_max_amount: float | None = None,
    primary_max_turnover: float | None = None,
    fallback_max_turnover: float | None = None,
    min_pred_10d: float = 0.0,
    min_hold_score_exit: int = 4,
    env: dict[str, str] | None = None,
) -> dict:
    run_env = dict(BASE_ENV)
    if env:
        run_env.update(env)
    run_env["GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT"] = str(min_hold_score_exit)
    return {
        "name": name,
        "primary_w10": primary_w10,
        "primary_w5": 1.0 - primary_w10,
        "fallback_w10": fallback_w10,
        "fallback_w5": 1.0 - fallback_w10,
        "primary_limit": primary_limit,
        "fallback_limit": fallback_limit,
        "primary_count_min": primary_count_min,
        "post_avg": post_avg,
        "rank_targets": rank_targets,
        "rank_max": rank_max,
        "holding_days": holding_days,
        "max_holding_days": max_holding_days,
        "max_positions": max_positions,
        "max_single": max_single,
        "primary_mv": primary_mv,
        "primary_amount": primary_amount,
        "primary_turnover": primary_turnover,
        "fallback_mv": fallback_mv,
        "fallback_amount": fallback_amount,
        "fallback_turnover": fallback_turnover,
        "primary_max_amount": primary_max_amount,
        "fallback_max_amount": fallback_max_amount,
        "primary_max_turnover": primary_max_turnover,
        "fallback_max_turnover": fallback_max_turnover,
        "min_pred_10d": min_pred_10d,
        "min_hold_score_exit": min_hold_score_exit,
        "env": run_env,
    }


RANK5_BALANCED = {1: 0.25, 2: 0.23, 3: 0.20, 4: 0.17, 5: 0.14}
RANK5_FULL = {1: 0.30, 2: 0.25, 3: 0.20, 4: 0.14, 5: 0.10}
RANK3_CONC = {1: 0.34, 2: 0.30, 3: 0.25}

VARIANTS = [
    _variant("rank5_10d90_5d10_h6_pc2", primary_w10=0.90, fallback_w10=0.60, primary_limit=2, fallback_limit=5, primary_count_min=2, post_avg=2.05, rank_targets=RANK5_BALANCED),
    _variant("rank5_10d80_5d20_h6_pc1", primary_w10=0.80, fallback_w10=0.80, primary_limit=2, fallback_limit=5, primary_count_min=1, post_avg=0.0, rank_targets=RANK5_BALANCED),
    _variant("rank5_10d70_5d30_h6_pc1", primary_w10=0.70, fallback_w10=0.70, primary_limit=2, fallback_limit=5, primary_count_min=1, post_avg=0.0, rank_targets=RANK5_BALANCED),
    _variant("rank5_fuller_10d80_h5", primary_w10=0.80, fallback_w10=0.80, primary_limit=3, fallback_limit=6, primary_count_min=1, post_avg=0.0, rank_targets=RANK5_FULL, holding_days=5, max_holding_days=5, max_single=0.32),
    _variant("rank3_conc_10d85_h5", primary_w10=0.85, fallback_w10=0.75, primary_limit=3, fallback_limit=4, primary_count_min=1, post_avg=0.0, rank_targets=RANK3_CONC, rank_max=3, holding_days=5, max_holding_days=5, max_positions=3, max_single=0.36),
    _variant("rank3_conc_10d70_h4", primary_w10=0.70, fallback_w10=0.70, primary_limit=3, fallback_limit=4, primary_count_min=1, post_avg=0.0, rank_targets=RANK3_CONC, rank_max=3, holding_days=4, max_holding_days=4, max_positions=3, max_single=0.36),
    _variant("rank5_liq_relaxed_h6", primary_w10=0.80, fallback_w10=0.70, primary_limit=3, fallback_limit=7, primary_count_min=1, post_avg=0.0, rank_targets=RANK5_BALANCED, primary_mv=220000.0, fallback_mv=260000.0, primary_amount=8000.0, fallback_amount=10000.0, primary_turnover=0.2, fallback_turnover=0.3),
    _variant("rank5_strict_liq_h6", primary_w10=0.85, fallback_w10=0.75, primary_limit=2, fallback_limit=5, primary_count_min=1, post_avg=0.0, rank_targets=RANK5_BALANCED, primary_mv=120000.0, fallback_mv=160000.0, primary_amount=30000.0, fallback_amount=30000.0, primary_turnover=0.8, fallback_turnover=0.8),
    _variant("top1_10d90_full_h2_nodd", primary_w10=0.90, fallback_w10=0.90, primary_limit=1, fallback_limit=2, primary_count_min=1, post_avg=0.0, rank_targets={1: 0.95}, rank_max=1, holding_days=2, max_holding_days=2, max_positions=1, max_single=0.95, primary_mv=220000.0, fallback_mv=260000.0, primary_amount=8000.0, fallback_amount=10000.0, primary_turnover=0.2, fallback_turnover=0.3, min_hold_score_exit=1, env={"GM_EQUITY_DD_RISK_MODE": "0", "GM_MAX_DAILY_SELLS": "3"}),
    _variant("top1_10d70_full_h2_nodd", primary_w10=0.70, fallback_w10=0.70, primary_limit=1, fallback_limit=2, primary_count_min=1, post_avg=0.0, rank_targets={1: 0.95}, rank_max=1, holding_days=2, max_holding_days=2, max_positions=1, max_single=0.95, primary_mv=220000.0, fallback_mv=260000.0, primary_amount=8000.0, fallback_amount=10000.0, primary_turnover=0.2, fallback_turnover=0.3, min_hold_score_exit=1, env={"GM_EQUITY_DD_RISK_MODE": "0", "GM_MAX_DAILY_SELLS": "3"}),
    _variant("top2_10d80_full_h3_nodd", primary_w10=0.80, fallback_w10=0.80, primary_limit=2, fallback_limit=3, primary_count_min=1, post_avg=0.0, rank_targets={1: 0.50, 2: 0.45}, rank_max=2, holding_days=3, max_holding_days=3, max_positions=2, max_single=0.55, primary_mv=220000.0, fallback_mv=260000.0, primary_amount=8000.0, fallback_amount=10000.0, primary_turnover=0.2, fallback_turnover=0.3, min_hold_score_exit=1, env={"GM_EQUITY_DD_RISK_MODE": "0", "GM_MAX_DAILY_SELLS": "3"}),
    _variant("top3_10d80_full_h3_nodd", primary_w10=0.80, fallback_w10=0.80, primary_limit=3, fallback_limit=4, primary_count_min=1, post_avg=0.0, rank_targets={1: 0.34, 2: 0.32, 3: 0.29}, rank_max=3, holding_days=3, max_holding_days=3, max_positions=3, max_single=0.36, primary_mv=220000.0, fallback_mv=260000.0, primary_amount=8000.0, fallback_amount=10000.0, primary_turnover=0.2, fallback_turnover=0.3, min_hold_score_exit=1, env={"GM_EQUITY_DD_RISK_MODE": "0", "GM_MAX_DAILY_SELLS": "3"}),
    _variant("top1_strict_10d90_full_h2_nodd", primary_w10=0.90, fallback_w10=0.90, primary_limit=1, fallback_limit=2, primary_count_min=1, post_avg=0.0, rank_targets={1: 0.95}, rank_max=1, holding_days=2, max_holding_days=2, max_positions=1, max_single=0.95, primary_mv=120000.0, fallback_mv=160000.0, primary_amount=30000.0, fallback_amount=30000.0, primary_turnover=0.8, fallback_turnover=0.8, min_hold_score_exit=1, env={"GM_EQUITY_DD_RISK_MODE": "0", "GM_MAX_DAILY_SELLS": "3"}),
    _variant("top3_strict_10d85_full_h3_nodd", primary_w10=0.85, fallback_w10=0.75, primary_limit=3, fallback_limit=4, primary_count_min=1, post_avg=0.0, rank_targets={1: 0.34, 2: 0.32, 3: 0.29}, rank_max=3, holding_days=3, max_holding_days=3, max_positions=3, max_single=0.36, primary_mv=120000.0, fallback_mv=160000.0, primary_amount=30000.0, fallback_amount=30000.0, primary_turnover=0.8, fallback_turnover=0.8, min_hold_score_exit=1, env={"GM_EQUITY_DD_RISK_MODE": "0", "GM_MAX_DAILY_SELLS": "3"}),
    _variant("top3_10d80_full_h2_nodd", primary_w10=0.80, fallback_w10=0.80, primary_limit=3, fallback_limit=4, primary_count_min=1, post_avg=0.0, rank_targets={1: 0.34, 2: 0.32, 3: 0.29}, rank_max=3, holding_days=2, max_holding_days=2, max_positions=3, max_single=0.36, primary_mv=220000.0, fallback_mv=260000.0, primary_amount=8000.0, fallback_amount=10000.0, primary_turnover=0.2, fallback_turnover=0.3, min_hold_score_exit=1, env={"GM_EQUITY_DD_RISK_MODE": "0", "GM_MAX_DAILY_SELLS": "3"}),
    _variant("top5_10d80_full_h2_nodd", primary_w10=0.80, fallback_w10=0.80, primary_limit=5, fallback_limit=6, primary_count_min=1, post_avg=0.0, rank_targets={1: 0.24, 2: 0.22, 3: 0.19, 4: 0.16, 5: 0.14}, rank_max=5, holding_days=2, max_holding_days=2, max_positions=5, max_single=0.26, primary_mv=220000.0, fallback_mv=260000.0, primary_amount=8000.0, fallback_amount=10000.0, primary_turnover=0.2, fallback_turnover=0.3, min_hold_score_exit=1, env={"GM_EQUITY_DD_RISK_MODE": "0", "GM_MAX_DAILY_SELLS": "5"}),
    _variant("top3_5dheavy_full_h3_nodd", primary_w10=0.40, fallback_w10=0.40, primary_limit=3, fallback_limit=4, primary_count_min=1, post_avg=0.0, rank_targets={1: 0.34, 2: 0.32, 3: 0.29}, rank_max=3, holding_days=3, max_holding_days=3, max_positions=3, max_single=0.36, primary_mv=220000.0, fallback_mv=260000.0, primary_amount=8000.0, fallback_amount=10000.0, primary_turnover=0.2, fallback_turnover=0.3, min_hold_score_exit=1, env={"GM_EQUITY_DD_RISK_MODE": "0", "GM_MAX_DAILY_SELLS": "3"}),
    _variant("top3_10d80_full_h3_noexit", primary_w10=0.80, fallback_w10=0.80, primary_limit=3, fallback_limit=4, primary_count_min=1, post_avg=0.0, rank_targets={1: 0.34, 2: 0.32, 3: 0.29}, rank_max=3, holding_days=3, max_holding_days=3, max_positions=3, max_single=0.36, primary_mv=220000.0, fallback_mv=260000.0, primary_amount=8000.0, fallback_amount=10000.0, primary_turnover=0.2, fallback_turnover=0.3, min_hold_score_exit=1, env={"GM_EQUITY_DD_RISK_MODE": "0", "GM_MAX_DAILY_SELLS": "3", "GM_OPEN_DAILY_SCORE_EXIT": "0"}),
    _variant("top5_10d80_full_h3_noexit", primary_w10=0.80, fallback_w10=0.80, primary_limit=5, fallback_limit=6, primary_count_min=1, post_avg=0.0, rank_targets={1: 0.24, 2: 0.22, 3: 0.19, 4: 0.16, 5: 0.14}, rank_max=5, holding_days=3, max_holding_days=3, max_positions=5, max_single=0.26, primary_mv=220000.0, fallback_mv=260000.0, primary_amount=8000.0, fallback_amount=10000.0, primary_turnover=0.2, fallback_turnover=0.3, min_hold_score_exit=1, env={"GM_EQUITY_DD_RISK_MODE": "0", "GM_MAX_DAILY_SELLS": "5", "GM_OPEN_DAILY_SCORE_EXIT": "0"}),
    _variant("top3_10d80_lowturn_h3_nodd", primary_w10=0.80, fallback_w10=0.80, primary_limit=3, fallback_limit=5, primary_count_min=1, post_avg=0.0, rank_targets={1: 0.34, 2: 0.32, 3: 0.29}, rank_max=3, holding_days=3, max_holding_days=3, max_positions=3, max_single=0.36, primary_mv=999999999.0, fallback_mv=999999999.0, primary_amount=10000.0, fallback_amount=10000.0, primary_turnover=0.0, fallback_turnover=0.0, primary_max_turnover=0.6, fallback_max_turnover=0.6, min_hold_score_exit=1, env={"GM_EQUITY_DD_RISK_MODE": "0", "GM_MAX_DAILY_SELLS": "3"}),
    _variant("top5_10d80_lowturn_h3_nodd", primary_w10=0.80, fallback_w10=0.80, primary_limit=5, fallback_limit=8, primary_count_min=1, post_avg=0.0, rank_targets={1: 0.24, 2: 0.22, 3: 0.19, 4: 0.16, 5: 0.14}, rank_max=5, holding_days=3, max_holding_days=3, max_positions=5, max_single=0.26, primary_mv=999999999.0, fallback_mv=999999999.0, primary_amount=10000.0, fallback_amount=10000.0, primary_turnover=0.0, fallback_turnover=0.0, primary_max_turnover=0.6, fallback_max_turnover=0.6, min_hold_score_exit=1, env={"GM_EQUITY_DD_RISK_MODE": "0", "GM_MAX_DAILY_SELLS": "5"}),
    _variant("top3_10d80_lowturn_h5_nodd", primary_w10=0.80, fallback_w10=0.80, primary_limit=3, fallback_limit=5, primary_count_min=1, post_avg=0.0, rank_targets={1: 0.34, 2: 0.32, 3: 0.29}, rank_max=3, holding_days=5, max_holding_days=5, max_positions=3, max_single=0.36, primary_mv=999999999.0, fallback_mv=999999999.0, primary_amount=10000.0, fallback_amount=10000.0, primary_turnover=0.0, fallback_turnover=0.0, primary_max_turnover=0.6, fallback_max_turnover=0.6, min_hold_score_exit=1, env={"GM_EQUITY_DD_RISK_MODE": "0", "GM_MAX_DAILY_SELLS": "3"}),
    _variant("top3_10d80_lowamt_h3_nodd", primary_w10=0.80, fallback_w10=0.80, primary_limit=3, fallback_limit=5, primary_count_min=1, post_avg=0.0, rank_targets={1: 0.34, 2: 0.32, 3: 0.29}, rank_max=3, holding_days=3, max_holding_days=3, max_positions=3, max_single=0.36, primary_mv=500000.0, fallback_mv=500000.0, primary_amount=8000.0, fallback_amount=8000.0, primary_turnover=0.0, fallback_turnover=0.0, primary_max_amount=30000.0, fallback_max_amount=30000.0, min_hold_score_exit=1, env={"GM_EQUITY_DD_RISK_MODE": "0", "GM_MAX_DAILY_SELLS": "3"}),
    _variant("top5_10d80_lowamt_h3_nodd", primary_w10=0.80, fallback_w10=0.80, primary_limit=5, fallback_limit=8, primary_count_min=1, post_avg=0.0, rank_targets={1: 0.24, 2: 0.22, 3: 0.19, 4: 0.16, 5: 0.14}, rank_max=5, holding_days=3, max_holding_days=3, max_positions=5, max_single=0.26, primary_mv=500000.0, fallback_mv=500000.0, primary_amount=8000.0, fallback_amount=8000.0, primary_turnover=0.0, fallback_turnover=0.0, primary_max_amount=30000.0, fallback_max_amount=30000.0, min_hold_score_exit=1, env={"GM_EQUITY_DD_RISK_MODE": "0", "GM_MAX_DAILY_SELLS": "5"}),
    _variant("top3_10d80_lowturn_h10_noexit", primary_w10=0.80, fallback_w10=0.80, primary_limit=3, fallback_limit=5, primary_count_min=1, post_avg=0.0, rank_targets={1: 0.34, 2: 0.32, 3: 0.29}, rank_max=3, holding_days=10, max_holding_days=10, max_positions=3, max_single=0.36, primary_mv=999999999.0, fallback_mv=999999999.0, primary_amount=10000.0, fallback_amount=10000.0, primary_turnover=0.0, fallback_turnover=0.0, primary_max_turnover=0.6, fallback_max_turnover=0.6, min_hold_score_exit=10, env={"GM_EQUITY_DD_RISK_MODE": "0", "GM_MAX_DAILY_SELLS": "3", "GM_OPEN_DAILY_SCORE_EXIT": "0"}),
    _variant("top5_10d80_lowturn_h10_noexit", primary_w10=0.80, fallback_w10=0.80, primary_limit=5, fallback_limit=8, primary_count_min=1, post_avg=0.0, rank_targets={1: 0.24, 2: 0.22, 3: 0.19, 4: 0.16, 5: 0.14}, rank_max=5, holding_days=10, max_holding_days=10, max_positions=5, max_single=0.26, primary_mv=999999999.0, fallback_mv=999999999.0, primary_amount=10000.0, fallback_amount=10000.0, primary_turnover=0.0, fallback_turnover=0.0, primary_max_turnover=0.6, fallback_max_turnover=0.6, min_hold_score_exit=10, env={"GM_EQUITY_DD_RISK_MODE": "0", "GM_MAX_DAILY_SELLS": "5", "GM_OPEN_DAILY_SCORE_EXIT": "0"}),
    _variant("top10_10d80_lowturn_h10_noexit", primary_w10=0.80, fallback_w10=0.80, primary_limit=10, fallback_limit=12, primary_count_min=1, post_avg=0.0, rank_targets={1: 0.11, 2: 0.105, 3: 0.10, 4: 0.095, 5: 0.09, 6: 0.09, 7: 0.085, 8: 0.085, 9: 0.08, 10: 0.08}, rank_max=10, holding_days=10, max_holding_days=10, max_positions=10, max_single=0.12, primary_mv=999999999.0, fallback_mv=999999999.0, primary_amount=10000.0, fallback_amount=10000.0, primary_turnover=0.0, fallback_turnover=0.0, primary_max_turnover=0.6, fallback_max_turnover=0.6, min_hold_score_exit=10, env={"GM_EQUITY_DD_RISK_MODE": "0", "GM_MAX_DAILY_SELLS": "10", "GM_OPEN_DAILY_SCORE_EXIT": "0"}),
    _variant("top1_10d80_full_h10_noexit", primary_w10=0.80, fallback_w10=0.80, primary_limit=1, fallback_limit=2, primary_count_min=1, post_avg=0.0, rank_targets={1: 0.95}, rank_max=1, holding_days=10, max_holding_days=10, max_positions=1, max_single=0.95, primary_mv=999999999.0, fallback_mv=999999999.0, primary_amount=8000.0, fallback_amount=8000.0, primary_turnover=0.0, fallback_turnover=0.0, min_hold_score_exit=10, env={"GM_EQUITY_DD_RISK_MODE": "0", "GM_MAX_DAILY_SELLS": "1", "GM_OPEN_DAILY_SCORE_EXIT": "0"}),
    _variant("top2_10d80_full_h10_noexit", primary_w10=0.80, fallback_w10=0.80, primary_limit=2, fallback_limit=3, primary_count_min=1, post_avg=0.0, rank_targets={1: 0.50, 2: 0.45}, rank_max=2, holding_days=10, max_holding_days=10, max_positions=2, max_single=0.55, primary_mv=999999999.0, fallback_mv=999999999.0, primary_amount=8000.0, fallback_amount=8000.0, primary_turnover=0.0, fallback_turnover=0.0, min_hold_score_exit=10, env={"GM_EQUITY_DD_RISK_MODE": "0", "GM_MAX_DAILY_SELLS": "2", "GM_OPEN_DAILY_SCORE_EXIT": "0"}),
    _variant("top3_10d80_full_h10_noexit", primary_w10=0.80, fallback_w10=0.80, primary_limit=3, fallback_limit=4, primary_count_min=1, post_avg=0.0, rank_targets={1: 0.34, 2: 0.32, 3: 0.29}, rank_max=3, holding_days=10, max_holding_days=10, max_positions=3, max_single=0.36, primary_mv=999999999.0, fallback_mv=999999999.0, primary_amount=8000.0, fallback_amount=8000.0, primary_turnover=0.0, fallback_turnover=0.0, min_hold_score_exit=10, env={"GM_EQUITY_DD_RISK_MODE": "0", "GM_MAX_DAILY_SELLS": "3", "GM_OPEN_DAILY_SCORE_EXIT": "0"}),
    _variant("top5_10d80_full_h10_noexit", primary_w10=0.80, fallback_w10=0.80, primary_limit=5, fallback_limit=6, primary_count_min=1, post_avg=0.0, rank_targets={1: 0.24, 2: 0.22, 3: 0.19, 4: 0.16, 5: 0.14}, rank_max=5, holding_days=10, max_holding_days=10, max_positions=5, max_single=0.26, primary_mv=999999999.0, fallback_mv=999999999.0, primary_amount=8000.0, fallback_amount=8000.0, primary_turnover=0.0, fallback_turnover=0.0, min_hold_score_exit=10, env={"GM_EQUITY_DD_RISK_MODE": "0", "GM_MAX_DAILY_SELLS": "5", "GM_OPEN_DAILY_SCORE_EXIT": "0"}),
    _variant("top3_10d100_full_h10_noexit", primary_w10=1.00, fallback_w10=1.00, primary_limit=3, fallback_limit=4, primary_count_min=1, post_avg=0.0, rank_targets={1: 0.34, 2: 0.32, 3: 0.29}, rank_max=3, holding_days=10, max_holding_days=10, max_positions=3, max_single=0.36, primary_mv=999999999.0, fallback_mv=999999999.0, primary_amount=8000.0, fallback_amount=8000.0, primary_turnover=0.0, fallback_turnover=0.0, min_hold_score_exit=10, env={"GM_EQUITY_DD_RISK_MODE": "0", "GM_MAX_DAILY_SELLS": "3", "GM_OPEN_DAILY_SCORE_EXIT": "0"}),
    _variant("top5_10d100_full_h10_noexit", primary_w10=1.00, fallback_w10=1.00, primary_limit=5, fallback_limit=6, primary_count_min=1, post_avg=0.0, rank_targets={1: 0.24, 2: 0.22, 3: 0.19, 4: 0.16, 5: 0.14}, rank_max=5, holding_days=10, max_holding_days=10, max_positions=5, max_single=0.26, primary_mv=999999999.0, fallback_mv=999999999.0, primary_amount=8000.0, fallback_amount=8000.0, primary_turnover=0.0, fallback_turnover=0.0, min_hold_score_exit=10, env={"GM_EQUITY_DD_RISK_MODE": "0", "GM_MAX_DAILY_SELLS": "5", "GM_OPEN_DAILY_SCORE_EXIT": "0"}),
]


def build_latest_fusion() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if FUSION_DB.exists():
        conn_check = None
        try:
            conn_check = sqlite3.connect(FUSION_DB)
            row = conn_check.execute(
                "SELECT COUNT(*), MAX(trade_date) FROM fusion_rank_base"
            ).fetchone()
            if row and int(row[0] or 0) > 0 and str(row[1] or "") >= SIGNAL_END_DATE:
                return
        except sqlite3.Error:
            pass
        finally:
            if conn_check is not None:
                conn_check.close()
        print(f"rebuilding incomplete fusion db: {FUSION_DB}", flush=True)
        FUSION_DB.unlink()
    print(f"building fusion db: {FUSION_DB}", flush=True)
    conn = sqlite3.connect(FUSION_DB)
    try:
        conn.execute("ATTACH DATABASE ? AS model", (str(MODEL_DB),))
        conn.execute("ATTACH DATABASE ? AS market", (str(MARKET_DB),))
        t5 = _quote_ident(TABLE_5D)
        t10 = _quote_ident(TABLE_10D)
        print("copying source subsets into local fusion db", flush=True)
        conn.executescript(
            f"""
            DROP TABLE IF EXISTS local_10d;
            DROP TABLE IF EXISTS local_5d;
            DROP TABLE IF EXISTS local_market;

            CREATE TABLE local_10d AS
            SELECT
                trade_date,
                stock_code,
                pred_prob,
                "10d_yield_rate",
                executable_10d_open_return,
                close_rate
            FROM model.{t10}
            WHERE trade_date >= '{START_DATE}' AND trade_date <= '20260622'
              AND pred_prob IS NOT NULL;

            CREATE TABLE local_5d AS
            SELECT trade_date, stock_code, pred_prob
            FROM model.{t5}
            WHERE trade_date >= '{START_DATE}' AND trade_date <= '20260622'
              AND pred_prob IS NOT NULL;

            CREATE TABLE local_market AS
            SELECT
                trade_date,
                stock_code,
                name,
                pre_close,
                open,
                close,
                amount,
                turnover_rate,
                turnover_rate_f,
                total_mv,
                circ_mv,
                volume_ratio,
                atr_qfq,
                limit_times,
                ST_TYPE AS st_type,
                ST_TYPE_name AS st_type_name
            FROM market.STOCK_DAILY_DATA
            WHERE trade_date >= '{START_DATE}' AND trade_date <= '20260622';

            CREATE INDEX idx_local_10d_trade_stock ON local_10d(trade_date, stock_code);
            CREATE INDEX idx_local_5d_trade_stock ON local_5d(trade_date, stock_code);
            CREATE INDEX idx_local_market_trade_stock ON local_market(trade_date, stock_code);
            """
        )
        conn.commit()
        print("creating fusion table", flush=True)
        conn.executescript(
            f"""
            DROP TABLE IF EXISTS fusion_rank_base;
            CREATE TABLE fusion_rank_base AS
            SELECT
                d10.trade_date,
                d10.stock_code,
                d10.pred_prob AS pred_10d,
                d5.pred_prob AS pred_5d,
                d10."10d_yield_rate" AS "10d_yield_rate",
                d10.executable_10d_open_return,
                m.name,
                m.pre_close,
                m.open,
                m.close,
                m.amount,
                m.turnover_rate,
                m.turnover_rate_f,
                m.total_mv,
                m.circ_mv,
                m.volume_ratio,
                m.atr_qfq,
                d10.close_rate,
                m.limit_times,
                m.ST_TYPE AS st_type,
                m.ST_TYPE_name AS st_type_name,
                d5.pred_prob AS rank_5d,
                d10.pred_prob AS rank_10d
            FROM local_10d d10
            INNER JOIN local_5d d5
                ON d10.trade_date = d5.trade_date
               AND d10.stock_code = d5.stock_code
            LEFT JOIN local_market m
                ON d10.trade_date = m.trade_date
               AND d10.stock_code = m.stock_code
            WHERE d10.pred_prob IS NOT NULL
              AND d5.pred_prob IS NOT NULL;
            CREATE INDEX idx_fusion_rank_base_trade_stock ON fusion_rank_base(trade_date, stock_code);
            CREATE INDEX idx_fusion_rank_base_trade_rank10d ON fusion_rank_base(trade_date, rank_10d);
            """
        )
        stats = conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT trade_date), COUNT(DISTINCT stock_code), MIN(trade_date), MAX(trade_date) FROM fusion_rank_base"
        ).fetchone()
        manifest = {
            "model_db": str(MODEL_DB),
            "market_db": str(MARKET_DB),
            "table_5d": TABLE_5D,
            "table_10d": TABLE_10D,
            "fusion_db": str(FUSION_DB),
            "row_count": int(stats[0] or 0),
            "trade_days": int(stats[1] or 0),
            "stock_count": int(stats[2] or 0),
            "min_trade_date": stats[3],
            "max_trade_date": stats[4],
            "no_industry_filter": True,
            "no_1d_score_separate_use": True,
        }
        (REPORT_DIR / "fusion_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        conn.commit()
        print(f"fusion built: rows={manifest['row_count']} max_trade_date={manifest['max_trade_date']}", flush=True)
    finally:
        conn.close()


def _pool_formula(w10: float, w5: float) -> str:
    return f"(rank_10d * {w10:.12g}) + (rank_5d * {w5:.12g})"


def _score_table_name(name: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")
    return f"latest_stclean_{clean}"


def _tradable_sql(prefix: str = "") -> str:
    p = prefix
    return f"""
        {p}stock_code NOT LIKE '%.BJ'
        AND COALESCE({p}name, '') NOT LIKE 'ST%'
        AND COALESCE({p}name, '') NOT LIKE '*ST%'
        AND COALESCE({p}name, '') NOT LIKE '%退市%'
        AND COALESCE({p}name, '') NOT LIKE '退%'
        AND COALESCE({p}name, '') NOT LIKE '%退'
        AND ({p}st_type IS NULL OR {p}st_type = '' OR {p}st_type = 'None' OR UPPER(CAST({p}st_type AS TEXT)) IN ('0', '0.0', 'FALSE', 'NONE', 'NAN'))
        AND COALESCE({p}st_type_name, '') NOT LIKE '%风险%'
        AND ({p}limit_times IS NULL OR {p}limit_times = '' OR {p}limit_times = 'None' OR CAST({p}limit_times AS REAL) = 0)
        AND {p}rank_5d IS NOT NULL
        AND {p}rank_10d IS NOT NULL
        AND {p}close IS NOT NULL
    """


def build_score_table(cfg: dict) -> str:
    table = _score_table_name(cfg["name"])
    SCORE_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(SCORE_DB)
    try:
        conn.execute("ATTACH DATABASE ? AS fusion", (str(FUSION_DB),))
        primary_formula = _pool_formula(float(cfg["primary_w10"]), float(cfg["primary_w5"]))
        fallback_formula = _pool_formula(float(cfg["fallback_w10"]), float(cfg["fallback_w5"]))
        primary = f"""
            {_tradable_sql()}
            AND total_mv IS NOT NULL AND total_mv <= {float(cfg["primary_mv"]):.12g}
            AND amount IS NOT NULL AND amount >= {float(cfg["primary_amount"]):.12g}
            AND turnover_rate IS NOT NULL AND turnover_rate >= {float(cfg["primary_turnover"]):.12g}
        """
        fallback = f"""
            {_tradable_sql()}
            AND total_mv IS NOT NULL AND total_mv <= {float(cfg["fallback_mv"]):.12g}
            AND amount IS NOT NULL AND amount >= {float(cfg["fallback_amount"]):.12g}
            AND turnover_rate IS NOT NULL AND turnover_rate >= {float(cfg["fallback_turnover"]):.12g}
        """
        if cfg.get("primary_max_amount") is not None:
            primary += f"\n            AND amount <= {float(cfg['primary_max_amount']):.12g}"
        if cfg.get("fallback_max_amount") is not None:
            fallback += f"\n            AND amount <= {float(cfg['fallback_max_amount']):.12g}"
        if cfg.get("primary_max_turnover") is not None:
            primary += f"\n            AND turnover_rate <= {float(cfg['primary_max_turnover']):.12g}"
        if cfg.get("fallback_max_turnover") is not None:
            fallback += f"\n            AND turnover_rate <= {float(cfg['fallback_max_turnover']):.12g}"
        conn.executescript(
            f"""
            DROP TABLE IF EXISTS {_quote_ident(table)};
            CREATE TABLE {_quote_ident(table)} AS
            SELECT
                trade_date,
                stock_code,
                CASE
                    WHEN {primary} THEN 2.0 + ({primary_formula})
                    WHEN {fallback} THEN 1.0 + ({fallback_formula})
                    ELSE ({fallback_formula})
                END AS pred_prob
            FROM fusion.fusion_rank_base;
            CREATE INDEX idx_{table}_trade_stock ON {_quote_ident(table)}(trade_date, stock_code);
            CREATE INDEX idx_{table}_trade_pred ON {_quote_ident(table)}(trade_date, pred_prob DESC);
            """
        )
        conn.commit()
    finally:
        conn.close()
    return table


def strategy_dir_for(cfg: dict, score_table: str) -> Path:
    out = REPORT_DIR / "strategy_dirs" / str(cfg["name"])
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(TEMPLATE_STRATEGY_DIR / "strategy_manifest.json", out / "strategy_manifest.json")
    rules = json.loads((TEMPLATE_STRATEGY_DIR / "trading_rules.json").read_text(encoding="utf-8"))
    rules["score_rule"]["score_asset"]["db_path"] = str(SCORE_DB)
    rules["score_rule"]["score_asset"]["table"] = score_table
    rules["score_rule"]["fusion_asset"]["db_path"] = str(FUSION_DB)
    pool_rule = rules["score_rule"]["pool_generation_rule"]
    pool_rule["base_target_pct"] = 0.10925
    pool_rule["primary_pool"].update(
        {
            "rank_weight_10d": float(cfg["primary_w10"]),
            "rank_weight_5d": float(cfg["primary_w5"]),
            "limit": int(cfg["primary_limit"]),
            "max_total_mv": float(cfg["primary_mv"]),
            "min_amount": float(cfg["primary_amount"]),
            "min_turnover_rate": float(cfg["primary_turnover"]),
        }
    )
    for key in ("normal_fallback_pool", "weak_fallback_pool"):
        pool_rule[key].update(
            {
                "rank_weight_10d": float(cfg["fallback_w10"]),
                "rank_weight_5d": float(cfg["fallback_w5"]),
                "limit": int(cfg["fallback_limit"]),
                "max_total_mv": float(cfg["fallback_mv"]),
                "min_amount": float(cfg["fallback_amount"]),
                "min_turnover_rate": float(cfg["fallback_turnover"]),
            }
        )
    rules["selection_rule"]["primary_count_min"] = int(cfg["primary_count_min"])
    rules["selection_rule"]["rank_max"] = int(cfg["rank_max"])
    rules["selection_rule"]["post_filter_avg_pred_min"] = float(cfg["post_avg"])
    rules["selection_rule"]["min_pred_10d"] = float(cfg["min_pred_10d"])
    rules["position_rule"]["max_positions"] = int(cfg["max_positions"])
    rules["position_rule"]["max_single_position_pct"] = float(cfg["max_single"])
    rules["position_rule"]["rank_target_pct"] = {str(key): value for key, value in cfg["rank_targets"].items()}
    rules["holding_rule"]["holding_days"] = int(cfg["holding_days"])
    rules["holding_rule"]["max_holding_days"] = int(cfg["max_holding_days"])
    rules["holding_rule"]["min_holding_days_before_score_exit"] = int(cfg["min_hold_score_exit"])
    (out / "trading_rules.json").write_text(json.dumps(rules, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def write_signal(cfg: dict, strategy_dir: Path, signal_file: Path) -> dict:
    rows = exporter._build_signals(strategy_dir, START_DATE, SIGNAL_END_DATE, str(MARKET_DB), None)
    signal_file.parent.mkdir(parents=True, exist_ok=True)
    write_gm_signals_csv(rows, signal_file)
    day_sums: dict[str, float] = {}
    for row in rows:
        day = str(row.get("buy_date") or "")
        day_sums[day] = day_sums.get(day, 0.0) + (_to_float(row.get("target_pct"), 0.0) or 0.0)
    return {
        "signal_count": len(rows),
        "buy_days": len(day_sums),
        "avg_day_target_sum": sum(day_sums.values()) / len(day_sums) if day_sums else None,
        "min_day_target_sum": min(day_sums.values()) if day_sums else None,
        "max_day_target_sum": max(day_sums.values()) if day_sums else None,
    }


def extract_indicator(log_file: Path) -> dict | None:
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


def exposure_stats(log_file: Path) -> dict:
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


def run_backtest(cfg: dict, signal_file: Path, log_file: Path, score_table: str) -> int:
    if log_file.exists() and extract_indicator(log_file):
        return 0
    env = os.environ.copy()
    env.update({str(key): str(value) for key, value in cfg["env"].items()})
    command = [
        str(JUEJIN_PYTHON),
        str(MAIN / "run_juejin_signal_backtest.py"),
        "--strategy-dir",
        str(JUEJIN_STRATEGY_DIR),
        "--signal-file",
        str(signal_file),
        "--log-file",
        str(log_file),
        "--max-positions",
        str(cfg["max_positions"]),
        "--holding-days",
        str(cfg["holding_days"]),
        "--max-holding-days",
        str(cfg["max_holding_days"]),
        "--target-position-pct",
        str(cfg["max_single"]),
        "--score-db",
        str(SCORE_DB),
        "--score-table",
        score_table,
        "--market-db",
        str(MARKET_DB),
        "--backtest-start",
        BACKTEST_START,
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
    return proc.returncode


def write_rows(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def metric(row: dict, key: str) -> float:
    return _to_float(row.get(key), float("-inf"))


def objective(row: dict) -> float:
    return min(metric(row, "annual") / 25.6725, metric(row, "sharpe") / 1.50, metric(row, "buy_days") / 30.0)


def main() -> int:
    build_latest_fusion()
    (REPORT_DIR / "signals").mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    results = read_rows(REPORT_DIR / "summary.csv")
    completed = {str(row.get("name")) for row in results if row.get("name")}
    for index, cfg in enumerate(VARIANTS, start=1):
        if cfg["name"] in completed and (REPORT_DIR / "logs" / f"{cfg['name']}.log").exists():
            print(f"[{index}/{len(VARIANTS)}] {cfg['name']} skipped_existing", flush=True)
            continue
        score_table = build_score_table(cfg)
        strategy_dir = strategy_dir_for(cfg, score_table)
        signal_file = REPORT_DIR / "signals" / f"{cfg['name']}.csv"
        log_file = REPORT_DIR / "logs" / f"{cfg['name']}.log"
        signal_stats = write_signal(cfg, strategy_dir, signal_file)
        returncode = run_backtest(cfg, signal_file, log_file, score_table)
        indicator = extract_indicator(log_file) or {}
        row = {
            "name": cfg["name"],
            "returncode": returncode,
            "annual": indicator.get("pnl_ratio_annual", indicator.get("annual_return")),
            "sharpe": indicator.get("sharp_ratio", indicator.get("sharpe_ratio")),
            "max_drawdown": indicator.get("max_drawdown"),
            "win_ratio": indicator.get("win_ratio"),
            "open_count": indicator.get("open_count"),
            "close_count": indicator.get("close_count"),
            "score_table": score_table,
            "config_json": json.dumps({key: value for key, value in cfg.items() if key != "env"}, ensure_ascii=False, sort_keys=True),
            "env_json": json.dumps(cfg["env"], ensure_ascii=False, sort_keys=True),
            "signal_file": str(signal_file),
            "log_file": str(log_file),
        }
        row.update(signal_stats)
        row.update(exposure_stats(log_file))
        results.append(row)
        write_rows(REPORT_DIR / "summary.csv", results)
        print(f"[{index}/{len(VARIANTS)}] {cfg['name']} annual={row.get('annual')} sharpe={row.get('sharpe')} buy_days={row.get('buy_days')}", flush=True)
    write_rows(REPORT_DIR / "summary_by_objective.csv", sorted(results, key=objective, reverse=True))
    write_rows(REPORT_DIR / "summary_by_annual.csv", sorted(results, key=lambda row: metric(row, "annual"), reverse=True))
    write_rows(REPORT_DIR / "summary_target_hits.csv", [row for row in results if metric(row, "annual") >= 25.6725 and metric(row, "buy_days") >= 30.0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
