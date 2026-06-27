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

from gm_signal_module import build_gm_signal_rows, load_market_rows_by_trade_date, write_gm_signals_csv
from selection_module import SelectionConfig


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
DATA_DIR = ROOT / "quant" / "data_file"
SOURCE_REPORT_DIR = DATA_DIR / "reports" / "strategy_agent_model_application_20260622" / "formal_5d10d_gate6_retest_20260622"
FUSION_DB = SOURCE_REPORT_DIR / "formal_5d10d_gate6_fusion_scores.db"
MARKET_DB = DATA_DIR / "STOCK_DAILY_DATA.db"
REPORT_DIR = DATA_DIR / "reports" / "strategy_agent_model_application_20260622" / "formal_5d10d_gate6_direct_grid_20260622"
STRATEGY_DIR = Path(r"D:\work\dfcf\juejin\strategy\ce327750-634e-11f1-b8d7-10ffe0295517")
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")
START_DATE = "20240604"
END_DATE = "20260618"


BASE_ENV = {
    "GM_OPEN_DAILY_SCORE_EXIT": "0",
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
    "GM_SCORE_CONTINUE_ENTRY_RATIO": "none",
}


FORMULAS = {
    "rank5d": "rank_5d",
    "rank10d": "rank_10d",
    "r5_80_r10_20": "(rank_5d * 0.80 + rank_10d * 0.20)",
    "r5_90_r10_10": "(rank_5d * 0.90 + rank_10d * 0.10)",
    "r5_70_r10_30": "(rank_5d * 0.70 + rank_10d * 0.30)",
    "r5_60_r10_40": "(rank_5d * 0.60 + rank_10d * 0.40)",
    "r10_60_r5_40": "(rank_10d * 0.60 + rank_5d * 0.40)",
    "r10_80_r5_20": "(rank_10d * 0.80 + rank_5d * 0.20)",
    "consensus_min": "MIN(rank_5d, rank_10d)",
    "any_max": "MAX(rank_5d, rank_10d)",
}

SCORE_TABLES = {
    name: f"score_{name}"
    for name in FORMULAS
}


def _to_float(value: Any, default: float | None = None) -> float | None:
    if value in (None, "", "None"):
        return default
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return value


def _safe(value: Any) -> str:
    return str(value).replace(".", "p").replace("-", "neg").replace("/", "_").replace("None", "none")


def _variant(name: str, **kwargs: Any) -> dict[str, Any]:
    base = {
        "name": name,
        "formula": "rank5d",
        "top_k": 5,
        "daily_pool": 40,
        "holding_days": 5,
        "max_holding_days": None,
        "max_positions": 5,
        "target_total_pct": 0.98,
        "target_position_pct": 0.196,
        "weight_mode": "equal",
        "score_floor": None,
        "rank5d_min": None,
        "rank10d_min": None,
        "amount_min": 10000.0,
        "amount_max": None,
        "turnover_min": 0.3,
        "turnover_max": None,
        "total_mv_min": None,
        "total_mv_max": 200000.0,
        "volume_ratio_min": None,
        "volume_ratio_max": None,
        "close_rate_min": 0.965,
        "close_rate_max": 1.085,
        "atr_required": False,
        "score_table": None,
        "layers": None,
        "boost_min_daily_target_pct": None,
        "boost_target_cap_pct": None,
        "env": dict(BASE_ENV),
    }
    base.update(kwargs)
    base["env"] = {**BASE_ENV, **dict(kwargs.get("env") or {})}
    return base


def _layer(name: str, **kwargs: Any) -> dict[str, Any]:
    layer = _variant(name, **kwargs)
    layer.pop("layers", None)
    return layer


VARIANTS = [
    _variant("rank5d_top5_h5", formula="rank5d", holding_days=5),
    _variant("rank5d_top5_h6", formula="rank5d", holding_days=6),
    _variant("rank5d_top3_h5", formula="rank5d", top_k=3, max_positions=3, target_position_pct=0.32667),
    _variant("rank5d_liq_h5", formula="rank5d", amount_min=100000.0, turnover_min=1.0, total_mv_min=20000.0),
    _variant("rank10d_top5_h5", formula="rank10d", holding_days=5),
    _variant("rank10d_top5_h6", formula="rank10d", holding_days=6),
    _variant("confirm10d_r5min45_mv180_cr965_1085", formula="rank10d", rank5d_min=0.45, top_k=5, max_positions=5, target_total_pct=0.98, target_position_pct=0.196, holding_days=5, total_mv_max=180000.0, close_rate_min=0.965, close_rate_max=1.085),
    _variant("confirm10d_r5min45_mv200_cr965_1085", formula="rank10d", rank5d_min=0.45, top_k=5, max_positions=5, target_total_pct=0.98, target_position_pct=0.196, holding_days=5, total_mv_max=200000.0, close_rate_min=0.965, close_rate_max=1.085),
    _variant("confirm10d_r5min45_mv250_cr965_1085", formula="rank10d", rank5d_min=0.45, top_k=5, max_positions=5, target_total_pct=0.98, target_position_pct=0.196, holding_days=5, total_mv_max=250000.0, close_rate_min=0.965, close_rate_max=1.085),
    _variant("confirm10d_r5min50_mv180_cr965_1085", formula="rank10d", rank5d_min=0.50, top_k=5, max_positions=5, target_total_pct=0.98, target_position_pct=0.196, holding_days=5, total_mv_max=180000.0, close_rate_min=0.965, close_rate_max=1.085),
    _variant("confirm10d_r5min50_mv200_cr965_1085", formula="rank10d", rank5d_min=0.50, top_k=5, max_positions=5, target_total_pct=0.98, target_position_pct=0.196, holding_days=5, total_mv_max=200000.0, close_rate_min=0.965, close_rate_max=1.085),
    _variant("confirm10d_r5min50_mv250_cr965_1085", formula="rank10d", rank5d_min=0.50, top_k=5, max_positions=5, target_total_pct=0.98, target_position_pct=0.196, holding_days=5, total_mv_max=250000.0, close_rate_min=0.965, close_rate_max=1.085),
    _variant("confirm10d_r5min55_mv180_cr965_1085", formula="rank10d", rank5d_min=0.55, top_k=5, max_positions=5, target_total_pct=0.98, target_position_pct=0.196, holding_days=5, total_mv_max=180000.0, close_rate_min=0.965, close_rate_max=1.085),
    _variant("confirm10d_r5min55_mv200_cr965_1085", formula="rank10d", rank5d_min=0.55, top_k=5, max_positions=5, target_total_pct=0.98, target_position_pct=0.196, holding_days=5, total_mv_max=200000.0, close_rate_min=0.965, close_rate_max=1.085),
    _variant("confirm10d_r5min55_mv250_cr965_1085", formula="rank10d", rank5d_min=0.55, top_k=5, max_positions=5, target_total_pct=0.98, target_position_pct=0.196, holding_days=5, total_mv_max=250000.0, close_rate_min=0.965, close_rate_max=1.085),
    _variant("confirm10d_r5min60_mv180_cr965_1085", formula="rank10d", rank5d_min=0.60, top_k=5, max_positions=5, target_total_pct=0.98, target_position_pct=0.196, holding_days=5, total_mv_max=180000.0, close_rate_min=0.965, close_rate_max=1.085),
    _variant("confirm10d_r5min60_mv200_cr965_1085", formula="rank10d", rank5d_min=0.60, top_k=5, max_positions=5, target_total_pct=0.98, target_position_pct=0.196, holding_days=5, total_mv_max=200000.0, close_rate_min=0.965, close_rate_max=1.085),
    _variant("confirm10d_r5min60_mv250_cr965_1085", formula="rank10d", rank5d_min=0.60, top_k=5, max_positions=5, target_total_pct=0.98, target_position_pct=0.196, holding_days=5, total_mv_max=250000.0, close_rate_min=0.965, close_rate_max=1.085),
    _variant("confirm10d_r5min50_mv200_cr980_1060", formula="rank10d", rank5d_min=0.50, top_k=5, max_positions=5, target_total_pct=0.98, target_position_pct=0.196, holding_days=5, total_mv_max=200000.0, close_rate_min=0.98, close_rate_max=1.06),
    _variant("confirm10d_r5min55_mv200_cr980_1060", formula="rank10d", rank5d_min=0.55, top_k=5, max_positions=5, target_total_pct=0.98, target_position_pct=0.196, holding_days=5, total_mv_max=200000.0, close_rate_min=0.98, close_rate_max=1.06),
    _variant("confirm10d_r5min60_mv200_cr980_1060", formula="rank10d", rank5d_min=0.60, top_k=5, max_positions=5, target_total_pct=0.98, target_position_pct=0.196, holding_days=5, total_mv_max=200000.0, close_rate_min=0.98, close_rate_max=1.06),
    _variant("r5_80_r10_20_h5", formula="r5_80_r10_20", holding_days=5),
    _variant("r5_60_r10_40_h5", formula="r5_60_r10_40", holding_days=5),
    _variant("r10_60_r5_40_h5", formula="r10_60_r5_40", holding_days=5),
    _variant("r10_80_r5_20_h5", formula="r10_80_r5_20", holding_days=5),
    _variant("consensus_min_h5", formula="consensus_min", holding_days=5),
    _variant("any_max_h5", formula="any_max", holding_days=5),
    _variant("rank5d_floor995", formula="rank5d", score_floor=0.995, holding_days=5),
    _variant("rank5d_floor990_top3", formula="rank5d", score_floor=0.990, top_k=3, max_positions=3, target_position_pct=0.32667, holding_days=5),
    _variant("r5_80_floor990_top3", formula="r5_80_r10_20", score_floor=0.990, top_k=3, max_positions=3, target_position_pct=0.32667, holding_days=5),
    _variant("consensus_floor970", formula="consensus_min", score_floor=0.970, holding_days=5),
    _variant("rank5d_h4", formula="rank5d", holding_days=4),
    _variant("rank5d_h7", formula="rank5d", holding_days=7),
    _variant("rank5d_no_stop", formula="rank5d", holding_days=5, env={"GM_STOP_LOSS_PCT": "none"}),
    _variant("rank5d_eqdd", formula="rank5d", holding_days=5, env={"GM_EQUITY_DD_RISK_MODE": "1", "GM_EQUITY_DD_SOFT_TRIGGER": "0.10", "GM_EQUITY_DD_HARD_TRIGGER": "0.18", "GM_EQUITY_DD_SOFT_SCALE": "0.85", "GM_EQUITY_DD_HARD_SCALE": "0.60", "GM_EQUITY_DD_RECOVER_TRIGGER": "0.05"}),
    _variant("r5_80_floor990_top3_h4", formula="r5_80_r10_20", score_floor=0.990, top_k=3, max_positions=3, target_position_pct=0.32667, holding_days=4),
    _variant("r5_80_floor990_top3_h6", formula="r5_80_r10_20", score_floor=0.990, top_k=3, max_positions=3, target_position_pct=0.32667, holding_days=6),
    _variant("r5_80_floor990_top3_h7", formula="r5_80_r10_20", score_floor=0.990, top_k=3, max_positions=3, target_position_pct=0.32667, holding_days=7),
    _variant("r5_80_floor992_top3", formula="r5_80_r10_20", score_floor=0.992, top_k=3, max_positions=3, target_position_pct=0.32667, holding_days=5),
    _variant("r5_80_floor995_top3", formula="r5_80_r10_20", score_floor=0.995, top_k=3, max_positions=3, target_position_pct=0.32667, holding_days=5),
    _variant("r5_80_floor985_top3", formula="r5_80_r10_20", score_floor=0.985, top_k=3, max_positions=3, target_position_pct=0.32667, holding_days=5),
    _variant("r5_80_floor990_top2", formula="r5_80_r10_20", score_floor=0.990, top_k=2, max_positions=2, target_position_pct=0.49, holding_days=5),
    _variant("r5_80_floor990_top4", formula="r5_80_r10_20", score_floor=0.990, top_k=4, max_positions=4, target_position_pct=0.245, holding_days=5),
    _variant("r5_80_floor990_top3_eqdd", formula="r5_80_r10_20", score_floor=0.990, top_k=3, max_positions=3, target_position_pct=0.32667, holding_days=5, env={"GM_EQUITY_DD_RISK_MODE": "1", "GM_EQUITY_DD_SOFT_TRIGGER": "0.10", "GM_EQUITY_DD_HARD_TRIGGER": "0.18", "GM_EQUITY_DD_SOFT_SCALE": "0.85", "GM_EQUITY_DD_HARD_SCALE": "0.60", "GM_EQUITY_DD_RECOVER_TRIGGER": "0.05"}),
    _variant("r5_80_floor990_top3_strictdd", formula="r5_80_r10_20", score_floor=0.990, top_k=3, max_positions=3, target_position_pct=0.32667, holding_days=5, env={"GM_EQUITY_DD_RISK_MODE": "1", "GM_EQUITY_DD_SOFT_TRIGGER": "0.08", "GM_EQUITY_DD_HARD_TRIGGER": "0.14", "GM_EQUITY_DD_SOFT_SCALE": "0.75", "GM_EQUITY_DD_HARD_SCALE": "0.45", "GM_EQUITY_DD_RECOVER_TRIGGER": "0.04"}),
    _variant("r5_80_floor990_top3_sl06", formula="r5_80_r10_20", score_floor=0.990, top_k=3, max_positions=3, target_position_pct=0.32667, holding_days=5, env={"GM_STOP_LOSS_PCT": "0.06"}),
    _variant("r5_80_floor990_top3_tp20", formula="r5_80_r10_20", score_floor=0.990, top_k=3, max_positions=3, target_position_pct=0.32667, holding_days=5, env={"GM_TAKE_PROFIT_PCT": "0.20"}),
    _variant("r5_80_floor990_top3_light05", formula="r5_80_r10_20", score_floor=0.990, top_k=3, max_positions=3, target_position_pct=0.32667, holding_days=5, env={"GM_LIGHT_STOP_LOSS_PCT": "0.05", "GM_MIN_HOLDING_DAYS_BEFORE_LIGHT_STOP": "1"}),
    _variant("consensus_floor970_h4", formula="consensus_min", score_floor=0.970, holding_days=4),
    _variant("consensus_floor970_top3", formula="consensus_min", score_floor=0.970, top_k=3, max_positions=3, target_position_pct=0.32667, holding_days=5),
    _variant("r5_80_floor990_top2_exit095_mh1", formula="r5_80_r10_20", score_floor=0.990, top_k=2, max_positions=2, target_position_pct=0.49, holding_days=5, env={"GM_OPEN_DAILY_SCORE_EXIT": "1", "GM_SCORE_EXIT_ENTRY_RATIO": "0.95", "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "1"}),
    _variant("r5_80_floor990_top2_exit100_mh1", formula="r5_80_r10_20", score_floor=0.990, top_k=2, max_positions=2, target_position_pct=0.49, holding_days=5, env={"GM_OPEN_DAILY_SCORE_EXIT": "1", "GM_SCORE_EXIT_ENTRY_RATIO": "1.00", "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "1"}),
    _variant("r5_80_floor990_top2_exit105_mh1", formula="r5_80_r10_20", score_floor=0.990, top_k=2, max_positions=2, target_position_pct=0.49, holding_days=5, env={"GM_OPEN_DAILY_SCORE_EXIT": "1", "GM_SCORE_EXIT_ENTRY_RATIO": "1.05", "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "1"}),
    _variant("r5_80_floor990_top2_daydrop90", formula="r5_80_r10_20", score_floor=0.990, top_k=2, max_positions=2, target_position_pct=0.49, holding_days=5, env={"GM_OPEN_DAILY_SCORE_EXIT": "1", "GM_SCORE_STOP_LOSS_DAY_DROP_RATIO": "0.90", "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "1"}),
    _variant("r5_80_floor990_top3_exit095_mh1", formula="r5_80_r10_20", score_floor=0.990, top_k=3, max_positions=3, target_position_pct=0.32667, holding_days=5, env={"GM_OPEN_DAILY_SCORE_EXIT": "1", "GM_SCORE_EXIT_ENTRY_RATIO": "0.95", "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "1"}),
    _variant("r5_80_floor990_top3_exit100_mh1", formula="r5_80_r10_20", score_floor=0.990, top_k=3, max_positions=3, target_position_pct=0.32667, holding_days=5, env={"GM_OPEN_DAILY_SCORE_EXIT": "1", "GM_SCORE_EXIT_ENTRY_RATIO": "1.00", "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "1"}),
    _variant("r5_80_floor990_top3_daydrop90", formula="r5_80_r10_20", score_floor=0.990, top_k=3, max_positions=3, target_position_pct=0.32667, holding_days=5, env={"GM_OPEN_DAILY_SCORE_EXIT": "1", "GM_SCORE_STOP_LOSS_DAY_DROP_RATIO": "0.90", "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "1"}),
    _variant("r5_80_floor992_top3_exit100_mh1", formula="r5_80_r10_20", score_floor=0.992, top_k=3, max_positions=3, target_position_pct=0.32667, holding_days=5, env={"GM_OPEN_DAILY_SCORE_EXIT": "1", "GM_SCORE_EXIT_ENTRY_RATIO": "1.00", "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "1"}),
    _variant("rank5d_floor990_top3_exit100_mh1", formula="rank5d", score_floor=0.990, top_k=3, max_positions=3, target_position_pct=0.32667, holding_days=5, env={"GM_OPEN_DAILY_SCORE_EXIT": "1", "GM_SCORE_EXIT_ENTRY_RATIO": "1.00", "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "1"}),
    _variant("consensus_floor970_exit100_mh1", formula="consensus_min", score_floor=0.970, holding_days=5, env={"GM_OPEN_DAILY_SCORE_EXIT": "1", "GM_SCORE_EXIT_ENTRY_RATIO": "1.00", "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "1"}),
    _variant("r5_80_floor990_top3_mv30_150", formula="r5_80_r10_20", score_floor=0.990, top_k=3, max_positions=3, target_position_pct=0.32667, holding_days=5, total_mv_min=30000.0, total_mv_max=150000.0),
    _variant("r5_80_floor990_top3_mv50_200", formula="r5_80_r10_20", score_floor=0.990, top_k=3, max_positions=3, target_position_pct=0.32667, holding_days=5, total_mv_min=50000.0, total_mv_max=200000.0),
    _variant("r5_80_floor990_top3_turn1_12", formula="r5_80_r10_20", score_floor=0.990, top_k=3, max_positions=3, target_position_pct=0.32667, holding_days=5, turnover_min=1.0, turnover_max=12.0),
    _variant("r5_80_floor990_top3_turn2_18", formula="r5_80_r10_20", score_floor=0.990, top_k=3, max_positions=3, target_position_pct=0.32667, holding_days=5, turnover_min=2.0, turnover_max=18.0),
    _variant("r5_80_floor990_top3_amt50m", formula="r5_80_r10_20", score_floor=0.990, top_k=3, max_positions=3, target_position_pct=0.32667, holding_days=5, amount_min=50000.0),
    _variant("r5_80_floor990_top3_amt100m", formula="r5_80_r10_20", score_floor=0.990, top_k=3, max_positions=3, target_position_pct=0.32667, holding_days=5, amount_min=100000.0),
    _variant("r5_80_floor990_top3_close98_106", formula="r5_80_r10_20", score_floor=0.990, top_k=3, max_positions=3, target_position_pct=0.32667, holding_days=5, close_rate_min=0.98, close_rate_max=1.06),
    _variant("r5_80_floor990_top3_atr", formula="r5_80_r10_20", score_floor=0.990, top_k=3, max_positions=3, target_position_pct=0.32667, holding_days=5, atr_required=True),
    _variant("r5_80_floor990_top4_close98_106", formula="r5_80_r10_20", score_floor=0.990, top_k=4, max_positions=4, target_position_pct=0.245, holding_days=5, close_rate_min=0.98, close_rate_max=1.06),
    _variant("r5_80_floor990_top5_close98_106", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_position_pct=0.196, holding_days=5, close_rate_min=0.98, close_rate_max=1.06),
    _variant("r5_80_floor990_top3_close99_105", formula="r5_80_r10_20", score_floor=0.990, top_k=3, max_positions=3, target_position_pct=0.32667, holding_days=5, close_rate_min=0.99, close_rate_max=1.05),
    _variant("r5_80_floor990_top4_close99_105", formula="r5_80_r10_20", score_floor=0.990, top_k=4, max_positions=4, target_position_pct=0.245, holding_days=5, close_rate_min=0.99, close_rate_max=1.05),
    _variant("r5_80_floor990_top5_close99_105", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_position_pct=0.196, holding_days=5, close_rate_min=0.99, close_rate_max=1.05),
    _variant("r5_80_floor990_top4_turn1_12", formula="r5_80_r10_20", score_floor=0.990, top_k=4, max_positions=4, target_position_pct=0.245, holding_days=5, turnover_min=1.0, turnover_max=12.0),
    _variant("r5_80_floor990_top5_turn1_12", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_position_pct=0.196, holding_days=5, turnover_min=1.0, turnover_max=12.0),
    _variant("r5_80_floor990_top4_close98_106_turn1_12", formula="r5_80_r10_20", score_floor=0.990, top_k=4, max_positions=4, target_position_pct=0.245, holding_days=5, close_rate_min=0.98, close_rate_max=1.06, turnover_min=1.0, turnover_max=12.0),
    _variant("r5_80_floor990_top5_close98_106_turn1_12", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_position_pct=0.196, holding_days=5, close_rate_min=0.98, close_rate_max=1.06, turnover_min=1.0, turnover_max=12.0),
    _variant("r5_80_floor990_top3_close98_106_strictdd", formula="r5_80_r10_20", score_floor=0.990, top_k=3, max_positions=3, target_position_pct=0.32667, holding_days=5, close_rate_min=0.98, close_rate_max=1.06, env={"GM_EQUITY_DD_RISK_MODE": "1", "GM_EQUITY_DD_SOFT_TRIGGER": "0.08", "GM_EQUITY_DD_HARD_TRIGGER": "0.14", "GM_EQUITY_DD_SOFT_SCALE": "0.75", "GM_EQUITY_DD_HARD_SCALE": "0.45", "GM_EQUITY_DD_RECOVER_TRIGGER": "0.04"}),
    _variant("r5_80_floor990_top3_close98_106_exit100", formula="r5_80_r10_20", score_floor=0.990, top_k=3, max_positions=3, target_position_pct=0.32667, holding_days=5, close_rate_min=0.98, close_rate_max=1.06, env={"GM_OPEN_DAILY_SCORE_EXIT": "1", "GM_SCORE_EXIT_ENTRY_RATIO": "1.00", "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "1"}),
    _variant("r5_80_floor990_top3_close98_106_daydrop90", formula="r5_80_r10_20", score_floor=0.990, top_k=3, max_positions=3, target_position_pct=0.32667, holding_days=5, close_rate_min=0.98, close_rate_max=1.06, env={"GM_OPEN_DAILY_SCORE_EXIT": "1", "GM_SCORE_STOP_LOSS_DAY_DROP_RATIO": "0.90", "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "1"}),
    _variant("r5_80_floor990_top3_close98_106_t90", formula="r5_80_r10_20", score_floor=0.990, top_k=3, max_positions=3, target_total_pct=0.90, target_position_pct=0.30, holding_days=5, close_rate_min=0.98, close_rate_max=1.06),
    _variant("r5_80_floor990_top3_close98_106_t85", formula="r5_80_r10_20", score_floor=0.990, top_k=3, max_positions=3, target_total_pct=0.85, target_position_pct=0.28333, holding_days=5, close_rate_min=0.98, close_rate_max=1.06),
    _variant("r5_90_floor990_top3_close98_106", formula="r5_90_r10_10", score_floor=0.990, top_k=3, max_positions=3, target_position_pct=0.32667, holding_days=5, close_rate_min=0.98, close_rate_max=1.06),
    _variant("r5_70_floor990_top3_close98_106", formula="r5_70_r10_30", score_floor=0.990, top_k=3, max_positions=3, target_position_pct=0.32667, holding_days=5, close_rate_min=0.98, close_rate_max=1.06),
    _variant("r5_90_floor990_top4_close98_106", formula="r5_90_r10_10", score_floor=0.990, top_k=4, max_positions=4, target_position_pct=0.245, holding_days=5, close_rate_min=0.98, close_rate_max=1.06),
    _variant("r5_70_floor990_top4_close98_106", formula="r5_70_r10_30", score_floor=0.990, top_k=4, max_positions=4, target_position_pct=0.245, holding_days=5, close_rate_min=0.98, close_rate_max=1.06),
    _variant("r5_80_floor990_top5_close99_105_t100", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.00, target_position_pct=0.20, holding_days=5, close_rate_min=0.99, close_rate_max=1.05),
    _variant("r5_80_floor990_top5_close99_105_t102", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.02, target_position_pct=0.204, holding_days=5, close_rate_min=0.99, close_rate_max=1.05),
    _variant("r5_80_floor990_top5_close99_105_h4", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_position_pct=0.196, holding_days=4, close_rate_min=0.99, close_rate_max=1.05),
    _variant("r5_80_floor990_top5_close99_105_h6", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_position_pct=0.196, holding_days=6, close_rate_min=0.99, close_rate_max=1.05),
    _variant("r5_80_floor988_top5_close99_105", formula="r5_80_r10_20", score_floor=0.988, top_k=5, max_positions=5, target_position_pct=0.196, holding_days=5, close_rate_min=0.99, close_rate_max=1.05),
    _variant("r5_80_floor992_top5_close99_105", formula="r5_80_r10_20", score_floor=0.992, top_k=5, max_positions=5, target_position_pct=0.196, holding_days=5, close_rate_min=0.99, close_rate_max=1.05),
    _variant("r5_80_floor990_top5_close985_1055", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_position_pct=0.196, holding_days=5, close_rate_min=0.985, close_rate_max=1.055),
    _variant("r5_80_floor990_top5_close995_105", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_position_pct=0.196, holding_days=5, close_rate_min=0.995, close_rate_max=1.05),
    _variant("r5_80_floor990_top5_close99_104", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_position_pct=0.196, holding_days=5, close_rate_min=0.99, close_rate_max=1.04),
    _variant("r5_80_floor990_top5_close99_105_turn05_12", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_position_pct=0.196, holding_days=5, close_rate_min=0.99, close_rate_max=1.05, turnover_min=0.5, turnover_max=12.0),
    _variant("r5_80_floor990_top5_close99_105_turn1_12", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_position_pct=0.196, holding_days=5, close_rate_min=0.99, close_rate_max=1.05, turnover_min=1.0, turnover_max=12.0),
    _variant("r5_80_floor990_top5_close99_105_mv30_200", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_position_pct=0.196, holding_days=5, close_rate_min=0.99, close_rate_max=1.05, total_mv_min=30000.0, total_mv_max=200000.0),
    _variant("r5_80_floor990_top5_close99_105_exit100", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_position_pct=0.196, holding_days=5, close_rate_min=0.99, close_rate_max=1.05, env={"GM_OPEN_DAILY_SCORE_EXIT": "1", "GM_SCORE_EXIT_ENTRY_RATIO": "1.00", "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "1"}),
    _variant("r5_80_floor990_top5_close99_105_strictdd", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_position_pct=0.196, holding_days=5, close_rate_min=0.99, close_rate_max=1.05, env={"GM_EQUITY_DD_RISK_MODE": "1", "GM_EQUITY_DD_SOFT_TRIGGER": "0.08", "GM_EQUITY_DD_HARD_TRIGGER": "0.14", "GM_EQUITY_DD_SOFT_SCALE": "0.75", "GM_EQUITY_DD_HARD_SCALE": "0.45", "GM_EQUITY_DD_RECOVER_TRIGGER": "0.04"}),
    _variant("r5_80_floor990_top5_close995_105_t120", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.20, target_position_pct=0.24, holding_days=5, close_rate_min=0.995, close_rate_max=1.05),
    _variant("r5_80_floor990_top5_close995_105_t140", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.40, target_position_pct=0.28, holding_days=5, close_rate_min=0.995, close_rate_max=1.05),
    _variant("r5_80_floor990_top5_close995_105_t160", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.60, target_position_pct=0.32, holding_days=5, close_rate_min=0.995, close_rate_max=1.05),
    _variant("r5_80_floor990_top5_close995_105_h6", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_position_pct=0.196, holding_days=6, close_rate_min=0.995, close_rate_max=1.05),
    _variant("r5_80_floor990_top5_close995_105_h7", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_position_pct=0.196, holding_days=7, close_rate_min=0.995, close_rate_max=1.05),
    _variant("r5_80_floor990_top5_close995_105_t120_h6", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.20, target_position_pct=0.24, holding_days=6, close_rate_min=0.995, close_rate_max=1.05),
    _variant("r5_80_floor990_top5_close995_105_t140_h6", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.40, target_position_pct=0.28, holding_days=6, close_rate_min=0.995, close_rate_max=1.05),
    _variant("r5_80_floor990_top5_close995_105_t120_exit100", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.20, target_position_pct=0.24, holding_days=5, close_rate_min=0.995, close_rate_max=1.05, env={"GM_OPEN_DAILY_SCORE_EXIT": "1", "GM_SCORE_EXIT_ENTRY_RATIO": "1.00", "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "1"}),
    _variant("r5_80_floor990_top5_close995_105_t120_strictdd", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.20, target_position_pct=0.24, holding_days=5, close_rate_min=0.995, close_rate_max=1.05, env={"GM_EQUITY_DD_RISK_MODE": "1", "GM_EQUITY_DD_SOFT_TRIGGER": "0.08", "GM_EQUITY_DD_HARD_TRIGGER": "0.14", "GM_EQUITY_DD_SOFT_SCALE": "0.75", "GM_EQUITY_DD_HARD_SCALE": "0.45", "GM_EQUITY_DD_RECOVER_TRIGGER": "0.04"}),
    _variant("r5_80_floor990_top4_close995_105_t120", formula="r5_80_r10_20", score_floor=0.990, top_k=4, max_positions=4, target_total_pct=1.20, target_position_pct=0.30, holding_days=5, close_rate_min=0.995, close_rate_max=1.05),
    _variant("r5_80_floor990_top3_close995_105_t120", formula="r5_80_r10_20", score_floor=0.990, top_k=3, max_positions=3, target_total_pct=1.20, target_position_pct=0.40, holding_days=5, close_rate_min=0.995, close_rate_max=1.05),
    _variant("hybrid995_core70_fill99_30_max8", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.20, holding_days=5, layers=[
        _layer("core995", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.70, holding_days=5, close_rate_min=0.995, close_rate_max=1.05),
        _layer("fill99", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.30, holding_days=5, close_rate_min=0.99, close_rate_max=1.05),
    ]),
    _variant("hybrid995_core80_fill99_20_max8", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.20, holding_days=5, layers=[
        _layer("core995", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.80, holding_days=5, close_rate_min=0.995, close_rate_max=1.05),
        _layer("fill99", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.20, holding_days=5, close_rate_min=0.99, close_rate_max=1.05),
    ]),
    _variant("hybrid995_core60_fill99_40_max8", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.20, holding_days=5, layers=[
        _layer("core995", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.60, holding_days=5, close_rate_min=0.995, close_rate_max=1.05),
        _layer("fill99", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.40, holding_days=5, close_rate_min=0.99, close_rate_max=1.05),
    ]),
    _variant("hybrid995_core70_fill985_30_max8", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.20, holding_days=5, layers=[
        _layer("core995", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.70, holding_days=5, close_rate_min=0.995, close_rate_max=1.05),
        _layer("fill985", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.30, holding_days=5, close_rate_min=0.985, close_rate_max=1.055),
    ]),
    _variant("hybrid995_core80_fill985_20_max8", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.20, holding_days=5, layers=[
        _layer("core995", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.80, holding_days=5, close_rate_min=0.995, close_rate_max=1.05),
        _layer("fill985", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.20, holding_days=5, close_rate_min=0.985, close_rate_max=1.055),
    ]),
    _variant("hybrid995_core70_fill99turn_30_max8", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.20, holding_days=5, layers=[
        _layer("core995", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.70, holding_days=5, close_rate_min=0.995, close_rate_max=1.05),
        _layer("fill99turn", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.30, holding_days=5, close_rate_min=0.99, close_rate_max=1.05, turnover_min=1.0, turnover_max=12.0),
    ]),
    _variant("hybrid995top3_core70_fill99_30_max6", formula="r5_80_r10_20", top_k=6, max_positions=6, target_position_pct=0.24, holding_days=5, layers=[
        _layer("core995top3", formula="r5_80_r10_20", score_floor=0.990, top_k=3, max_positions=3, target_total_pct=0.70, holding_days=5, close_rate_min=0.995, close_rate_max=1.05),
        _layer("fill99", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.30, holding_days=5, close_rate_min=0.99, close_rate_max=1.05),
    ]),
    _variant("hybrid995top3_core60_fill99_40_max8", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.24, holding_days=5, layers=[
        _layer("core995top3", formula="r5_80_r10_20", score_floor=0.990, top_k=3, max_positions=3, target_total_pct=0.60, holding_days=5, close_rate_min=0.995, close_rate_max=1.05),
        _layer("fill99", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.40, holding_days=5, close_rate_min=0.99, close_rate_max=1.05),
    ]),
    _variant("hybrid99_core80_fill98_20_max8", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.20, holding_days=5, layers=[
        _layer("core99", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.80, holding_days=5, close_rate_min=0.99, close_rate_max=1.05),
        _layer("fill98", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.20, holding_days=5, close_rate_min=0.98, close_rate_max=1.06),
    ]),
    _variant("hybrid99_core70_fill98_30_max8", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.20, holding_days=5, layers=[
        _layer("core99", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.70, holding_days=5, close_rate_min=0.99, close_rate_max=1.05),
        _layer("fill98", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.30, holding_days=5, close_rate_min=0.98, close_rate_max=1.06),
    ]),
    _variant("hybrid995_core120_fill985_40_max8", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.30, holding_days=5, layers=[
        _layer("core995", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.20, holding_days=5, close_rate_min=0.995, close_rate_max=1.05),
        _layer("fill985", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.40, holding_days=5, close_rate_min=0.985, close_rate_max=1.055),
    ]),
    _variant("hybrid995_core140_fill985_40_max8", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.32, holding_days=5, layers=[
        _layer("core995", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.40, holding_days=5, close_rate_min=0.995, close_rate_max=1.05),
        _layer("fill985", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.40, holding_days=5, close_rate_min=0.985, close_rate_max=1.055),
    ]),
    _variant("hybrid995_core120_fill99_40_max8", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.30, holding_days=5, layers=[
        _layer("core995", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.20, holding_days=5, close_rate_min=0.995, close_rate_max=1.05),
        _layer("fill99", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.40, holding_days=5, close_rate_min=0.99, close_rate_max=1.05),
    ]),
    _variant("hybrid995_core120_fill99turn_40_max8", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.30, holding_days=5, layers=[
        _layer("core995", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.20, holding_days=5, close_rate_min=0.995, close_rate_max=1.05),
        _layer("fill99turn", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.40, holding_days=5, close_rate_min=0.99, close_rate_max=1.05, turnover_min=1.0, turnover_max=12.0),
    ]),
    _variant("hybrid995top3_core100_fill99_40_max8", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.36, holding_days=5, layers=[
        _layer("core995top3", formula="r5_80_r10_20", score_floor=0.990, top_k=3, max_positions=3, target_total_pct=1.00, holding_days=5, close_rate_min=0.995, close_rate_max=1.05),
        _layer("fill99", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.40, holding_days=5, close_rate_min=0.99, close_rate_max=1.05),
    ]),
    _variant("hybrid995top3_core90_fill985_30_max6", formula="r5_80_r10_20", top_k=6, max_positions=6, target_position_pct=0.34, holding_days=5, layers=[
        _layer("core995top3", formula="r5_80_r10_20", score_floor=0.990, top_k=3, max_positions=3, target_total_pct=0.90, holding_days=5, close_rate_min=0.995, close_rate_max=1.05),
        _layer("fill985", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.30, holding_days=5, close_rate_min=0.985, close_rate_max=1.055),
    ]),
    _variant("hybrid99_core100_fill98_30_max8", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.26, holding_days=5, layers=[
        _layer("core99", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.00, holding_days=5, close_rate_min=0.99, close_rate_max=1.05),
        _layer("fill98", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.30, holding_days=5, close_rate_min=0.98, close_rate_max=1.06),
    ]),
    _variant("hybrid99_core110_fill98_40_max8", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.28, holding_days=5, layers=[
        _layer("core99", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.10, holding_days=5, close_rate_min=0.99, close_rate_max=1.05),
        _layer("fill98", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.40, holding_days=5, close_rate_min=0.98, close_rate_max=1.06),
    ]),
    _variant("hybrid995_core120_fill99turn_60_max8", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.30, holding_days=5, layers=[
        _layer("core995", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.20, holding_days=5, close_rate_min=0.995, close_rate_max=1.05),
        _layer("fill99turn", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.60, holding_days=5, close_rate_min=0.99, close_rate_max=1.05, turnover_min=1.0, turnover_max=12.0),
    ]),
    _variant("hybrid995_core120_fill99turn_80_max10", formula="r5_80_r10_20", top_k=10, max_positions=10, target_position_pct=0.30, holding_days=5, layers=[
        _layer("core995", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.20, holding_days=5, close_rate_min=0.995, close_rate_max=1.05),
        _layer("fill99turn", formula="r5_80_r10_20", score_floor=0.990, top_k=8, max_positions=8, target_total_pct=0.80, holding_days=5, close_rate_min=0.99, close_rate_max=1.05, turnover_min=1.0, turnover_max=12.0),
    ]),
    _variant("hybrid995_core120_fill99turn_40_h6_max8", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.30, holding_days=6, layers=[
        _layer("core995", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.20, holding_days=6, close_rate_min=0.995, close_rate_max=1.05),
        _layer("fill99turn", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.40, holding_days=6, close_rate_min=0.99, close_rate_max=1.05, turnover_min=1.0, turnover_max=12.0),
    ]),
    _variant("hybrid995_core120_fill99turn_40_h7_max8", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.30, holding_days=7, layers=[
        _layer("core995", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.20, holding_days=7, close_rate_min=0.995, close_rate_max=1.05),
        _layer("fill99turn", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.40, holding_days=7, close_rate_min=0.99, close_rate_max=1.05, turnover_min=1.0, turnover_max=12.0),
    ]),
    _variant("hybrid995_core120_fill985turn_40_max10", formula="r5_80_r10_20", top_k=10, max_positions=10, target_position_pct=0.30, holding_days=5, layers=[
        _layer("core995", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.20, holding_days=5, close_rate_min=0.995, close_rate_max=1.05),
        _layer("fill985turn", formula="r5_80_r10_20", score_floor=0.990, top_k=8, max_positions=8, target_total_pct=0.40, holding_days=5, close_rate_min=0.985, close_rate_max=1.055, turnover_min=1.0, turnover_max=12.0),
    ]),
    _variant("hybrid995_core120_fill98turn_40_max10", formula="r5_80_r10_20", top_k=10, max_positions=10, target_position_pct=0.30, holding_days=5, layers=[
        _layer("core995", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.20, holding_days=5, close_rate_min=0.995, close_rate_max=1.05),
        _layer("fill98turn", formula="r5_80_r10_20", score_floor=0.990, top_k=8, max_positions=8, target_total_pct=0.40, holding_days=5, close_rate_min=0.98, close_rate_max=1.06, turnover_min=1.0, turnover_max=12.0),
    ]),
    _variant("hybrid995_core120_fill99turn_40_max10", formula="r5_80_r10_20", top_k=10, max_positions=10, target_position_pct=0.30, holding_days=5, layers=[
        _layer("core995", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.20, holding_days=5, close_rate_min=0.995, close_rate_max=1.05),
        _layer("fill99turn", formula="r5_80_r10_20", score_floor=0.990, top_k=8, max_positions=8, target_total_pct=0.40, holding_days=5, close_rate_min=0.99, close_rate_max=1.05, turnover_min=1.0, turnover_max=12.0),
    ]),
    _variant("hybrid995_core140_fill99turn_40_max8", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.32, holding_days=5, layers=[
        _layer("core995", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.40, holding_days=5, close_rate_min=0.995, close_rate_max=1.05),
        _layer("fill99turn", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.40, holding_days=5, close_rate_min=0.99, close_rate_max=1.05, turnover_min=1.0, turnover_max=12.0),
    ]),
    _variant("hybrid995_core120_fill99turn_40_max8_sl06", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.30, holding_days=5, env={"GM_STOP_LOSS_PCT": "0.06"}, layers=[
        _layer("core995", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.20, holding_days=5, close_rate_min=0.995, close_rate_max=1.05),
        _layer("fill99turn", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.40, holding_days=5, close_rate_min=0.99, close_rate_max=1.05, turnover_min=1.0, turnover_max=12.0),
    ]),
    _variant("hybrid995_core120_fill99turn_40_max8_sl10", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.30, holding_days=5, env={"GM_STOP_LOSS_PCT": "0.10"}, layers=[
        _layer("core995", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.20, holding_days=5, close_rate_min=0.995, close_rate_max=1.05),
        _layer("fill99turn", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.40, holding_days=5, close_rate_min=0.99, close_rate_max=1.05, turnover_min=1.0, turnover_max=12.0),
    ]),
    _variant("hybrid995_core120_fill99turn_40_max8_nosl", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.30, holding_days=5, env={"GM_STOP_LOSS_PCT": "none"}, layers=[
        _layer("core995", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.20, holding_days=5, close_rate_min=0.995, close_rate_max=1.05),
        _layer("fill99turn", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.40, holding_days=5, close_rate_min=0.99, close_rate_max=1.05, turnover_min=1.0, turnover_max=12.0),
    ]),
    _variant("hybrid995_core120_fill99turn_40_max8_strictdd", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.30, holding_days=5, env={"GM_EQUITY_DD_RISK_MODE": "1", "GM_EQUITY_DD_SOFT_TRIGGER": "0.08", "GM_EQUITY_DD_HARD_TRIGGER": "0.14", "GM_EQUITY_DD_SOFT_SCALE": "0.75", "GM_EQUITY_DD_HARD_SCALE": "0.45", "GM_EQUITY_DD_RECOVER_TRIGGER": "0.04"}, layers=[
        _layer("core995", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.20, holding_days=5, close_rate_min=0.995, close_rate_max=1.05),
        _layer("fill99turn", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.40, holding_days=5, close_rate_min=0.99, close_rate_max=1.05, turnover_min=1.0, turnover_max=12.0),
    ]),
    _variant("hybrid99_core110_fill98_40_max8_sl06", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.28, holding_days=5, env={"GM_STOP_LOSS_PCT": "0.06"}, layers=[
        _layer("core99", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.10, holding_days=5, close_rate_min=0.99, close_rate_max=1.05),
        _layer("fill98", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.40, holding_days=5, close_rate_min=0.98, close_rate_max=1.06),
    ]),
    _variant("hybrid99_core110_fill98_40_max8_sl10", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.28, holding_days=5, env={"GM_STOP_LOSS_PCT": "0.10"}, layers=[
        _layer("core99", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.10, holding_days=5, close_rate_min=0.99, close_rate_max=1.05),
        _layer("fill98", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.40, holding_days=5, close_rate_min=0.98, close_rate_max=1.06),
    ]),
    _variant("hybrid99_core110_fill98_40_max8_strictdd", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.28, holding_days=5, env={"GM_EQUITY_DD_RISK_MODE": "1", "GM_EQUITY_DD_SOFT_TRIGGER": "0.08", "GM_EQUITY_DD_HARD_TRIGGER": "0.14", "GM_EQUITY_DD_SOFT_SCALE": "0.75", "GM_EQUITY_DD_HARD_SCALE": "0.45", "GM_EQUITY_DD_RECOVER_TRIGGER": "0.04"}, layers=[
        _layer("core99", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.10, holding_days=5, close_rate_min=0.99, close_rate_max=1.05),
        _layer("fill98", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.40, holding_days=5, close_rate_min=0.98, close_rate_max=1.06),
    ]),
    _variant("hybrid995_core120_fill99turn_40_max8_sells0", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.30, holding_days=5, env={"GM_MAX_DAILY_SELLS": "0"}, layers=[
        _layer("core995", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.20, holding_days=5, close_rate_min=0.995, close_rate_max=1.05),
        _layer("fill99turn", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.40, holding_days=5, close_rate_min=0.99, close_rate_max=1.05, turnover_min=1.0, turnover_max=12.0),
    ]),
    _variant("hybrid995_core120_fill99turn_40_max8_sells2", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.30, holding_days=5, env={"GM_MAX_DAILY_SELLS": "2"}, layers=[
        _layer("core995", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.20, holding_days=5, close_rate_min=0.995, close_rate_max=1.05),
        _layer("fill99turn", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.40, holding_days=5, close_rate_min=0.99, close_rate_max=1.05, turnover_min=1.0, turnover_max=12.0),
    ]),
    _variant("hybrid995_core120_fill99turn_40_max8_continue098_mh8", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.30, holding_days=5, max_holding_days=8, env={"GM_SCORE_CONTINUE_ENTRY_RATIO": "0.98"}, layers=[
        _layer("core995", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.20, holding_days=5, close_rate_min=0.995, close_rate_max=1.05),
        _layer("fill99turn", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.40, holding_days=5, close_rate_min=0.99, close_rate_max=1.05, turnover_min=1.0, turnover_max=12.0),
    ]),
    _variant("hybrid995_core120_fill99turn_40_max8_continue100_mh8", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.30, holding_days=5, max_holding_days=8, env={"GM_SCORE_CONTINUE_ENTRY_RATIO": "1.00"}, layers=[
        _layer("core995", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.20, holding_days=5, close_rate_min=0.995, close_rate_max=1.05),
        _layer("fill99turn", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.40, holding_days=5, close_rate_min=0.99, close_rate_max=1.05, turnover_min=1.0, turnover_max=12.0),
    ]),
    _variant("hybrid995_core120_fill99turn_40_max8_continue098_mh10", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.30, holding_days=5, max_holding_days=10, env={"GM_SCORE_CONTINUE_ENTRY_RATIO": "0.98"}, layers=[
        _layer("core995", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.20, holding_days=5, close_rate_min=0.995, close_rate_max=1.05),
        _layer("fill99turn", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.40, holding_days=5, close_rate_min=0.99, close_rate_max=1.05, turnover_min=1.0, turnover_max=12.0),
    ]),
    _variant("hybrid995_core120_fill99turn_40_max8_defer8", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.30, holding_days=5, env={"GM_DEFER_EXITS_WITHOUT_BUY_SIGNAL": "1", "GM_NO_SIGNAL_MAX_HOLDING_DAYS": "8"}, layers=[
        _layer("core995", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.20, holding_days=5, close_rate_min=0.995, close_rate_max=1.05),
        _layer("fill99turn", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.40, holding_days=5, close_rate_min=0.99, close_rate_max=1.05, turnover_min=1.0, turnover_max=12.0),
    ]),
    _variant("hybrid995_core120_fill99turn_40_max8_resize", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.30, holding_days=5, env={"GM_RESIZE_HELD_ON_SIGNAL": "1", "GM_RESIZE_HELD_TARGET_MULT": "1.20", "GM_RESIZE_HELD_MIN_DELTA_PCT": "0.005"}, layers=[
        _layer("core995", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.20, holding_days=5, close_rate_min=0.995, close_rate_max=1.05),
        _layer("fill99turn", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.40, holding_days=5, close_rate_min=0.99, close_rate_max=1.05, turnover_min=1.0, turnover_max=12.0),
    ]),
    _variant("hybrid99_core110_fill98_40_max8_sells0", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.28, holding_days=5, env={"GM_MAX_DAILY_SELLS": "0"}, layers=[
        _layer("core99", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.10, holding_days=5, close_rate_min=0.99, close_rate_max=1.05),
        _layer("fill98", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.40, holding_days=5, close_rate_min=0.98, close_rate_max=1.06),
    ]),
    _variant("hybrid99_core110_fill98_40_max8_sells2", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.28, holding_days=5, env={"GM_MAX_DAILY_SELLS": "2"}, layers=[
        _layer("core99", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.10, holding_days=5, close_rate_min=0.99, close_rate_max=1.05),
        _layer("fill98", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.40, holding_days=5, close_rate_min=0.98, close_rate_max=1.06),
    ]),
    _variant("hybrid99_core110_fill98_40_max8_continue098_mh8", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.28, holding_days=5, max_holding_days=8, env={"GM_SCORE_CONTINUE_ENTRY_RATIO": "0.98"}, layers=[
        _layer("core99", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.10, holding_days=5, close_rate_min=0.99, close_rate_max=1.05),
        _layer("fill98", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.40, holding_days=5, close_rate_min=0.98, close_rate_max=1.06),
    ]),
    _variant("hybrid99_core110_fill98_40_max8_continue100_mh8", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.28, holding_days=5, max_holding_days=8, env={"GM_SCORE_CONTINUE_ENTRY_RATIO": "1.00"}, layers=[
        _layer("core99", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.10, holding_days=5, close_rate_min=0.99, close_rate_max=1.05),
        _layer("fill98", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.40, holding_days=5, close_rate_min=0.98, close_rate_max=1.06),
    ]),
    _variant("hybrid99_core110_fill98_40_max8_scoreexit098", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.28, holding_days=5, env={"GM_OPEN_DAILY_SCORE_EXIT": "1", "GM_SCORE_EXIT_ENTRY_RATIO": "0.98", "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "1"}, layers=[
        _layer("core99", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.10, holding_days=5, close_rate_min=0.99, close_rate_max=1.05),
        _layer("fill98", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.40, holding_days=5, close_rate_min=0.98, close_rate_max=1.06),
    ]),
    _variant("hybrid99_core110_fill98_40_max8_defer8", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.28, holding_days=5, env={"GM_DEFER_EXITS_WITHOUT_BUY_SIGNAL": "1", "GM_NO_SIGNAL_MAX_HOLDING_DAYS": "8"}, layers=[
        _layer("core99", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.10, holding_days=5, close_rate_min=0.99, close_rate_max=1.05),
        _layer("fill98", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.40, holding_days=5, close_rate_min=0.98, close_rate_max=1.06),
    ]),
    _variant("r5_80_floor990_top5_close995_105_t140_rankw", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.40, target_position_pct=0.28, holding_days=5, close_rate_min=0.995, close_rate_max=1.05, weight_mode="rank"),
    _variant("r5_80_floor990_top5_close995_105_t140_scorew", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.40, target_position_pct=0.28, holding_days=5, close_rate_min=0.995, close_rate_max=1.05, weight_mode="score"),
    _variant("r5_80_floor990_top5_close99_105_t102_rankw", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.02, target_position_pct=0.204, holding_days=5, close_rate_min=0.99, close_rate_max=1.05, weight_mode="rank"),
    _variant("r5_80_floor990_top5_close99_105_t102_scorew", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.02, target_position_pct=0.204, holding_days=5, close_rate_min=0.99, close_rate_max=1.05, weight_mode="score"),
    _variant("r5_80_floor990_top5_close98_106_rankw", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_position_pct=0.196, holding_days=5, close_rate_min=0.98, close_rate_max=1.06, weight_mode="rank"),
    _variant("r5_80_floor990_top5_close98_106_scorew", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_position_pct=0.196, holding_days=5, close_rate_min=0.98, close_rate_max=1.06, weight_mode="score"),
    _variant("r5_80_floor990_top8_close995_105_t160_rankw", formula="r5_80_r10_20", score_floor=0.990, top_k=8, max_positions=8, target_total_pct=1.60, target_position_pct=0.20, holding_days=5, close_rate_min=0.995, close_rate_max=1.05, weight_mode="rank"),
    _variant("r5_80_floor990_top8_close995_105_t160_scorew", formula="r5_80_r10_20", score_floor=0.990, top_k=8, max_positions=8, target_total_pct=1.60, target_position_pct=0.20, holding_days=5, close_rate_min=0.995, close_rate_max=1.05, weight_mode="score"),
    _variant("r5_80_floor990_top8_close99_105_t120_rankw", formula="r5_80_r10_20", score_floor=0.990, top_k=8, max_positions=8, target_total_pct=1.20, target_position_pct=0.15, holding_days=5, close_rate_min=0.99, close_rate_max=1.05, weight_mode="rank"),
    _variant("r5_80_floor990_top8_close99_105_t120_scorew", formula="r5_80_r10_20", score_floor=0.990, top_k=8, max_positions=8, target_total_pct=1.20, target_position_pct=0.15, holding_days=5, close_rate_min=0.99, close_rate_max=1.05, weight_mode="score"),
    _variant("hybrid995_core120_fill99turn_40_max8_rankw", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.30, holding_days=5, layers=[
        _layer("core995", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.20, holding_days=5, close_rate_min=0.995, close_rate_max=1.05, weight_mode="rank"),
        _layer("fill99turn", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.40, holding_days=5, close_rate_min=0.99, close_rate_max=1.05, turnover_min=1.0, turnover_max=12.0, weight_mode="rank"),
    ]),
    _variant("hybrid995_core120_fill99turn_40_max8_scorew", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.30, holding_days=5, layers=[
        _layer("core995", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.20, holding_days=5, close_rate_min=0.995, close_rate_max=1.05, weight_mode="score"),
        _layer("fill99turn", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.40, holding_days=5, close_rate_min=0.99, close_rate_max=1.05, turnover_min=1.0, turnover_max=12.0, weight_mode="score"),
    ]),
    _variant("hybrid99_core110_fill98_40_max8_rankw", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.28, holding_days=5, layers=[
        _layer("core99", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.10, holding_days=5, close_rate_min=0.99, close_rate_max=1.05, weight_mode="rank"),
        _layer("fill98", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.40, holding_days=5, close_rate_min=0.98, close_rate_max=1.06, weight_mode="rank"),
    ]),
    _variant("hybrid99_core110_fill98_40_max8_scorew", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.28, holding_days=5, layers=[
        _layer("core99", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.10, holding_days=5, close_rate_min=0.99, close_rate_max=1.05, weight_mode="score"),
        _layer("fill98", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.40, holding_days=5, close_rate_min=0.98, close_rate_max=1.06, weight_mode="score"),
    ]),
    _variant("hybrid995_core120_fill99turn_boost80_cap33", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.33, holding_days=5, boost_min_daily_target_pct=0.80, boost_target_cap_pct=0.33, env={"GM_STOP_LOSS_PCT": "none"}, layers=[
        _layer("core995", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.20, holding_days=5, close_rate_min=0.995, close_rate_max=1.05),
        _layer("fill99turn", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.40, holding_days=5, close_rate_min=0.99, close_rate_max=1.05, turnover_min=1.0, turnover_max=12.0),
    ]),
    _variant("hybrid995_core120_fill99turn_boost80_cap40", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.40, holding_days=5, boost_min_daily_target_pct=0.80, boost_target_cap_pct=0.40, env={"GM_STOP_LOSS_PCT": "none"}, layers=[
        _layer("core995", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.20, holding_days=5, close_rate_min=0.995, close_rate_max=1.05),
        _layer("fill99turn", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.40, holding_days=5, close_rate_min=0.99, close_rate_max=1.05, turnover_min=1.0, turnover_max=12.0),
    ]),
    _variant("hybrid995_core120_fill99turn_boost80_cap49", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.49, holding_days=5, boost_min_daily_target_pct=0.80, boost_target_cap_pct=0.49, env={"GM_STOP_LOSS_PCT": "none"}, layers=[
        _layer("core995", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.20, holding_days=5, close_rate_min=0.995, close_rate_max=1.05),
        _layer("fill99turn", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.40, holding_days=5, close_rate_min=0.99, close_rate_max=1.05, turnover_min=1.0, turnover_max=12.0),
    ]),
    _variant("hybrid995_core120_fill99turn_boost90_cap49", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.49, holding_days=5, boost_min_daily_target_pct=0.90, boost_target_cap_pct=0.49, env={"GM_STOP_LOSS_PCT": "none"}, layers=[
        _layer("core995", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.20, holding_days=5, close_rate_min=0.995, close_rate_max=1.05),
        _layer("fill99turn", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.40, holding_days=5, close_rate_min=0.99, close_rate_max=1.05, turnover_min=1.0, turnover_max=12.0),
    ]),
    _variant("hybrid995_core120_fill99turn_boost98_cap49", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.49, holding_days=5, boost_min_daily_target_pct=0.98, boost_target_cap_pct=0.49, env={"GM_STOP_LOSS_PCT": "none"}, layers=[
        _layer("core995", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.20, holding_days=5, close_rate_min=0.995, close_rate_max=1.05),
        _layer("fill99turn", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.40, holding_days=5, close_rate_min=0.99, close_rate_max=1.05, turnover_min=1.0, turnover_max=12.0),
    ]),
    _variant("hybrid995_core120_fill99turn_boost80_cap60", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.60, holding_days=5, boost_min_daily_target_pct=0.80, boost_target_cap_pct=0.60, env={"GM_STOP_LOSS_PCT": "none"}, layers=[
        _layer("core995", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.20, holding_days=5, close_rate_min=0.995, close_rate_max=1.05),
        _layer("fill99turn", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.40, holding_days=5, close_rate_min=0.99, close_rate_max=1.05, turnover_min=1.0, turnover_max=12.0),
    ]),
    _variant("hybrid995_core120_fill99turn_boost80_cap80", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.80, holding_days=5, boost_min_daily_target_pct=0.80, boost_target_cap_pct=0.80, env={"GM_STOP_LOSS_PCT": "none"}, layers=[
        _layer("core995", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.20, holding_days=5, close_rate_min=0.995, close_rate_max=1.05),
        _layer("fill99turn", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.40, holding_days=5, close_rate_min=0.99, close_rate_max=1.05, turnover_min=1.0, turnover_max=12.0),
    ]),
    _variant("hybrid99_core110_fill98_boost80_cap40", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.40, holding_days=5, boost_min_daily_target_pct=0.80, boost_target_cap_pct=0.40, layers=[
        _layer("core99", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.10, holding_days=5, close_rate_min=0.99, close_rate_max=1.05),
        _layer("fill98", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.40, holding_days=5, close_rate_min=0.98, close_rate_max=1.06),
    ]),
    _variant("hybrid99_core110_fill98_boost90_cap49", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.49, holding_days=5, boost_min_daily_target_pct=0.90, boost_target_cap_pct=0.49, layers=[
        _layer("core99", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.10, holding_days=5, close_rate_min=0.99, close_rate_max=1.05),
        _layer("fill98", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.40, holding_days=5, close_rate_min=0.98, close_rate_max=1.06),
    ]),
    _variant("hybrid99_core110_fill98_boost98_cap49", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.49, holding_days=5, boost_min_daily_target_pct=0.98, boost_target_cap_pct=0.49, layers=[
        _layer("core99", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.10, holding_days=5, close_rate_min=0.99, close_rate_max=1.05),
        _layer("fill98", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.40, holding_days=5, close_rate_min=0.98, close_rate_max=1.06),
    ]),
    _variant("hybrid995_core120_fill99turn_boost80_cap49_continue098_mh8", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.49, holding_days=5, max_holding_days=8, boost_min_daily_target_pct=0.80, boost_target_cap_pct=0.49, env={"GM_STOP_LOSS_PCT": "none", "GM_SCORE_CONTINUE_ENTRY_RATIO": "0.98"}, layers=[
        _layer("core995", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.20, holding_days=5, close_rate_min=0.995, close_rate_max=1.05),
        _layer("fill99turn", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.40, holding_days=5, close_rate_min=0.99, close_rate_max=1.05, turnover_min=1.0, turnover_max=12.0),
    ]),
    _variant("hybrid995_core120_fill99turn_boost80_cap49_continue100_mh8", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.49, holding_days=5, max_holding_days=8, boost_min_daily_target_pct=0.80, boost_target_cap_pct=0.49, env={"GM_STOP_LOSS_PCT": "none", "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.00"}, layers=[
        _layer("core995", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.20, holding_days=5, close_rate_min=0.995, close_rate_max=1.05),
        _layer("fill99turn", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.40, holding_days=5, close_rate_min=0.99, close_rate_max=1.05, turnover_min=1.0, turnover_max=12.0),
    ]),
    _variant("hybrid995_core120_fill99turn_boost80_cap49_continue098_mh10", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.49, holding_days=5, max_holding_days=10, boost_min_daily_target_pct=0.80, boost_target_cap_pct=0.49, env={"GM_STOP_LOSS_PCT": "none", "GM_SCORE_CONTINUE_ENTRY_RATIO": "0.98"}, layers=[
        _layer("core995", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.20, holding_days=5, close_rate_min=0.995, close_rate_max=1.05),
        _layer("fill99turn", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.40, holding_days=5, close_rate_min=0.99, close_rate_max=1.05, turnover_min=1.0, turnover_max=12.0),
    ]),
    _variant("hybrid995_core120_fill99turn_boost80_cap49_defer8", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.49, holding_days=5, boost_min_daily_target_pct=0.80, boost_target_cap_pct=0.49, env={"GM_STOP_LOSS_PCT": "none", "GM_DEFER_EXITS_WITHOUT_BUY_SIGNAL": "1", "GM_NO_SIGNAL_MAX_HOLDING_DAYS": "8"}, layers=[
        _layer("core995", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.20, holding_days=5, close_rate_min=0.995, close_rate_max=1.05),
        _layer("fill99turn", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.40, holding_days=5, close_rate_min=0.99, close_rate_max=1.05, turnover_min=1.0, turnover_max=12.0),
    ]),
    _variant("hybrid995_core120_fill99turn_boost80_cap49_resize", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.49, holding_days=5, boost_min_daily_target_pct=0.80, boost_target_cap_pct=0.49, env={"GM_STOP_LOSS_PCT": "none", "GM_RESIZE_HELD_ON_SIGNAL": "1", "GM_RESIZE_HELD_TARGET_MULT": "1.20", "GM_RESIZE_HELD_MIN_DELTA_PCT": "0.005"}, layers=[
        _layer("core995", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.20, holding_days=5, close_rate_min=0.995, close_rate_max=1.05),
        _layer("fill99turn", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.40, holding_days=5, close_rate_min=0.99, close_rate_max=1.05, turnover_min=1.0, turnover_max=12.0),
    ]),
    _variant("hybrid995_core120_fill99turn_boost80_cap49_sells2", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.49, holding_days=5, boost_min_daily_target_pct=0.80, boost_target_cap_pct=0.49, env={"GM_STOP_LOSS_PCT": "none", "GM_MAX_DAILY_SELLS": "2"}, layers=[
        _layer("core995", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.20, holding_days=5, close_rate_min=0.995, close_rate_max=1.05),
        _layer("fill99turn", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.40, holding_days=5, close_rate_min=0.99, close_rate_max=1.05, turnover_min=1.0, turnover_max=12.0),
    ]),
    _variant("hybrid995_core120_fill99turn_boost80_cap49_h6", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.49, holding_days=6, boost_min_daily_target_pct=0.80, boost_target_cap_pct=0.49, env={"GM_STOP_LOSS_PCT": "none"}, layers=[
        _layer("core995", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.20, holding_days=6, close_rate_min=0.995, close_rate_max=1.05),
        _layer("fill99turn", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.40, holding_days=6, close_rate_min=0.99, close_rate_max=1.05, turnover_min=1.0, turnover_max=12.0),
    ]),
    _variant("hybrid995_core120_fill99turn_boost80_cap49_h7", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.49, holding_days=7, boost_min_daily_target_pct=0.80, boost_target_cap_pct=0.49, env={"GM_STOP_LOSS_PCT": "none"}, layers=[
        _layer("core995", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.20, holding_days=7, close_rate_min=0.995, close_rate_max=1.05),
        _layer("fill99turn", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.40, holding_days=7, close_rate_min=0.99, close_rate_max=1.05, turnover_min=1.0, turnover_max=12.0),
    ]),
    _variant("hybrid995_core120_fill99turn_boost80_cap33_h7", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.33, holding_days=7, boost_min_daily_target_pct=0.80, boost_target_cap_pct=0.33, env={"GM_STOP_LOSS_PCT": "none"}, layers=[
        _layer("core995", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.20, holding_days=7, close_rate_min=0.995, close_rate_max=1.05),
        _layer("fill99turn", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.40, holding_days=7, close_rate_min=0.99, close_rate_max=1.05, turnover_min=1.0, turnover_max=12.0),
    ]),
    _variant("hybrid995_core120_fill99turn_boost80_cap40_h7", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.40, holding_days=7, boost_min_daily_target_pct=0.80, boost_target_cap_pct=0.40, env={"GM_STOP_LOSS_PCT": "none"}, layers=[
        _layer("core995", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.20, holding_days=7, close_rate_min=0.995, close_rate_max=1.05),
        _layer("fill99turn", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.40, holding_days=7, close_rate_min=0.99, close_rate_max=1.05, turnover_min=1.0, turnover_max=12.0),
    ]),
    _variant("hybrid995_core120_fill99turn_boost90_cap49_h7", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.49, holding_days=7, boost_min_daily_target_pct=0.90, boost_target_cap_pct=0.49, env={"GM_STOP_LOSS_PCT": "none"}, layers=[
        _layer("core995", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.20, holding_days=7, close_rate_min=0.995, close_rate_max=1.05),
        _layer("fill99turn", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.40, holding_days=7, close_rate_min=0.99, close_rate_max=1.05, turnover_min=1.0, turnover_max=12.0),
    ]),
    _variant("hybrid995_core120_fill99turn_boost80_cap40_h8", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.40, holding_days=8, boost_min_daily_target_pct=0.80, boost_target_cap_pct=0.40, env={"GM_STOP_LOSS_PCT": "none"}, layers=[
        _layer("core995", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.20, holding_days=8, close_rate_min=0.995, close_rate_max=1.05),
        _layer("fill99turn", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.40, holding_days=8, close_rate_min=0.99, close_rate_max=1.05, turnover_min=1.0, turnover_max=12.0),
    ]),
    _variant("hybrid995_core120_fill99turn_boost80_cap49_h8", formula="r5_80_r10_20", top_k=8, max_positions=8, target_position_pct=0.49, holding_days=8, boost_min_daily_target_pct=0.80, boost_target_cap_pct=0.49, env={"GM_STOP_LOSS_PCT": "none"}, layers=[
        _layer("core995", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=1.20, holding_days=8, close_rate_min=0.995, close_rate_max=1.05),
        _layer("fill99turn", formula="r5_80_r10_20", score_floor=0.990, top_k=5, max_positions=5, target_total_pct=0.40, holding_days=8, close_rate_min=0.99, close_rate_max=1.05, turnover_min=1.0, turnover_max=12.0),
    ]),
]


def _quote_ident(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _formula_sql(name: str) -> str:
    if name not in FORMULAS:
        raise ValueError(f"unknown formula: {name}")
    return FORMULAS[name]


def ensure_score_tables() -> None:
    conn = sqlite3.connect(FUSION_DB)
    try:
        for name, formula in FORMULAS.items():
            table_name = SCORE_TABLES[name]
            conn.executescript(
                f"""
                DROP TABLE IF EXISTS {_quote_ident(table_name)};
                CREATE TABLE {_quote_ident(table_name)} AS
                SELECT
                    trade_date,
                    stock_code,
                    {formula} AS pred_prob
                FROM {_quote_ident("fusion_rank_base")}
                WHERE rank_5d IS NOT NULL
                  AND rank_10d IS NOT NULL;
                CREATE INDEX idx_{table_name}_trade_stock ON {_quote_ident(table_name)}(trade_date, stock_code);
                CREATE INDEX idx_{table_name}_trade_pred ON {_quote_ident(table_name)}(trade_date, pred_prob DESC);
                """
            )
        conn.commit()
    finally:
        conn.close()


def _load_rows(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    formula = _formula_sql(str(cfg["formula"]))
    where = [
        "trade_date >= ?",
        "trade_date <= ?",
        "stock_code NOT LIKE '%.BJ'",
        "COALESCE(name, '') NOT LIKE 'ST%'",
        "COALESCE(name, '') NOT LIKE '*ST%'",
        "COALESCE(name, '') NOT LIKE ?",
        "COALESCE(name, '') NOT LIKE ?",
        "COALESCE(name, '') NOT LIKE ?",
        "(limit_times IS NULL OR limit_times = '' OR limit_times = 'None' OR CAST(limit_times AS REAL) = 0)",
        "rank_5d IS NOT NULL",
        "rank_10d IS NOT NULL",
        "close IS NOT NULL",
    ]
    params: list[Any] = [START_DATE, END_DATE, "%退市%", "退%", "%退"]
    for field, op, value in [
        ("amount", ">=", cfg.get("amount_min")),
        ("amount", "<=", cfg.get("amount_max")),
        ("turnover_rate", ">=", cfg.get("turnover_min")),
        ("turnover_rate", "<=", cfg.get("turnover_max")),
        ("total_mv", ">=", cfg.get("total_mv_min")),
        ("total_mv", "<=", cfg.get("total_mv_max")),
        ("volume_ratio", ">=", cfg.get("volume_ratio_min")),
        ("volume_ratio", "<=", cfg.get("volume_ratio_max")),
        ("close_rate", ">=", cfg.get("close_rate_min")),
        ("close_rate", "<=", cfg.get("close_rate_max")),
        ("rank_5d", ">=", cfg.get("rank5d_min")),
        ("rank_10d", ">=", cfg.get("rank10d_min")),
    ]:
        if value is None:
            continue
        where.append(f"{field} IS NOT NULL AND {field} {op} ?")
        params.append(float(value))
    if cfg.get("atr_required"):
        where.append("atr_qfq IS NOT NULL")
    if cfg.get("score_floor") is not None:
        where.append(f"({formula}) >= ?")
        params.append(float(cfg["score_floor"]))
    params.append(int(cfg["daily_pool"]))
    conn = sqlite3.connect(FUSION_DB)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            f"""
            SELECT *
            FROM (
                SELECT
                    trade_date,
                    stock_code,
                    {formula} AS pred_prob,
                    pred_5d,
                    pred_10d,
                    rank_5d,
                    rank_10d,
                    name,
                    pre_close,
                    open,
                    close,
                    amount,
                    turnover_rate,
                    total_mv,
                    atr_qfq,
                    limit_times,
                    ROW_NUMBER() OVER (
                        PARTITION BY trade_date
                        ORDER BY ({formula}) DESC, stock_code
                    ) AS rn
                FROM {_quote_ident("fusion_rank_base")}
                WHERE {" AND ".join(where)}
            )
            WHERE rn <= ?
            ORDER BY trade_date, pred_prob DESC, stock_code
            """,
            params,
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def _attach_source_fields(signals: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    by_key = {(str(row["trade_date"]), str(row["stock_code"])): row for row in rows}
    for signal in signals:
        source = by_key.get((str(signal.get("signal_date") or ""), str(signal.get("stock_code") or "")), {})
        for field in ["pred_5d", "pred_10d", "rank_5d", "rank_10d", "amount", "turnover_rate", "total_mv", "atr_qfq"]:
            signal[field] = source.get(field)


def _boost_daily_target_pct(signals: list[dict[str, Any]], cfg: dict[str, Any]) -> None:
    min_daily_target = _to_float(cfg.get("boost_min_daily_target_pct"))
    cap = _to_float(cfg.get("boost_target_cap_pct"))
    if min_daily_target is None or min_daily_target <= 0:
        return
    cap = cap if cap is not None and cap > 0 else float(cfg["target_position_pct"])
    by_buy_date: dict[str, list[dict[str, Any]]] = {}
    for signal in signals:
        buy_date = str(signal.get("buy_date") or "")
        if not buy_date:
            continue
        by_buy_date.setdefault(buy_date, []).append(signal)
    for day_signals in by_buy_date.values():
        current_targets = [_to_float(row.get("target_pct"), float(cfg["target_position_pct"])) or 0.0 for row in day_signals]
        current_total = sum(current_targets)
        if current_total <= 0 or current_total >= min_daily_target:
            continue
        scale = min_daily_target / current_total
        boosted = [min(value * scale, cap) for value in current_targets]
        for row, target_pct in zip(day_signals, boosted):
            row["target_pct"] = target_pct


def _build_single_layer_signals(rows: list[dict[str, Any]], cfg: dict[str, Any], market_rows: dict[str, dict[str, dict]]) -> list[dict[str, Any]]:
    signals = build_gm_signal_rows(
        rows,
        config=SelectionConfig(
            top_k=int(cfg["top_k"]),
            pred_col="pred_prob",
            min_pred_prob=None,
            min_pred_quantile=None,
            max_atr_ratio=None,
            min_amount=None,
            min_turnover_rate=None,
            max_total_mv=None,
            max_per_industry=999999,
            exclude_bj=True,
            exclude_st=True,
            exclude_delisting=True,
            exclude_current_limit=True,
        ),
        market_rows_by_trade_date=market_rows,
        holding_days=int(cfg["holding_days"]),
        max_positions=int(cfg["max_positions"]),
        weight_mode=str(cfg["weight_mode"]),
        target_total_pct=float(cfg["target_total_pct"]),
    )
    _attach_source_fields(signals, rows)
    _boost_daily_target_pct(signals, cfg)
    return signals


def _write_layered_signal(cfg: dict[str, Any], signal_file: Path, market_rows: dict[str, dict[str, dict]]) -> None:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    layer_counts: dict[str, int] = {}
    for layer_index, layer_cfg in enumerate(cfg.get("layers") or [], start=1):
        rows = _load_rows(layer_cfg)
        signals = _build_single_layer_signals(rows, layer_cfg, market_rows)
        for signal in signals:
            buy_date = str(signal.get("buy_date") or "")
            symbol = str(signal.get("symbol") or "")
            key = (buy_date, symbol)
            if key in seen:
                continue
            seen.add(key)
            signal["layer"] = layer_cfg.get("name")
            signal["layer_index"] = layer_index
            signal["layer_rank"] = signal.get("rank")
            layer_counts[buy_date] = layer_counts.get(buy_date, 0) + 1
            signal["rank"] = layer_counts[buy_date]
            merged.append(signal)
    merged.sort(key=lambda row: (str(row.get("buy_date") or ""), int(row.get("layer_index") or 0), int(row.get("layer_rank") or 0), str(row.get("symbol") or "")))
    if not merged:
        raise ValueError(f"No layered signals to write for {cfg.get('name')}")
    _boost_daily_target_pct(merged, cfg)
    signal_file.parent.mkdir(parents=True, exist_ok=True)
    write_gm_signals_csv(merged, signal_file)


def _write_signal(rows: list[dict[str, Any]], cfg: dict[str, Any], signal_file: Path, market_rows: dict[str, dict[str, dict]]) -> None:
    if cfg.get("layers"):
        _write_layered_signal(cfg, signal_file, market_rows)
        return
    signals = _build_single_layer_signals(rows, cfg, market_rows)
    signal_file.parent.mkdir(parents=True, exist_ok=True)
    write_gm_signals_csv(signals, signal_file)


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


def _run_backtest(cfg: dict[str, Any], signal_file: Path, log_file: Path) -> int:
    if log_file.exists() and _extract_indicator(log_file):
        return 0
    env = os.environ.copy()
    env.update({str(key): str(value) for key, value in cfg["env"].items()})
    score_table = str(cfg.get("score_table") or SCORE_TABLES[str(cfg["formula"])])
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
        str(int(cfg["max_positions"])),
        "--holding-days",
        str(int(cfg["holding_days"])),
        "--max-holding-days",
        str(int(cfg.get("max_holding_days") or cfg["holding_days"])),
        "--target-position-pct",
        f"{float(cfg['target_position_pct']):.5f}",
        "--score-db",
        str(FUSION_DB),
        "--score-table",
        score_table,
        "--market-db",
        str(MARKET_DB),
        "--backtest-start",
        "2024-06-05 09:00:00",
        "--backtest-end",
        "2026-06-18 15:30:00",
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


def _exposure_stats(log_file: Path) -> dict[str, Any]:
    values: list[float] = []
    active: list[int] = []
    pattern = re.compile(r"invested_pct=([0-9.]+).*active_positions=(\d+)")
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


def _signal_stats(signal_file: Path) -> dict[str, Any]:
    with signal_file.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    targets = [_to_float(row.get("target_pct")) for row in rows]
    targets = [value for value in targets if value is not None]
    return {
        "signal_count": len(rows),
        "buy_days": len({row.get("buy_date") for row in rows if row.get("buy_date")}),
        "avg_signal_target_pct": sum(targets) / len(targets) if targets else None,
        "max_signal_target_pct": max(targets) if targets else None,
    }


def _summarize(cfg: dict[str, Any], signal_file: Path, log_file: Path, returncode: int) -> dict[str, Any]:
    indicator = _extract_indicator(log_file) or {}
    row = {
        **{key: value for key, value in cfg.items() if key != "env"},
        "env_json": json.dumps(cfg["env"], ensure_ascii=False, sort_keys=True),
        "returncode": returncode,
        "signal_file": str(signal_file),
        "log_file": str(log_file),
        "annual_return": indicator.get("pnl_ratio_annual"),
        "sharpe": indicator.get("sharp_ratio", indicator.get("sharpe_ratio")),
        "max_drawdown": indicator.get("max_drawdown"),
        "cum_return": indicator.get("pnl_ratio"),
        "win_ratio": indicator.get("win_ratio"),
        "open_count": indicator.get("open_count"),
        "close_count": indicator.get("close_count"),
    }
    row.update(_signal_stats(signal_file))
    row.update(_exposure_stats(log_file))
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def _metric(row: dict[str, Any], key: str) -> float:
    return _to_float(row.get(key), float("-inf")) or float("-inf")


def main() -> None:
    if not FUSION_DB.exists():
        raise RuntimeError(f"missing fusion db, run research_formal_5d10d_gate6_retest_20260622.py first: {FUSION_DB}")
    REPORT_DIR.joinpath("signals").mkdir(parents=True, exist_ok=True)
    REPORT_DIR.joinpath("logs").mkdir(parents=True, exist_ok=True)
    ensure_score_tables()
    market_rows = load_market_rows_by_trade_date(MARKET_DB, START_DATE, END_DATE)
    results: list[dict[str, Any]] = []
    for cfg in VARIANTS:
        name = str(cfg["name"])
        signal_file = REPORT_DIR / "signals" / f"{_safe(name)}.csv"
        log_file = REPORT_DIR / "logs" / f"{_safe(name)}.log"
        if not (signal_file.exists() and log_file.exists() and _extract_indicator(log_file)):
            _write_signal(_load_rows(cfg), cfg, signal_file, market_rows)
        returncode = _run_backtest(cfg, signal_file, log_file)
        row = _summarize(cfg, signal_file, log_file, returncode)
        results.append(row)
        _write_csv(REPORT_DIR / "summary.csv", results)
        print(json.dumps(row, ensure_ascii=False, default=str), flush=True)
    _write_csv(REPORT_DIR / "summary_by_annual.csv", sorted(results, key=lambda row: _metric(row, "annual_return"), reverse=True))
    _write_csv(REPORT_DIR / "summary_by_sharpe.csv", sorted(results, key=lambda row: _metric(row, "sharpe"), reverse=True))
    print(json.dumps({"report_dir": str(REPORT_DIR), "variants": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
