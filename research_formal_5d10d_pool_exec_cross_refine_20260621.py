from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import os
import re
import sqlite3
import subprocess
from pathlib import Path

from gm_signal_module import build_gm_signal_rows, load_market_rows_by_trade_date, write_gm_signals_csv
from selection_module import SelectionConfig


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA_DIR = ROOT / "quant" / "data_file"
SOURCE_DB = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_5d10d_l5_candidate_grid_20260621"
    / "fusion_5d10d.db"
)
REPORT_DIR = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_5d10d_pool_exec_cross_refine_20260621"
)
SCORE_DB = REPORT_DIR / "pool_exec_scores.db"
MARKET_DB = DATA_DIR / "STOCK_DAILY_DATA.db"
STRATEGY_DIR = Path(r"D:\work\dfcf\juejin\strategy\ce327750-634e-11f1-b8d7-10ffe0295517")
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")
BACKTEST_START = "2024-06-05 09:00:00"
BACKTEST_END = "2026-06-30 15:30:00"
START_DATE = "20240604"
END_DATE = "20260618"


VARIANTS = [
    {
        "name": "base_p90f60",
        "primary_asset": "rank_10d90_5d10",
        "primary_limit": 2,
        "primary_max_total_mv": 150000.0,
        "primary_min_amount": 10000.0,
        "primary_min_turnover_rate": 0.3,
        "fallback_asset": "rank_10d60_5d40",
        "fallback_max_total_mv": 200000.0,
        "fallback_min_amount": 10000.0,
        "fallback_min_turnover_rate": 0.3,
        "holding_days": 5,
        "target_pct": 0.109,
    },
    {
        "name": "plim1_p90f60",
        "primary_asset": "rank_10d90_5d10",
        "primary_limit": 1,
        "primary_max_total_mv": 150000.0,
        "primary_min_amount": 10000.0,
        "primary_min_turnover_rate": 0.3,
        "fallback_asset": "rank_10d60_5d40",
        "fallback_max_total_mv": 200000.0,
        "fallback_min_amount": 10000.0,
        "fallback_min_turnover_rate": 0.3,
        "holding_days": 5,
        "target_pct": 0.109,
    },
    {
        "name": "plim3_p90f60",
        "primary_asset": "rank_10d90_5d10",
        "primary_limit": 3,
        "primary_max_total_mv": 150000.0,
        "primary_min_amount": 10000.0,
        "primary_min_turnover_rate": 0.3,
        "fallback_asset": "rank_10d60_5d40",
        "fallback_max_total_mv": 200000.0,
        "fallback_min_amount": 10000.0,
        "fallback_min_turnover_rate": 0.3,
        "holding_days": 5,
        "target_pct": 0.109,
    },
    {
        "name": "pmv100_p90f60",
        "primary_asset": "rank_10d90_5d10",
        "primary_limit": 2,
        "primary_max_total_mv": 100000.0,
        "primary_min_amount": 10000.0,
        "primary_min_turnover_rate": 0.3,
        "fallback_asset": "rank_10d60_5d40",
        "fallback_max_total_mv": 200000.0,
        "fallback_min_amount": 10000.0,
        "fallback_min_turnover_rate": 0.3,
        "holding_days": 5,
        "target_pct": 0.109,
    },
    {
        "name": "pmv200_p90f60",
        "primary_asset": "rank_10d90_5d10",
        "primary_limit": 2,
        "primary_max_total_mv": 200000.0,
        "primary_min_amount": 10000.0,
        "primary_min_turnover_rate": 0.3,
        "fallback_asset": "rank_10d60_5d40",
        "fallback_max_total_mv": 200000.0,
        "fallback_min_amount": 10000.0,
        "fallback_min_turnover_rate": 0.3,
        "holding_days": 5,
        "target_pct": 0.109,
    },
    {
        "name": "fallback150_p90f60",
        "primary_asset": "rank_10d90_5d10",
        "primary_limit": 2,
        "primary_max_total_mv": 150000.0,
        "primary_min_amount": 10000.0,
        "primary_min_turnover_rate": 0.3,
        "fallback_asset": "rank_10d60_5d40",
        "fallback_max_total_mv": 150000.0,
        "fallback_min_amount": 10000.0,
        "fallback_min_turnover_rate": 0.3,
        "holding_days": 5,
        "target_pct": 0.109,
    },
    {
        "name": "fallback300_p90f60",
        "primary_asset": "rank_10d90_5d10",
        "primary_limit": 2,
        "primary_max_total_mv": 150000.0,
        "primary_min_amount": 10000.0,
        "primary_min_turnover_rate": 0.3,
        "fallback_asset": "rank_10d60_5d40",
        "fallback_max_total_mv": 300000.0,
        "fallback_min_amount": 10000.0,
        "fallback_min_turnover_rate": 0.3,
        "holding_days": 5,
        "target_pct": 0.109,
    },
    {
        "name": "fallback_liq_p90f60",
        "primary_asset": "rank_10d90_5d10",
        "primary_limit": 2,
        "primary_max_total_mv": 150000.0,
        "primary_min_amount": 10000.0,
        "primary_min_turnover_rate": 0.3,
        "fallback_asset": "rank_10d60_5d40",
        "fallback_max_total_mv": 200000.0,
        "fallback_min_amount": 20000.0,
        "fallback_min_turnover_rate": 0.5,
        "holding_days": 5,
        "target_pct": 0.109,
    },
    {
        "name": "fallback_f50_liq",
        "primary_asset": "rank_10d90_5d10",
        "primary_limit": 2,
        "primary_max_total_mv": 150000.0,
        "primary_min_amount": 10000.0,
        "primary_min_turnover_rate": 0.3,
        "fallback_asset": "rank_10d50_5d50",
        "fallback_max_total_mv": 200000.0,
        "fallback_min_amount": 20000.0,
        "fallback_min_turnover_rate": 0.5,
        "holding_days": 5,
        "target_pct": 0.109,
    },
    {
        "name": "fallback_f70",
        "primary_asset": "rank_10d90_5d10",
        "primary_limit": 2,
        "primary_max_total_mv": 150000.0,
        "primary_min_amount": 10000.0,
        "primary_min_turnover_rate": 0.3,
        "fallback_asset": "rank_10d70_5d30",
        "fallback_max_total_mv": 200000.0,
        "fallback_min_amount": 10000.0,
        "fallback_min_turnover_rate": 0.3,
        "holding_days": 5,
        "target_pct": 0.109,
    },
    {
        "name": "primary80_f60",
        "primary_asset": "rank_10d80_5d20",
        "primary_limit": 2,
        "primary_max_total_mv": 150000.0,
        "primary_min_amount": 10000.0,
        "primary_min_turnover_rate": 0.3,
        "fallback_asset": "rank_10d60_5d40",
        "fallback_max_total_mv": 200000.0,
        "fallback_min_amount": 10000.0,
        "fallback_min_turnover_rate": 0.3,
        "holding_days": 5,
        "target_pct": 0.109,
    },
    {
        "name": "base_h4",
        "primary_asset": "rank_10d90_5d10",
        "primary_limit": 2,
        "primary_max_total_mv": 150000.0,
        "primary_min_amount": 10000.0,
        "primary_min_turnover_rate": 0.3,
        "fallback_asset": "rank_10d60_5d40",
        "fallback_max_total_mv": 200000.0,
        "fallback_min_amount": 10000.0,
        "fallback_min_turnover_rate": 0.3,
        "holding_days": 4,
        "target_pct": 0.109,
    },
    {
        "name": "base_h6",
        "primary_asset": "rank_10d90_5d10",
        "primary_limit": 2,
        "primary_max_total_mv": 150000.0,
        "primary_min_amount": 10000.0,
        "primary_min_turnover_rate": 0.3,
        "fallback_asset": "rank_10d60_5d40",
        "fallback_max_total_mv": 200000.0,
        "fallback_min_amount": 10000.0,
        "fallback_min_turnover_rate": 0.3,
        "holding_days": 6,
        "target_pct": 0.109,
    },
    {
        "name": "base_h7",
        "primary_asset": "rank_10d90_5d10",
        "primary_limit": 2,
        "primary_max_total_mv": 150000.0,
        "primary_min_amount": 10000.0,
        "primary_min_turnover_rate": 0.3,
        "fallback_asset": "rank_10d60_5d40",
        "fallback_max_total_mv": 200000.0,
        "fallback_min_amount": 10000.0,
        "fallback_min_turnover_rate": 0.3,
        "holding_days": 7,
        "target_pct": 0.109,
    },
    {
        "name": "base_h7_tp1085",
        "primary_asset": "rank_10d90_5d10",
        "primary_limit": 2,
        "primary_max_total_mv": 150000.0,
        "primary_min_amount": 10000.0,
        "primary_min_turnover_rate": 0.3,
        "fallback_asset": "rank_10d60_5d40",
        "fallback_max_total_mv": 200000.0,
        "fallback_min_amount": 10000.0,
        "fallback_min_turnover_rate": 0.3,
        "holding_days": 7,
        "target_pct": 0.1085,
    },
    {
        "name": "base_h7_tp1088",
        "primary_asset": "rank_10d90_5d10",
        "primary_limit": 2,
        "primary_max_total_mv": 150000.0,
        "primary_min_amount": 10000.0,
        "primary_min_turnover_rate": 0.3,
        "fallback_asset": "rank_10d60_5d40",
        "fallback_max_total_mv": 200000.0,
        "fallback_min_amount": 10000.0,
        "fallback_min_turnover_rate": 0.3,
        "holding_days": 7,
        "target_pct": 0.1088,
    },
    {
        "name": "base_h7_tp1091",
        "primary_asset": "rank_10d90_5d10",
        "primary_limit": 2,
        "primary_max_total_mv": 150000.0,
        "primary_min_amount": 10000.0,
        "primary_min_turnover_rate": 0.3,
        "fallback_asset": "rank_10d60_5d40",
        "fallback_max_total_mv": 200000.0,
        "fallback_min_amount": 10000.0,
        "fallback_min_turnover_rate": 0.3,
        "holding_days": 7,
        "target_pct": 0.1091,
    },
    {
        "name": "base_h7_tp1092",
        "primary_asset": "rank_10d90_5d10",
        "primary_limit": 2,
        "primary_max_total_mv": 150000.0,
        "primary_min_amount": 10000.0,
        "primary_min_turnover_rate": 0.3,
        "fallback_asset": "rank_10d60_5d40",
        "fallback_max_total_mv": 200000.0,
        "fallback_min_amount": 10000.0,
        "fallback_min_turnover_rate": 0.3,
        "holding_days": 7,
        "target_pct": 0.1092,
    },
    {
        "name": "base_h7_tp10922",
        "primary_asset": "rank_10d90_5d10",
        "primary_limit": 2,
        "primary_max_total_mv": 150000.0,
        "primary_min_amount": 10000.0,
        "primary_min_turnover_rate": 0.3,
        "fallback_asset": "rank_10d60_5d40",
        "fallback_max_total_mv": 200000.0,
        "fallback_min_amount": 10000.0,
        "fallback_min_turnover_rate": 0.3,
        "holding_days": 7,
        "target_pct": 0.10922,
    },
    {
        "name": "base_h7_tp10925",
        "primary_asset": "rank_10d90_5d10",
        "primary_limit": 2,
        "primary_max_total_mv": 150000.0,
        "primary_min_amount": 10000.0,
        "primary_min_turnover_rate": 0.3,
        "fallback_asset": "rank_10d60_5d40",
        "fallback_max_total_mv": 200000.0,
        "fallback_min_amount": 10000.0,
        "fallback_min_turnover_rate": 0.3,
        "holding_days": 7,
        "target_pct": 0.10925,
    },
    {
        "name": "base_h7_tp10928",
        "primary_asset": "rank_10d90_5d10",
        "primary_limit": 2,
        "primary_max_total_mv": 150000.0,
        "primary_min_amount": 10000.0,
        "primary_min_turnover_rate": 0.3,
        "fallback_asset": "rank_10d60_5d40",
        "fallback_max_total_mv": 200000.0,
        "fallback_min_amount": 10000.0,
        "fallback_min_turnover_rate": 0.3,
        "holding_days": 7,
        "target_pct": 0.10928,
    },
    {
        "name": "base_h7_tp1093",
        "primary_asset": "rank_10d90_5d10",
        "primary_limit": 2,
        "primary_max_total_mv": 150000.0,
        "primary_min_amount": 10000.0,
        "primary_min_turnover_rate": 0.3,
        "fallback_asset": "rank_10d60_5d40",
        "fallback_max_total_mv": 200000.0,
        "fallback_min_amount": 10000.0,
        "fallback_min_turnover_rate": 0.3,
        "holding_days": 7,
        "target_pct": 0.1093,
    },
    {
        "name": "base_h7_tp1095",
        "primary_asset": "rank_10d90_5d10",
        "primary_limit": 2,
        "primary_max_total_mv": 150000.0,
        "primary_min_amount": 10000.0,
        "primary_min_turnover_rate": 0.3,
        "fallback_asset": "rank_10d60_5d40",
        "fallback_max_total_mv": 200000.0,
        "fallback_min_amount": 10000.0,
        "fallback_min_turnover_rate": 0.3,
        "holding_days": 7,
        "target_pct": 0.1095,
    },
    {
        "name": "base_h7_mp12_tp085",
        "primary_asset": "rank_10d90_5d10",
        "primary_limit": 2,
        "primary_max_total_mv": 150000.0,
        "primary_min_amount": 10000.0,
        "primary_min_turnover_rate": 0.3,
        "fallback_asset": "rank_10d60_5d40",
        "fallback_max_total_mv": 200000.0,
        "fallback_min_amount": 10000.0,
        "fallback_min_turnover_rate": 0.3,
        "holding_days": 7,
        "max_positions": 12,
        "target_pct": 0.085,
    },
    {
        "name": "base_h7_mp12_tp09",
        "primary_asset": "rank_10d90_5d10",
        "primary_limit": 2,
        "primary_max_total_mv": 150000.0,
        "primary_min_amount": 10000.0,
        "primary_min_turnover_rate": 0.3,
        "fallback_asset": "rank_10d60_5d40",
        "fallback_max_total_mv": 200000.0,
        "fallback_min_amount": 10000.0,
        "fallback_min_turnover_rate": 0.3,
        "holding_days": 7,
        "max_positions": 12,
        "target_pct": 0.09,
    },
    {
        "name": "base_h8",
        "primary_asset": "rank_10d90_5d10",
        "primary_limit": 2,
        "primary_max_total_mv": 150000.0,
        "primary_min_amount": 10000.0,
        "primary_min_turnover_rate": 0.3,
        "fallback_asset": "rank_10d60_5d40",
        "fallback_max_total_mv": 200000.0,
        "fallback_min_amount": 10000.0,
        "fallback_min_turnover_rate": 0.3,
        "holding_days": 8,
        "target_pct": 0.109,
    },
    {
        "name": "base_h9",
        "primary_asset": "rank_10d90_5d10",
        "primary_limit": 2,
        "primary_max_total_mv": 150000.0,
        "primary_min_amount": 10000.0,
        "primary_min_turnover_rate": 0.3,
        "fallback_asset": "rank_10d60_5d40",
        "fallback_max_total_mv": 200000.0,
        "fallback_min_amount": 10000.0,
        "fallback_min_turnover_rate": 0.3,
        "holding_days": 9,
        "target_pct": 0.109,
    },
    {
        "name": "base_h9_tp1092",
        "primary_asset": "rank_10d90_5d10",
        "primary_limit": 2,
        "primary_max_total_mv": 150000.0,
        "primary_min_amount": 10000.0,
        "primary_min_turnover_rate": 0.3,
        "fallback_asset": "rank_10d60_5d40",
        "fallback_max_total_mv": 200000.0,
        "fallback_min_amount": 10000.0,
        "fallback_min_turnover_rate": 0.3,
        "holding_days": 9,
        "target_pct": 0.1092,
    },
    {
        "name": "base_h10",
        "primary_asset": "rank_10d90_5d10",
        "primary_limit": 2,
        "primary_max_total_mv": 150000.0,
        "primary_min_amount": 10000.0,
        "primary_min_turnover_rate": 0.3,
        "fallback_asset": "rank_10d60_5d40",
        "fallback_max_total_mv": 200000.0,
        "fallback_min_amount": 10000.0,
        "fallback_min_turnover_rate": 0.3,
        "holding_days": 10,
        "target_pct": 0.109,
    },
    {
        "name": "base_h6_tp1085",
        "primary_asset": "rank_10d90_5d10",
        "primary_limit": 2,
        "primary_max_total_mv": 150000.0,
        "primary_min_amount": 10000.0,
        "primary_min_turnover_rate": 0.3,
        "fallback_asset": "rank_10d60_5d40",
        "fallback_max_total_mv": 200000.0,
        "fallback_min_amount": 10000.0,
        "fallback_min_turnover_rate": 0.3,
        "holding_days": 6,
        "target_pct": 0.1085,
    },
    {
        "name": "base_h6_tp1088",
        "primary_asset": "rank_10d90_5d10",
        "primary_limit": 2,
        "primary_max_total_mv": 150000.0,
        "primary_min_amount": 10000.0,
        "primary_min_turnover_rate": 0.3,
        "fallback_asset": "rank_10d60_5d40",
        "fallback_max_total_mv": 200000.0,
        "fallback_min_amount": 10000.0,
        "fallback_min_turnover_rate": 0.3,
        "holding_days": 6,
        "target_pct": 0.1088,
    },
    {
        "name": "base_h6_tp1091",
        "primary_asset": "rank_10d90_5d10",
        "primary_limit": 2,
        "primary_max_total_mv": 150000.0,
        "primary_min_amount": 10000.0,
        "primary_min_turnover_rate": 0.3,
        "fallback_asset": "rank_10d60_5d40",
        "fallback_max_total_mv": 200000.0,
        "fallback_min_amount": 10000.0,
        "fallback_min_turnover_rate": 0.3,
        "holding_days": 6,
        "target_pct": 0.1091,
    },
    {
        "name": "atr10_base",
        "primary_asset": "rank_10d90_5d10",
        "primary_limit": 2,
        "primary_max_total_mv": 150000.0,
        "primary_min_amount": 10000.0,
        "primary_min_turnover_rate": 0.3,
        "fallback_asset": "rank_10d60_5d40",
        "fallback_max_total_mv": 200000.0,
        "fallback_min_amount": 10000.0,
        "fallback_min_turnover_rate": 0.3,
        "max_atr_ratio": 0.10,
        "holding_days": 5,
        "target_pct": 0.109,
    },
    {
        "name": "atr08_base",
        "primary_asset": "rank_10d90_5d10",
        "primary_limit": 2,
        "primary_max_total_mv": 150000.0,
        "primary_min_amount": 10000.0,
        "primary_min_turnover_rate": 0.3,
        "fallback_asset": "rank_10d60_5d40",
        "fallback_max_total_mv": 200000.0,
        "fallback_min_amount": 10000.0,
        "fallback_min_turnover_rate": 0.3,
        "max_atr_ratio": 0.08,
        "holding_days": 5,
        "target_pct": 0.109,
    },
]


def _safe(value) -> str:
    return str(value).replace(".", "p").replace("-", "neg").replace("None", "none").replace(" ", "")


def _quote_ident(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _quote_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _formula(asset: str) -> str:
    formulas = {
        "rank_10d90_5d10": "(rank_10d * 0.90) + (rank_5d * 0.10)",
        "rank_10d80_5d20": "(rank_10d * 0.80) + (rank_5d * 0.20)",
        "rank_10d70_5d30": "(rank_10d * 0.70) + (rank_5d * 0.30)",
        "rank_10d60_5d40": "(rank_10d * 0.60) + (rank_5d * 0.40)",
        "rank_10d50_5d50": "(rank_10d * 0.50) + (rank_5d * 0.50)",
    }
    if asset not in formulas:
        raise ValueError(asset)
    return formulas[asset]


def _pool_dict(variant: dict, prefix: str) -> dict:
    return {
        "asset": variant[f"{prefix}_asset"],
        "max_total_mv": variant[f"{prefix}_max_total_mv"],
        "min_amount": variant[f"{prefix}_min_amount"],
        "min_turnover_rate": variant[f"{prefix}_min_turnover_rate"],
        "max_atr_ratio": variant.get("max_atr_ratio"),
    }


def _condition(alias: str, pool: dict, parameterized: bool) -> tuple[str, list[object]]:
    checks = [
        f"{alias}.trade_date >= ?",
        f"{alias}.trade_date <= ?",
        f"{alias}.stock_code NOT LIKE '%.BJ'",
        f"COALESCE({alias}.name, '') NOT LIKE 'ST%'",
        f"COALESCE({alias}.name, '') NOT LIKE '*ST%'",
        f"COALESCE({alias}.name, '') NOT LIKE '%退市%'",
        f"COALESCE({alias}.name, '') NOT LIKE '退%'",
        f"({alias}.limit_times IS NULL OR {alias}.limit_times = '' OR {alias}.limit_times = 'None' OR CAST({alias}.limit_times AS REAL) = 0)",
        f"{alias}.total_mv IS NOT NULL AND {alias}.total_mv <= ?",
        f"{alias}.amount IS NOT NULL AND {alias}.amount >= ?",
        f"{alias}.turnover_rate IS NOT NULL AND {alias}.turnover_rate >= ?",
        f"{alias}.close IS NOT NULL",
    ]
    values: list[object] = [
        START_DATE,
        END_DATE,
        float(pool["max_total_mv"]),
        float(pool["min_amount"]),
        float(pool["min_turnover_rate"]),
    ]
    if pool.get("max_atr_ratio") is not None:
        checks.append(f"{alias}.atr_qfq IS NOT NULL AND {alias}.close > 0 AND ({alias}.atr_qfq / {alias}.close) <= ?")
        values.append(float(pool["max_atr_ratio"]))
    if not parameterized:
        for value in values:
            placeholder = "NULL" if value is None else str(float(value)) if isinstance(value, float) else _quote_literal(str(value))
            checks[checks.index(next(item for item in checks if "?" in item))] = next(
                item for item in checks if "?" in item
            ).replace("?", placeholder, 1)
        values = []
    return " AND ".join(checks), values


def _literal_condition(alias: str, pool: dict) -> str:
    where, values = _condition(alias, pool, True)
    for value in values:
        if isinstance(value, (int, float)):
            literal = str(float(value))
        else:
            literal = _quote_literal(str(value))
        where = where.replace("?", literal, 1)
    return where


def _load_pool_rows(pool: dict, limit: int, tier_offset: float) -> list[dict]:
    formula = _formula(pool["asset"])
    where, values = _condition("b", pool, True)
    conn = sqlite3.connect(SOURCE_DB)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            f"""
            SELECT *
            FROM (
                SELECT
                    b.trade_date,
                    b.stock_code,
                    {tier_offset} + ({formula}) AS pred_prob,
                    b.name,
                    b.pre_close,
                    b.open,
                    b.close,
                    b.amount,
                    b.turnover_rate,
                    b.total_mv,
                    b.atr_qfq,
                    b.limit_times,
                    ROW_NUMBER() OVER (
                        PARTITION BY b.trade_date
                        ORDER BY ({formula}) DESC, b.stock_code
                    ) AS rn
                FROM fusion_rank_base b
                WHERE {where}
            )
            WHERE rn <= ?
            ORDER BY trade_date, pred_prob DESC, stock_code
            """,
            [*values, int(limit)],
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def _score_table_name(variant: dict) -> str:
    return (
        f"{variant['name']}_p{variant['primary_asset']}_plim{variant['primary_limit']}"
        f"_pmv{_safe(variant['primary_max_total_mv'])}_f{variant['fallback_asset']}"
        f"_fmv{_safe(variant['fallback_max_total_mv'])}_atr{_safe(variant.get('max_atr_ratio'))}"
    )


def _build_score_table(variant: dict) -> str:
    SCORE_DB.parent.mkdir(parents=True, exist_ok=True)
    table_name = _score_table_name(variant)
    conn = sqlite3.connect(SCORE_DB)
    try:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        ).fetchone()
        if exists:
            return table_name
        conn.execute(f"ATTACH DATABASE {_quote_literal(str(SOURCE_DB))} AS source")
        primary = _pool_dict(variant, "primary")
        fallback = _pool_dict(variant, "fallback")
        pcond = _literal_condition("b", primary)
        fcond = _literal_condition("b", fallback)
        primary_expr = _formula(primary["asset"])
        fallback_expr = _formula(fallback["asset"])
        conn.executescript(
            f"""
            DROP TABLE IF EXISTS {_quote_ident(table_name)};
            CREATE TABLE {_quote_ident(table_name)} AS
            SELECT
                b.trade_date,
                b.stock_code,
                CASE
                    WHEN {pcond} THEN 2.0 + ({primary_expr})
                    WHEN {fcond} THEN 1.0 + ({fallback_expr})
                    ELSE ({fallback_expr})
                END AS pred_prob
            FROM source.fusion_rank_base b
            WHERE b.trade_date >= '{START_DATE}' AND b.trade_date <= '{END_DATE}'
              AND b.rank_5d IS NOT NULL
              AND b.rank_10d IS NOT NULL;
            CREATE INDEX IF NOT EXISTS idx_{table_name}_trade_stock ON {_quote_ident(table_name)}(trade_date, stock_code);
            """
        )
        conn.commit()
        return table_name
    finally:
        conn.close()


def _write_signal(variant: dict, signal_file: Path) -> None:
    primary = _pool_dict(variant, "primary")
    fallback = _pool_dict(variant, "fallback")
    rows = []
    seen = set()
    for row in [*_load_pool_rows(primary, int(variant["primary_limit"]), 2.0), *_load_pool_rows(fallback, 5, 1.0)]:
        key = (row.get("trade_date"), row.get("stock_code"))
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    signals = build_gm_signal_rows(
        rows,
        config=SelectionConfig(
            top_k=5,
            pred_col="pred_prob",
            min_pred_prob=None,
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
        market_rows_by_trade_date=load_market_rows_by_trade_date(MARKET_DB, START_DATE, END_DATE),
        holding_days=int(variant["holding_days"]),
        max_positions=int(variant.get("max_positions", 10)),
        weight_mode="equal",
        target_total_pct=float(variant["target_pct"]) * float(variant.get("max_positions", 10)),
    )
    for signal in signals:
        signal["target_pct"] = f"{float(variant['target_pct']):.5f}"
        signal["score_exit_entry_ratio"] = "1.0"
        signal["min_holding_days_before_score_exit"] = "3"
    write_gm_signals_csv(signals, signal_file)


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


def _signal_stats(signal_file: Path) -> dict:
    with signal_file.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    return {"signal_count": len(rows), "buy_days": len({row.get("buy_date") for row in rows if row.get("buy_date")})}


def _run_backtest(variant: dict, table_name: str, signal_file: Path, log_file: Path) -> int:
    if log_file.exists() and _extract_indicator(log_file):
        return 0
    env = os.environ.copy()
    env.update(
        {
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
            "GM_FORCE_MARKET_ORDER": "0",
            "GM_FORCE_BUY_MARKET_ORDER": "0",
            "GM_SYNC_POSITIONS": "0",
            "GM_CASH_BUFFER": "0.995",
            "GM_VERBOSE_TRADES": "0",
            "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.0",
        }
    )
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
        str(variant.get("max_positions", 10)),
        "--holding-days",
        str(variant["holding_days"]),
        "--max-holding-days",
        str(variant["holding_days"]),
        "--target-position-pct",
        f"{float(variant['target_pct']):.5f}",
        "--score-db",
        str(SCORE_DB),
        "--score-table",
        table_name,
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


def _sort_by_annual(row: dict) -> tuple[float, float, float]:
    return (
        float(row.get("annual") or -999.0),
        float(row.get("sharpe") or -999.0),
        float(row.get("avg_invested_pct") or -999.0),
    )


def _sort_by_sharpe(row: dict) -> tuple[float, float, float]:
    return (
        float(row.get("sharpe") or -999.0),
        float(row.get("annual") or -999.0),
        float(row.get("avg_invested_pct") or -999.0),
    )


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    (REPORT_DIR / "signals").mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    results = []
    for index, variant in enumerate(VARIANTS, start=1):
        table_name = _build_score_table(variant)
        signal_file = REPORT_DIR / "signals" / f"{variant['name']}.csv"
        log_file = REPORT_DIR / "logs" / f"{variant['name']}.log"
        if not signal_file.exists():
            _write_signal(variant, signal_file)
        returncode = _run_backtest(variant, table_name, signal_file, log_file)
        indicator = _extract_indicator(log_file) or {}
        result = {
            **variant,
            "score_db": str(SCORE_DB),
            "score_table": table_name,
            "returncode": returncode,
            "signal_file": str(signal_file),
            "log_file": str(log_file),
            "annual": indicator.get("pnl_ratio_annual"),
            "sharpe": indicator.get("sharp_ratio"),
            "max_drawdown": indicator.get("max_drawdown"),
            "win_ratio": indicator.get("win_ratio"),
            "open_count": indicator.get("open_count"),
            "close_count": indicator.get("close_count"),
            **_signal_stats(signal_file),
            **_exposure_stats(log_file),
        }
        results.append(result)
        _write_rows(REPORT_DIR / "summary.csv", results)
        _write_rows(REPORT_DIR / "summary_by_annual.csv", sorted(results, key=_sort_by_annual, reverse=True))
        _write_rows(REPORT_DIR / "summary_by_sharpe.csv", sorted(results, key=_sort_by_sharpe, reverse=True))
        print(
            f"[{index}/{len(VARIANTS)}] {variant['name']} annual={result.get('annual')} sharpe={result.get('sharpe')} avg={result.get('avg_invested_pct')}",
            flush=True,
        )
    qualified = [
        row
        for row in sorted(results, key=_sort_by_sharpe, reverse=True)
        if float(row.get("annual") or -999.0) >= 2.0
        and float(row.get("sharpe") or -999.0) >= 3.0
        and float(row.get("avg_invested_pct") or -999.0) >= 0.80
    ]
    _write_rows(REPORT_DIR / "summary_qualified_target.csv", qualified)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
