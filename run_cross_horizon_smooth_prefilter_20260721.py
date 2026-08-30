from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "quant" / "main" / "run_observation_validation_strategy_optimization_20260721.py"
SPEC = importlib.util.spec_from_file_location("strict_observation_base", BASE)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)

MODULE.__file__ = str(Path(__file__).resolve())
MODULE.CONFIG = (
    ROOT
    / "quant"
    / "main"
    / "config"
    / "strategy_research"
    / "cross_horizon_smooth_prefilter_20260721.json"
)
MODULE.OUT = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_cross_horizon_smooth_prefilter_20260721"
)
MODULE.POOL_DB = MODULE.OUT / "current_l4_pool.duckdb"
MODULE.SIGNALS = MODULE.OUT / "signals"
MODULE.SCORES = MODULE.OUT / "score_assets"
MODULE.LOGS = MODULE.OUT / "juejin_logs"
MODULE.LOCK = MODULE.OUT / "preregistration_lock.json"
MODULE.SCREEN = MODULE.OUT / "entry_screen_results.csv"
MODULE.ENTRY_FINALISTS = MODULE.OUT / "entry_finalists.json"
MODULE.JUEJIN_TUNE = MODULE.OUT / "juejin_tuning_results.csv"
MODULE.JUEJIN_FINALISTS = MODULE.OUT / "juejin_finalists_frozen.json"


def score_expr(spec: dict) -> str:
    base = (
        f"({float(spec['w3'])} * rank_3d + "
        f"{float(spec['w5'])} * rank_5d + "
        f"{float(spec['w10'])} * rank_10d)"
    )
    penalty = float(spec.get("agreement_penalty", 0.0))
    if penalty:
        return f"({base} - {penalty} * abs(rank_10d - rank_5d))"
    return base


def load_blend_frame(con, weights: dict, min_rank: float, include_holdout: bool) -> pd.DataFrame:
    expr = score_expr(weights)
    holdout_clause = "" if include_holdout else "AND signal_date <= '20251231'"
    source_columns = (
        "* EXCLUDE(next_open_raw, next_open_return_raw, open_return_5d_raw)"
        if include_holdout
        else "*"
    )
    return con.execute(
        f"""
        WITH scored AS (
          SELECT {source_columns}, {expr} blend_score
          FROM pool
          WHERE amount >= 90000 AND total_mv >= 200000 {holdout_clause}
        ), ranked_score AS (
          SELECT *, percent_rank() OVER(
            PARTITION BY signal_date ORDER BY blend_score
          ) blend_rank
          FROM scored
        )
        SELECT * FROM ranked_score WHERE blend_rank >= {float(min_rank)}
        ORDER BY signal_date, blend_score DESC, stock_code
        """
    ).fetchdf()


def filter_case_frame(base: pd.DataFrame, case: dict) -> pd.DataFrame:
    support = case["support"]
    pct = case["pct"]
    regime = case["regime"]
    mask = (
        (base["blend_rank"] >= float(case["blend_rank_min"]))
        & (base["rank_10d"] >= float(support["rank_10d_min"]))
        & (base["rank_5d"] >= float(support["rank_5d_min"]))
        & (base["rank_3d"] >= float(support.get("rank_3d_min", 0.0)))
        & (base["amount"] >= float(case["amount_min"]))
        & (base["total_mv"] >= float(case["mv_min"]))
        & base["signal_pct_chg_raw"].between(float(pct["min"]), float(pct["max"]))
        & (base["market_breadth_up"] >= float(regime["breadth_up_min"]))
        & (base["market_mean_pct"] >= float(regime["market_mean_pct_min"]))
    )
    out = base.loc[mask].copy()
    out["open_scale"] = 1.0
    out = out.sort_values(
        ["signal_date", "blend_score", "stock_code"],
        ascending=[True, False, True],
    )
    out["pick_rank"] = out.groupby("signal_date").cumcount() + 1
    return out[out["pick_rank"] <= int(case["topn"])].copy()


def build_score_asset(con, blend: str, weights: dict) -> Path:
    path = MODULE.SCORES / f"{blend}.duckdb"
    if path.exists():
        return path
    expr = score_expr(weights)
    con.execute(f"ATTACH '{path.as_posix()}' AS scoreout")
    con.execute(
        "CREATE TABLE scoreout.blended_rank_score AS "
        f"SELECT signal_date trade_date, stock_code, cast({expr} AS DOUBLE) pred_prob FROM pool"
    )
    con.execute("DETACH scoreout")
    return path


MODULE.load_blend_frame = load_blend_frame
MODULE.filter_case_frame = filter_case_frame
MODULE.build_score_asset = build_score_asset


if __name__ == "__main__":
    MODULE.main()
