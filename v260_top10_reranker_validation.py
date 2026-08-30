"""Machine verifier for the frozen one-shot Top10 reranker validation."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


BASELINE_COST = "0.0030"
STRESS_COST = "0.0065"
MINIMUM_DAYS = 120


def align_daily(baseline: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "return", "turnover"}
    if not required.issubset(baseline.columns) or not required.issubset(candidate.columns):
        raise RuntimeError("validation daily schema mismatch")
    if baseline["date"].astype(str).duplicated().any() or candidate["date"].astype(str).duplicated().any():
        raise RuntimeError("duplicate validation date")
    left = baseline[["date", "return", "turnover"]].copy()
    right = candidate[["date", "return", "turnover"]].copy()
    left["date"] = left["date"].astype(str)
    right["date"] = right["date"].astype(str)
    merged = left.merge(right, on="date", how="outer", validate="one_to_one", suffixes=("_baseline", "_candidate"), indicator=True)
    if not merged["_merge"].eq("both").all() or len(merged) < MINIMUM_DAYS:
        raise RuntimeError("validation common date domain incomplete")
    numeric = ["return_baseline", "return_candidate", "turnover_baseline", "turnover_candidate"]
    if not np.isfinite(merged[numeric].to_numpy(dtype=float)).all():
        raise RuntimeError("nonfinite validation metric input")
    return merged.sort_values("date", kind="mergesort").reset_index(drop=True)


def metrics(returns: np.ndarray, turnover: np.ndarray) -> dict:
    equity = np.cumprod(1.0 + returns)
    if np.any(equity <= 0):
        raise RuntimeError("nonpositive validation equity")
    std = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    peaks = np.maximum.accumulate(equity)
    return {
        "cumulative_return": float(equity[-1] - 1.0),
        "sharpe": float(np.mean(returns) / std * math.sqrt(252.0)) if std > 0 else 0.0,
        "max_drawdown": float(np.max(1.0 - equity / peaks)),
        "turnover_sum": float(np.sum(turnover)),
    }


def monthly_diagnostics(frame: pd.DataFrame) -> dict:
    work = frame.copy()
    work["month"] = work["date"].str[:6]
    work["excess_log"] = np.log1p(work["return_candidate"]) - np.log1p(work["return_baseline"])
    monthly = work.groupby("month", sort=True)["excess_log"].sum()
    nonzero = monthly[np.abs(monthly) > 1e-12]
    positive = nonzero[nonzero > 0].sort_values(ascending=False)
    total = float(positive.sum())
    return {
        "months": int(len(monthly)),
        "changed_months": int(len(nonzero)),
        "positive_changed_months": int(len(positive)),
        "positive_changed_month_ratio": float(len(positive) / len(nonzero)) if len(nonzero) else 0.0,
        "top1_positive_month_excess_share": float(positive.iloc[:1].sum() / total) if total else 0.0,
        "top3_positive_month_excess_share": float(positive.iloc[:3].sum() / total) if total else 0.0,
    }


def evaluate_validation(
    daily_by_cost: dict[str, dict[str, pd.DataFrame]],
    *,
    deterministic_replay: bool,
) -> dict:
    if set(daily_by_cost) != {BASELINE_COST, STRESS_COST}:
        raise RuntimeError("validation cost set mismatch")
    summaries = {}
    aligned = {}
    for cost in (BASELINE_COST, STRESS_COST):
        cases = daily_by_cost[cost]
        if set(cases) != {"baseline", "candidate"}:
            raise RuntimeError("validation case set mismatch")
        frame = align_daily(cases["baseline"], cases["candidate"])
        aligned[cost] = frame
        summaries[cost] = {
            "baseline": metrics(
                frame["return_baseline"].to_numpy(dtype=float),
                frame["turnover_baseline"].to_numpy(dtype=float),
            ),
            "candidate": metrics(
                frame["return_candidate"].to_numpy(dtype=float),
                frame["turnover_candidate"].to_numpy(dtype=float),
            ),
        }
    base = summaries[BASELINE_COST]
    stress = summaries[STRESS_COST]
    tolerance = 1e-12
    gates = {
        "baseline_cost_return_higher": base["candidate"]["cumulative_return"] > base["baseline"]["cumulative_return"],
        "baseline_cost_sharpe_higher": base["candidate"]["sharpe"] > base["baseline"]["sharpe"],
        "baseline_cost_mdd_not_higher": base["candidate"]["max_drawdown"] <= base["baseline"]["max_drawdown"] + tolerance,
        "stress_return_not_lower": stress["candidate"]["cumulative_return"] + tolerance >= stress["baseline"]["cumulative_return"],
        "stress_sharpe_not_lower": stress["candidate"]["sharpe"] + tolerance >= stress["baseline"]["sharpe"],
        "stress_mdd_not_higher": stress["candidate"]["max_drawdown"] <= stress["baseline"]["max_drawdown"] + tolerance,
        "turnover_increase_within_five_percent": base["candidate"]["turnover_sum"] <= base["baseline"]["turnover_sum"] * 1.05 + tolerance,
        "deterministic_replay": bool(deterministic_replay),
    }
    return {
        "decision": "validation_passed_candidate_for_production_review" if all(gates.values()) else "validation_rejected",
        "trading_days": len(aligned[BASELINE_COST]),
        "summaries": summaries,
        "monthly_diagnostics": monthly_diagnostics(aligned[BASELINE_COST]),
        "gates": gates,
        "production_modified": False,
    }
