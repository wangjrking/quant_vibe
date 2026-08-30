from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(r"D:/work/quant/quant_mcp")
SOURCE_HARNESS = (
    REPO
    / "quant/data_file/runtime/agent_workspaces/strategy-agent/work"
    / "prod_v260_fixed10_equalweight_development_comparison_20260812_r2"
    / "observation_attempt_v1/run_comparison.py"
)
OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/strategy_agent_v260_simple_core_tuning_20260821"
)
SOURCE_MIN_DATE = "20220606"
FIRST_BUY_DATE = "20220607"
DEVELOPMENT_END = "20251231"
VALIDATION_START = "20260105"
VALIDATION_END = "20260820"


def load_harness():
    spec = importlib.util.spec_from_file_location("v260_comparison_harness", SOURCE_HARNESS)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load frozen production comparison harness")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SOURCE_MIN_DATE = SOURCE_MIN_DATE
    module.END_DATE = VALIDATION_END
    module.FIRST_BUY_DATE = FIRST_BUY_DATE
    return module


def score_pair(arrays: dict[str, np.ndarray], weight_1d: float, window: int, alpha: float):
    raw = (
        float(weight_1d) * arrays["rank_1d"]
        + (1.0 - float(weight_1d)) * arrays["rank_10d"]
    ).astype(np.float32)
    finite = np.isfinite(raw)
    sums = np.vstack(
        [
            np.zeros((1, raw.shape[1]), dtype=np.float64),
            np.cumsum(np.where(finite, raw, 0.0), axis=0, dtype=np.float64),
        ]
    )
    counts = np.vstack(
        [
            np.zeros((1, raw.shape[1]), dtype=np.int32),
            np.cumsum(finite, axis=0, dtype=np.int32),
        ]
    )
    end = np.arange(1, raw.shape[0] + 1)
    start = np.maximum(end - int(window), 0)
    total = sums[end] - sums[start]
    count = counts[end] - counts[start]
    smooth = np.divide(
        total,
        count,
        out=np.full_like(total, np.nan),
        where=count > 0,
    ).astype(np.float32)
    score = (float(alpha) * raw + (1.0 - float(alpha)) * smooth).astype(np.float32)
    order = np.argsort(-np.nan_to_num(score, nan=-np.inf), axis=1, kind="stable")
    return score, order


def simple_definition(params: dict) -> dict:
    return {
        "min_hold_days": 4,
        "max_hold_days": int(params["max_hold_days"]),
        "sell_score_below": float(params["sell_score_below"]),
        "replacement_advantage": float(params["replacement_advantage"]),
    }


def run_simple(harness, arrays, protocol, params, end_date: str, record_actions: bool = False):
    score, order = score_pair(
        arrays,
        params["weight_1d"],
        params["smooth_window"],
        params["current_weight"],
    )
    definition = simple_definition(params)
    case_protocol = harness.v162.case_protocol(copy.deepcopy(protocol), definition)
    momentum = harness.v162.v134.market_momentum(arrays, 10)
    gross = np.where(
        np.isfinite(momentum) & (momentum >= 0.0),
        1.0,
        float(params["weak_gross"]),
    ).astype(float)
    width = int(params["max_positions"])
    target = gross / float(width)
    min_hold = np.full(len(arrays["dates"]), 4, dtype=np.int16)
    positions = np.full(len(arrays["dates"]), width, dtype=np.int16)
    return harness.v109.simulate(
        arrays,
        score,
        order,
        definition,
        case_protocol,
        end_date,
        FIRST_BUY_DATE,
        gross_target_override=gross,
        target_cohorts_override=width,
        target_pct_override=target,
        min_hold_days_override=min_hold,
        selection_mask_override=None,
        candidate_target_multiplier_override=None,
        max_positions_schedule_override=positions,
        record_actions=record_actions,
    )


def slice_frame(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    dates = frame["date"].astype(str)
    return frame[(dates >= start) & (dates <= end)].copy().reset_index(drop=True)


def metric_summary(daily: pd.DataFrame) -> dict:
    if daily.empty:
        raise RuntimeError("empty metric interval")
    returns = daily["return"].astype(float).to_numpy()
    curve = np.cumprod(1.0 + returns)
    dates = pd.to_datetime(daily["date"].astype(str), format="%Y%m%d")
    years = max((dates.iloc[-1] - dates.iloc[0]).days / 365.25, len(daily) / 252.0)
    cumulative = float(curve[-1] - 1.0)
    cagr = float(curve[-1] ** (1.0 / years) - 1.0)
    std = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    sharpe = float(np.mean(returns) / std * math.sqrt(252.0)) if std > 0 else 0.0
    running_max = np.maximum.accumulate(curve)
    max_drawdown = float(np.max(1.0 - curve / running_max))
    annual_returns = {}
    for year, group in daily.assign(year=dates.dt.year.astype(str)).groupby("year"):
        annual_returns[str(year)] = float(np.prod(1.0 + group["return"].astype(float)) - 1.0)
    return {
        "start": str(daily.iloc[0]["date"]),
        "end": str(daily.iloc[-1]["date"]),
        "days": int(len(daily)),
        "cumulative_return": cumulative,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "annual_returns": annual_returns,
        "turnover_sum": float(daily["turnover"].astype(float).sum()),
        "turnover_annualized": float(daily["turnover"].astype(float).sum() / years),
        "average_invested_ratio": float(daily["invested_ratio"].astype(float).mean()),
        "average_positions": float(daily["positions"].astype(float).mean()),
        "trade_events": int(daily["trades"].astype(int).sum()),
    }


def development_utility(metrics: dict) -> float:
    full_years = [
        float(value)
        for year, value in metrics["annual_returns"].items()
        if year in {"2023", "2024", "2025"}
    ]
    dispersion = float(np.std(full_years, ddof=0)) if full_years else 0.0
    return float(
        0.50 * metrics["cagr"]
        + 0.20 * metrics["sharpe"]
        - 0.30 * metrics["max_drawdown"]
        - 0.20 * dispersion
        - 0.001 * metrics["turnover_annualized"]
    )


def evaluate_simple(harness, arrays, protocol, params: dict) -> dict:
    daily = run_simple(harness, arrays, protocol, params, DEVELOPMENT_END)
    metrics = metric_summary(slice_frame(daily, FIRST_BUY_DATE, DEVELOPMENT_END))
    return {**params, "development_utility": development_utility(metrics), **metrics}


def best_row(rows: list[dict]) -> dict:
    return max(
        rows,
        key=lambda row: (
            row["development_utility"],
            row["sharpe"],
            row["cagr"],
            -row["max_drawdown"],
            json.dumps(row, sort_keys=True, default=str),
        ),
    )


def hash_frame(frame: pd.DataFrame) -> str:
    return hashlib.sha256(
        frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    harness = load_harness()
    protocol, rules, manifests = harness.validate_control_plane()
    arrays, access = harness.build_development_arrays()
    if str(arrays["dates"][0]) != SOURCE_MIN_DATE or str(arrays["dates"][-1]) != VALIDATION_END:
        raise PermissionError("research date boundary mismatch")

    baseline_definition = harness.production_definition(protocol)
    baseline_score, baseline_order = harness.v95.score_pair(arrays, 0.0, 7, 0.1)
    baseline_daily, baseline_actions = harness.v260.run_case(
        arrays,
        baseline_score,
        baseline_order,
        baseline_definition,
        protocol,
        VALIDATION_END,
        FIRST_BUY_DATE,
        record_actions=True,
    )
    baseline_dev = metric_summary(slice_frame(baseline_daily, FIRST_BUY_DATE, DEVELOPMENT_END))

    base = {
        "weight_1d": 0.0,
        "smooth_window": 7,
        "current_weight": 0.1,
        "max_positions": 10,
        "weak_gross": 0.2,
        "sell_score_below": 0.85,
        "replacement_advantage": 0.05,
        "max_hold_days": 20,
    }

    score_rows = []
    for weight_1d in (0.0, 0.25, 0.50, 0.70):
        for window in (3, 5, 7, 10):
            for alpha in (0.10, 0.30, 0.50):
                score_rows.append(
                    evaluate_simple(
                        harness,
                        arrays,
                        protocol,
                        {
                            **base,
                            "weight_1d": weight_1d,
                            "smooth_window": window,
                            "current_weight": alpha,
                        },
                    )
                )
    score_winner = best_row(score_rows)
    score_params = {key: score_winner[key] for key in base}

    portfolio_rows = []
    for width in (8, 10, 12, 15):
        for weak_gross in (0.10, 0.20, 0.40):
            portfolio_rows.append(
                evaluate_simple(
                    harness,
                    arrays,
                    protocol,
                    {**score_params, "max_positions": width, "weak_gross": weak_gross},
                )
            )
    portfolio_winner = best_row(portfolio_rows)
    portfolio_params = {key: portfolio_winner[key] for key in base}

    exit_rows = []
    for sell_below in (0.80, 0.85, 0.90):
        for advantage in (0.03, 0.05, 0.08):
            for max_hold in (15, 20, 30):
                exit_rows.append(
                    evaluate_simple(
                        harness,
                        arrays,
                        protocol,
                        {
                            **portfolio_params,
                            "sell_score_below": sell_below,
                            "replacement_advantage": advantage,
                            "max_hold_days": max_hold,
                        },
                    )
                )
    final_winner = best_row(exit_rows)
    frozen_params = {key: final_winner[key] for key in base}

    candidate_daily, candidate_actions = run_simple(
        harness,
        arrays,
        protocol,
        frozen_params,
        VALIDATION_END,
        record_actions=True,
    )
    repeat_daily, repeat_actions = run_simple(
        harness,
        arrays,
        protocol,
        frozen_params,
        VALIDATION_END,
        record_actions=True,
    )
    deterministic = {
        "daily": hash_frame(candidate_daily) == hash_frame(repeat_daily),
        "actions": hash_frame(candidate_actions) == hash_frame(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("candidate deterministic replay mismatch")

    candidate_dev = metric_summary(slice_frame(candidate_daily, FIRST_BUY_DATE, DEVELOPMENT_END))
    baseline_validation = metric_summary(
        slice_frame(baseline_daily, VALIDATION_START, VALIDATION_END)
    )
    candidate_validation = metric_summary(
        slice_frame(candidate_daily, VALIDATION_START, VALIDATION_END)
    )

    pd.DataFrame(score_rows).sort_values("development_utility", ascending=False).to_csv(
        OUTPUT_ROOT / "stage1_score_search.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(portfolio_rows).sort_values("development_utility", ascending=False).to_csv(
        OUTPUT_ROOT / "stage2_portfolio_search.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(exit_rows).sort_values("development_utility", ascending=False).to_csv(
        OUTPUT_ROOT / "stage3_exit_search.csv", index=False, encoding="utf-8-sig"
    )
    baseline_daily.to_csv(OUTPUT_ROOT / "baseline_daily.csv", index=False, encoding="utf-8-sig")
    candidate_daily.to_csv(OUTPUT_ROOT / "candidate_daily.csv", index=False, encoding="utf-8-sig")
    baseline_actions.to_csv(OUTPUT_ROOT / "baseline_actions.csv", index=False, encoding="utf-8-sig")
    candidate_actions.to_csv(OUTPUT_ROOT / "candidate_actions.csv", index=False, encoding="utf-8-sig")

    result = {
        "schema_version": 1,
        "status": "completed_research_only",
        "strategy_id": "v260_simple_core_score_tuned_v1",
        "source_strategy": rules["strategy_id"],
        "research_design": {
            "development": [FIRST_BUY_DATE, DEVELOPMENT_END],
            "independent_validation": [VALIDATION_START, VALIDATION_END],
            "validation_used_for_tuning": False,
            "parameter_evaluations": len(score_rows) + len(portfolio_rows) + len(exit_rows),
            "removed_high_overfit_layers": [
                "score_deterioration_entry_gate",
                "score_trend_position_multiplier",
                "trend_warmup",
                "position_warmup",
                "14_15_16_regime_position_schedule",
                "three_level_new_name_target_pct",
            ],
            "retained": [
                "formal 1D and 10D cross-sectional ranks",
                "finite rank smoothing",
                "binary 10-day median market momentum gross gate",
                "fixed position width and normalized target weight",
                "minimum hold, score exit, replacement advantage and maximum hold",
                "production eligibility, T+1 raw open, limit, round lot, cash and costs",
            ],
        },
        "frozen_params_before_validation": frozen_params,
        "stage_winners": {
            "score": score_winner,
            "portfolio": portfolio_winner,
            "exit": final_winner,
        },
        "development": {
            "baseline": baseline_dev,
            "candidate": candidate_dev,
        },
        "validation_2026": {
            "baseline": baseline_validation,
            "candidate": candidate_validation,
            "delta": {
                key: float(candidate_validation[key] - baseline_validation[key])
                for key in ("cumulative_return", "cagr", "sharpe", "max_drawdown", "turnover_annualized", "average_invested_ratio")
            },
        },
        "deterministic_replay": deterministic,
        "access": access,
        "formal_manifests": manifests,
        "production_modified": False,
    }
    (OUTPUT_ROOT / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["frozen_params_before_validation"], ensure_ascii=False))
    print(json.dumps(result["development"], ensure_ascii=False))
    print(json.dumps(result["validation_2026"], ensure_ascii=False))


if __name__ == "__main__":
    main()
