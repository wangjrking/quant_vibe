from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_confirmed_severe_weak_defensive_sleeve_20260822 as confirmed
import research_v260_fixed10_current_weak_pressure_age_boundary_20260822 as age_boundary
import research_v260_fixed10_entry_beta_tail_guard_20260822 as percentile
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_pressure_extra_age_guard_20260822 as age_guard
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness
import research_v260_fixed10_regime_exit_threshold_20260822 as regime
import research_v260_fixed10_weak_market_defensive_rerank_20260822 as defensive
import research_v260_fixed10_weak_market_volatility_sell_priority_20260822 as priority
import research_v260_lowrisk_score_tuning_20260821 as research_base


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_paired_block_bootstrap_20260822"
)
CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
PREVIOUS_CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_confirmed_weak_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
PRODUCTION_DAILY = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_hold_optimization_20260822/production_daily.csv"
)
BLOCK_LENGTHS = (5, 20, 60)
REPLICATIONS = 10_000
SEED = 260_202_608_22


def run_policy(context, policy: dict, confirmation_days: int):
    strong = regime.strong_market_mask(context.score, context.protocol)
    confirmed_weak = confirmed.confirmed_weak_mask(strong, confirmation_days)
    extra_age = age_boundary.pressure_age_schedule(strong, 10)
    volatility = defensive.trailing_log_volatility(
        context.arrays["close_qfq"], defensive.VOLATILITY_WINDOW
    )
    volatility_rank = percentile.cross_sectional_percent_rank(volatility)
    priority_matrix = priority.sell_priority_matrix(
        context.score, volatility_rank, ~confirmed_weak, 0.05
    )
    return age_guard.run_policy_at_cost(
        context,
        policy,
        extra_age,
        round1.BASELINE_COST,
        score_sell_priority_override=priority_matrix,
    )


def maximum_drawdown(returns: np.ndarray) -> float:
    equity = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(equity)
    return float(np.max(1.0 - equity / peak))


def sample_metrics(returns: np.ndarray) -> tuple[float, float, float]:
    log_returns = np.log1p(returns)
    annualized_log_return = float(np.mean(log_returns) * 252.0)
    standard_deviation = float(np.std(returns, ddof=1))
    sharpe = (
        float(np.mean(returns) / standard_deviation * np.sqrt(252.0))
        if standard_deviation > 0.0
        else np.nan
    )
    return annualized_log_return, sharpe, maximum_drawdown(returns)


def circular_block_indices(
    rng: np.random.Generator, length: int, block_length: int
) -> np.ndarray:
    block_count = int(np.ceil(length / block_length))
    starts = rng.integers(0, length, size=block_count)
    offsets = np.arange(block_length, dtype=np.int64)
    return ((starts[:, None] + offsets[None, :]) % length).ravel()[:length]


def interval(values: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(values)),
        "p025": float(np.quantile(values, 0.025)),
        "p50": float(np.quantile(values, 0.50)),
        "p975": float(np.quantile(values, 0.975)),
    }


def paired_bootstrap(
    candidate: np.ndarray,
    reference: np.ndarray,
    block_length: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    deltas = np.empty((REPLICATIONS, 3), dtype=np.float64)
    for replication in range(REPLICATIONS):
        indices = circular_block_indices(rng, len(candidate), block_length)
        candidate_metrics = sample_metrics(candidate[indices])
        reference_metrics = sample_metrics(reference[indices])
        deltas[replication] = np.asarray(candidate_metrics) - np.asarray(
            reference_metrics
        )
    return {
        "replications": REPLICATIONS,
        "annualized_log_return_delta": interval(deltas[:, 0]),
        "sharpe_delta": interval(deltas[:, 1]),
        "max_drawdown_delta": interval(deltas[:, 2]),
        "probability_annualized_log_return_higher": float(
            np.mean(deltas[:, 0] > 0.0)
        ),
        "probability_sharpe_higher": float(np.mean(deltas[:, 1] > 0.0)),
        "probability_max_drawdown_lower": float(np.mean(deltas[:, 2] < 0.0)),
    }


def frame_returns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[["date", "return"]].copy()
    result["date"] = result["date"].astype(str)
    return result


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    previous = json.loads(PREVIOUS_CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"] or previous["validation_2026_opened"]:
        raise PermissionError("2026 data entered paired block bootstrap")
    context = harness.load_context(checkpoint["selected_policy"])
    current_daily, current_actions = run_policy(
        context, checkpoint["selected_policy"], 2
    )
    previous_daily, previous_actions = run_policy(
        context, previous["selected_policy"], 3
    )
    current_metrics = round1.evaluate_run(
        current_daily,
        current_actions,
        research_base.FIRST_BUY,
        round1.DEVELOPMENT_END,
    )
    previous_metrics = round1.evaluate_run(
        previous_daily,
        previous_actions,
        research_base.FIRST_BUY,
        round1.DEVELOPMENT_END,
    )
    production = pd.read_csv(PRODUCTION_DAILY, dtype={"date": str})
    joined = (
        frame_returns(current_daily)
        .rename(columns={"return": "current"})
        .merge(
            frame_returns(previous_daily).rename(columns={"return": "previous"}),
            on="date",
            validate="one_to_one",
        )
        .merge(
            frame_returns(production).rename(columns={"return": "production"}),
            on="date",
            validate="one_to_one",
        )
    )
    if len(joined) != int(checkpoint["current_best_equalweight"]["days"]):
        raise RuntimeError("paired block bootstrap date alignment drifted")
    current_equivalence = {
        key: bool(
            np.isclose(
                current_metrics[key],
                checkpoint["current_best_equalweight"][key],
                rtol=0.0,
                atol=1e-12,
            )
        )
        for key in ("cumulative_return", "cagr", "sharpe", "max_drawdown")
    }
    previous_equivalence = {
        key: bool(
            np.isclose(
                previous_metrics[key],
                previous["current_best_equalweight"][key],
                rtol=0.0,
                atol=1e-12,
            )
        )
        for key in ("cumulative_return", "cagr", "sharpe", "max_drawdown")
    }
    if not all(current_equivalence.values()) or not all(previous_equivalence.values()):
        raise RuntimeError("paired block bootstrap strategy replay drifted")

    current = joined["current"].to_numpy(dtype=np.float64)
    prior = joined["previous"].to_numpy(dtype=np.float64)
    production_returns = joined["production"].to_numpy(dtype=np.float64)
    result = {
        "status": "paired_block_bootstrap_complete_2026_not_opened",
        "method": (
            "paired circular moving-block bootstrap over aligned daily returns; "
            "all strategy paths are replayed unchanged"
        ),
        "dates": [str(joined["date"].min()), str(joined["date"].max())],
        "days": int(len(joined)),
        "seed": SEED,
        "comparisons": {
            "current_weak2_vs_production": {
                str(block): paired_bootstrap(
                    current, production_returns, block, SEED + block
                )
                for block in BLOCK_LENGTHS
            },
            "current_weak2_vs_simpler_weak3": {
                str(block): paired_bootstrap(current, prior, block, SEED + 1000 + block)
                for block in BLOCK_LENGTHS
            },
        },
        "current_equivalence": current_equivalence,
        "previous_equivalence": previous_equivalence,
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "paired_block_bootstrap.json", result)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
