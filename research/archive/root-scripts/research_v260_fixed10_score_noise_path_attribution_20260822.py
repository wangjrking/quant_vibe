from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np


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
import research_v260_fixed10_pressure_regime_trigger_20260822 as trigger
import research_v260_fixed10_regime_exit_threshold_20260822 as regime
import research_v260_fixed10_score_noise_robustness_20260822 as noise
import research_v260_fixed10_weak_market_defensive_rerank_20260822 as defensive
import research_v260_fixed10_weak_market_volatility_sell_priority_20260822 as priority
import research_v260_lowrisk_score_tuning_20260821 as research_base


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_score_noise_path_attribution_20260822"
)
CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
SIGMA = 0.0025
SEEDS = tuple(range(10))


def run(context, policy: dict, order: np.ndarray, priority_matrix):
    strong = regime.strong_market_mask(context.score, context.protocol)
    case = copy.copy(context)
    case.order = order
    return age_guard.run_policy_at_cost(
        case,
        policy,
        age_boundary.pressure_age_schedule(strong, 10),
        round1.BASELINE_COST,
        pressure_trigger_override=trigger.pressure_trigger_schedule(strong),
        score_sell_priority_override=priority_matrix,
    )


def evaluate(context, policy: dict, order: np.ndarray, priority_matrix) -> dict:
    daily, actions = run(context, policy, order, priority_matrix)
    return round1.evaluate_run(
        daily, actions, research_base.FIRST_BUY, round1.DEVELOPMENT_END
    )


def summarize(rows: list[dict], baseline: dict) -> dict:
    result = {"runs": len(rows)}
    for key in (
        "cumulative_return", "cagr", "sharpe", "max_drawdown",
        "turnover_annualized", "average_invested_ratio",
    ):
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        result[key] = {
            "median": float(np.median(values)),
            "p10": float(np.quantile(values, 0.10)),
            "p90": float(np.quantile(values, 0.90)),
            "median_delta": float(np.median(values) - baseline[key]),
        }
    result["all_years_positive_fraction"] = float(np.mean([
        min(row["annual_returns"].values()) > 0.0 for row in rows
    ]))
    return result


def splice_order(
    baseline: np.ndarray,
    perturbed: np.ndarray,
    start: int,
    stop: int,
) -> np.ndarray:
    result = np.asarray(baseline, dtype=np.int64).copy()
    result[int(start):int(stop)] = perturbed[int(start):int(stop)]
    return result


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered score-noise path attribution")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)
    strong = regime.strong_market_mask(context.score, context.protocol)
    weak2 = confirmed.confirmed_weak_mask(strong, 2)
    volatility = defensive.trailing_log_volatility(
        context.arrays["close_qfq"], defensive.VOLATILITY_WINDOW
    )
    volatility_rank = percentile.cross_sectional_percent_rank(volatility)
    priority_matrix = priority.sell_priority_matrix(
        context.score, volatility_rank, ~weak2, 0.05
    )
    baseline = evaluate(context, policy, context.order, priority_matrix)
    expected = checkpoint["current_best_equalweight"]
    equivalent = {
        key: bool(np.isclose(baseline[key], expected[key], rtol=0.0, atol=1e-12))
        for key in (
            "cumulative_return", "cagr", "sharpe", "max_drawdown",
            "turnover_annualized", "average_invested_ratio",
        )
    }
    if not all(equivalent.values()):
        raise RuntimeError("noise-path baseline drifted")

    rows = {
        "first_signal_only": [],
        "first_20_sessions": [],
        "after_first_20_sessions": [],
        "all_sessions": [],
    }
    size = context.order.shape[0]
    for seed in SEEDS:
        perturbed = noise.perturbed_context(context, SIGMA, seed).order
        orders = {
            "first_signal_only": splice_order(context.order, perturbed, 0, 1),
            "first_20_sessions": splice_order(context.order, perturbed, 0, 20),
            "after_first_20_sessions": splice_order(context.order, perturbed, 20, size),
            "all_sessions": perturbed,
        }
        for key, order in orders.items():
            rows[key].append(evaluate(context, policy, order, priority_matrix))

    summary = {key: summarize(value, baseline) for key, value in rows.items()}
    result = {
        "status": "score_noise_path_attribution_complete_2026_not_opened",
        "sigma": SIGMA,
        "baseline": baseline,
        "summary": summary,
        "raw_runs": rows,
        "baseline_equivalence": equivalent,
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "score_noise_path_attribution.json", result)
    print(json.dumps({"status": result["status"], "summary": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
