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
    "strategy_agent_v260_fixed10_score_noise_attribution_20260822"
)
CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
SIGMA = 0.0025
SEEDS = tuple(range(10))


def run(context, policy: dict):
    strong = regime.strong_market_mask(context.score, context.protocol)
    weak2 = confirmed.confirmed_weak_mask(strong, 2)
    volatility = defensive.trailing_log_volatility(
        context.arrays["close_qfq"], defensive.VOLATILITY_WINDOW
    )
    volatility_rank = percentile.cross_sectional_percent_rank(volatility)
    priority_matrix = priority.sell_priority_matrix(
        context.score, volatility_rank, ~weak2, 0.05
    )
    return age_guard.run_policy_at_cost(
        context,
        policy,
        age_boundary.pressure_age_schedule(strong, 10),
        round1.BASELINE_COST,
        pressure_trigger_override=trigger.pressure_trigger_schedule(strong),
        score_sell_priority_override=priority_matrix,
    )


def evaluate(context, policy: dict) -> dict:
    daily, actions = run(context, policy)
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
    result["exactly10_fraction"] = float(np.mean([
        row["full_10_position_ratio"] == 1.0 for row in rows
    ]))
    return result


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered score-noise attribution")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)
    baseline = evaluate(context, policy)
    expected = checkpoint["current_best_equalweight"]
    equivalent = {
        key: bool(np.isclose(baseline[key], expected[key], rtol=0.0, atol=1e-12))
        for key in (
            "cumulative_return", "cagr", "sharpe", "max_drawdown",
            "turnover_annualized", "average_invested_ratio",
        )
    }
    if not all(equivalent.values()):
        raise RuntimeError("noise-attribution baseline drifted")

    rows = {"order_only": [], "score_values_only": [], "score_and_order": []}
    for seed in SEEDS:
        perturbed = noise.perturbed_context(context, SIGMA, seed)

        order_only = copy.copy(context)
        order_only.order = perturbed.order
        rows["order_only"].append(evaluate(order_only, policy))

        score_only = copy.copy(context)
        score_only.score = perturbed.score
        rows["score_values_only"].append(evaluate(score_only, policy))

        rows["score_and_order"].append(evaluate(perturbed, policy))

    summary = {key: summarize(value, baseline) for key, value in rows.items()}
    repeat = noise.perturbed_context(context, SIGMA, 7)
    repeat_score_only = copy.copy(context)
    repeat_score_only.score = repeat.score
    replay = evaluate(repeat_score_only, policy)
    deterministic = all(
        np.isclose(replay[key], rows["score_values_only"][7][key], atol=1e-12, rtol=0.0)
        for key in ("cumulative_return", "cagr", "sharpe", "max_drawdown")
    )
    if not deterministic:
        raise RuntimeError("noise-attribution replay failed")
    result = {
        "status": "score_noise_attribution_complete_2026_not_opened",
        "sigma": SIGMA,
        "baseline": baseline,
        "summary": summary,
        "raw_runs": rows,
        "baseline_equivalence": equivalent,
        "deterministic_replay": deterministic,
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "score_noise_attribution.json", result)
    print(json.dumps({
        "status": result["status"],
        "summary": summary,
        "deterministic_replay": deterministic,
    }, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
