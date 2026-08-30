from __future__ import annotations

import copy
import itertools
import json
import sys
from pathlib import Path

import numpy as np


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_confirmed_severe_weak_defensive_sleeve_20260822 as confirmed
import research_v260_fixed10_current_rule_ablation_20260822 as ablation
import research_v260_fixed10_current_weak_pressure_age_boundary_20260822 as age_boundary
import research_v260_fixed10_entry_beta_tail_guard_20260822 as percentile
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_pressure_extra_age_guard_20260822 as age_guard
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness
import research_v260_fixed10_regime_exit_threshold_20260822 as regime
import research_v260_fixed10_weak_market_defensive_rerank_20260822 as defensive
import research_v260_fixed10_weak_market_volatility_sell_priority_20260822 as priority


REPORTS = REPO / "quant/data_file/reports"
CHECKPOINT = (
    REPORTS
    / "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
OUTPUT_ROOT = (
    REPORTS / "strategy_agent_v260_fixed10_pairwise_rule_interaction_audit_20260822"
)
RULES = (
    "pressure_second_exit",
    "regime_trigger",
    "weak_extra_age_slowdown",
    "weak_volatility_sell_priority",
    "max_hold_renewal",
    "periodic_equalweight_maintenance",
)
YEARS = ("2022", "2023", "2024", "2025")


def pair_ids() -> list[tuple[str, str]]:
    return list(itertools.combinations(RULES, 2))


def build_components(context) -> dict:
    strong = regime.strong_market_mask(context.score, context.protocol)
    confirmed_weak = confirmed.confirmed_weak_mask(strong, 2)
    extra_age = age_boundary.pressure_age_schedule(strong, 10)
    volatility = defensive.trailing_log_volatility(
        context.arrays["close_qfq"], defensive.VOLATILITY_WINDOW
    )
    volatility_rank = percentile.cross_sectional_percent_rank(volatility)
    priority_matrix = priority.sell_priority_matrix(
        context.score, volatility_rank, ~confirmed_weak, 0.05
    )
    return {
        "extra_age": extra_age,
        "priority_matrix": priority_matrix,
    }


def apply_removals(
    policy: dict,
    context,
    components: dict,
    removals: tuple[str, ...],
) -> tuple[dict, object, object, object]:
    unknown = set(removals) - set(RULES)
    if unknown:
        raise ValueError(f"unknown rule removals: {sorted(unknown)}")
    case_policy = copy.deepcopy(policy)
    extra_age = components["extra_age"]
    pressure_trigger = None
    sell_priority = components["priority_matrix"]

    if "pressure_second_exit" in removals:
        case_policy["score_sell_pressure_limit"] = 1
    if "regime_trigger" in removals:
        pressure_trigger = 4
    if "weak_extra_age_slowdown" in removals:
        extra_age = 4
    if "weak_volatility_sell_priority" in removals:
        sell_priority = context.score
    if "max_hold_renewal" in removals:
        case_policy["renewal_policy"] = "none"
        case_policy["max_hold_renewal_score"] = None
    if "periodic_equalweight_maintenance" in removals:
        case_policy["portfolio_rebalance_interval_days"] = 10000
    return case_policy, extra_age, pressure_trigger, sell_priority


def support_summary(full: dict, removed: dict) -> dict:
    yearly_gain = {
        year: float(
            full["metrics_0_30pct"]["annual_returns"][year]
            - removed["metrics_0_30pct"]["annual_returns"][year]
        )
        for year in YEARS
    }
    gains = {
        "overall_cumulative_return": float(
            full["metrics_0_30pct"]["cumulative_return"]
            - removed["metrics_0_30pct"]["cumulative_return"]
        ),
        "train_2022_2024_cumulative_return": float(
            full["train_2022_2024"]["cumulative_return"]
            - removed["train_2022_2024"]["cumulative_return"]
        ),
        "holdout_2025_cumulative_return": float(
            full["holdout_2025"]["cumulative_return"]
            - removed["holdout_2025"]["cumulative_return"]
        ),
        "stress_cumulative_return": float(
            full["metrics_0_65pct"]["cumulative_return"]
            - removed["metrics_0_65pct"]["cumulative_return"]
        ),
    }
    positive_partitions = sum(value > 0.0 for value in gains.values())
    return {
        "yearly_return_gain_with_pair": yearly_gain,
        "positive_year_count": sum(value > 0.0 for value in yearly_gain.values()),
        "partition_gains_with_pair": gains,
        "positive_partition_count": positive_partitions,
        "pair_jointly_supported": positive_partitions == len(gains),
        "pair_removal_dominates_full": ablation.removal_dominates(removed, full),
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint.get("validation_2026_opened") is not False:
        raise PermissionError("pairwise rule audit cannot consume 2026 validation")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)
    components = build_components(context)

    baseline_policy, baseline_age, baseline_trigger, baseline_priority = apply_removals(
        policy, context, components, ()
    )
    baseline, baseline_daily, baseline_actions = age_guard.run_policy(
        context,
        baseline_policy,
        baseline_age,
        pressure_trigger_override=baseline_trigger,
        score_sell_priority_override=baseline_priority,
    )
    expected = checkpoint["current_best_equalweight"]
    equivalence = {
        key: bool(
            np.isclose(
                baseline["metrics_0_30pct"][key],
                expected[key],
                rtol=0.0,
                atol=1e-12,
            )
        )
        for key in (
            "cumulative_return",
            "cagr",
            "sharpe",
            "max_drawdown",
            "turnover_annualized",
            "average_invested_ratio",
        )
    }
    if not all(equivalence.values()):
        raise RuntimeError("pairwise rule audit baseline drifted")

    pairs = {}
    for pair in pair_ids():
        case_policy, extra_age, pressure_trigger, sell_priority = apply_removals(
            policy, context, components, pair
        )
        metrics, _, _ = age_guard.run_policy(
            context,
            case_policy,
            extra_age,
            pressure_trigger_override=pressure_trigger,
            score_sell_priority_override=sell_priority,
        )
        pair_id = "+".join(pair)
        pairs[pair_id] = {
            "removed_rules": list(pair),
            "metrics": metrics,
            "support": support_summary(baseline, metrics),
        }

    dominating_pairs = [
        pair_id
        for pair_id, item in pairs.items()
        if item["support"]["pair_removal_dominates_full"]
    ]
    unsupported_pairs = [
        pair_id
        for pair_id, item in pairs.items()
        if not item["support"]["pair_jointly_supported"]
    ]
    repeat, repeat_daily, repeat_actions = age_guard.run_policy(
        context,
        baseline_policy,
        baseline_age,
        pressure_trigger_override=baseline_trigger,
        score_sell_priority_override=baseline_priority,
    )
    deterministic = {
        "daily": round1.frame_hash(baseline_daily) == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(baseline_actions)
        == round1.frame_hash(repeat_actions),
        "metrics": baseline["metrics_0_30pct"] == repeat["metrics_0_30pct"],
    }
    if not all(deterministic.values()):
        raise RuntimeError("pairwise rule audit deterministic replay failed")

    result = {
        "status": "pre2026_pairwise_rule_interaction_audit_complete",
        "method": (
            "remove each of the 15 pairs formed by six frozen core rules; retain the "
            "candidate unless a pair removal dominates training, 2025 holdout, stress "
            "and drawdown"
        ),
        "pair_count": len(pairs),
        "pairs": pairs,
        "dominating_pair_removals": dominating_pairs,
        "pairs_without_all_partition_support": unsupported_pairs,
        "decision": (
            "simplify_candidate_remove_dominating_pair"
            if dominating_pairs
            else "retain_frozen_candidate_no_pairwise_dominating_removal"
        ),
        "candidate_changed": False,
        "baseline_equivalence": equivalence,
        "deterministic_replay": deterministic,
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "pairwise_rule_interaction_audit.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "pair_count": result["pair_count"],
                "dominating_pair_removals": dominating_pairs,
                "pairs_without_all_partition_support": unsupported_pairs,
                "decision": result["decision"],
                "validation_2026_opened": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
