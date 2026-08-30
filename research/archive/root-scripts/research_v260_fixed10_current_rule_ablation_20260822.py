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
import research_v260_fixed10_regime_exit_threshold_20260822 as regime
import research_v260_fixed10_weak_market_defensive_rerank_20260822 as defensive
import research_v260_fixed10_weak_market_volatility_sell_priority_20260822 as priority


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_current_rule_ablation_20260822"
)
CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)


def deltas(candidate: dict, baseline: dict) -> dict:
    return {
        section: {
            key: float(candidate[section][key] - baseline[section][key])
            for key in (
                "cumulative_return",
                "cagr",
                "sharpe",
                "max_drawdown",
                "turnover_annualized",
            )
        }
        for section in (
            "metrics_0_30pct",
            "train_2022_2024",
            "holdout_2025",
            "metrics_0_65pct",
        )
    }


def removal_dominates(candidate: dict, baseline: dict) -> bool:
    return bool(
        candidate["train_2022_2024"]["cagr"]
        >= baseline["train_2022_2024"]["cagr"]
        and candidate["train_2022_2024"]["sharpe"]
        >= baseline["train_2022_2024"]["sharpe"]
        and candidate["holdout_2025"]["cumulative_return"]
        >= baseline["holdout_2025"]["cumulative_return"]
        and candidate["metrics_0_65pct"]["cumulative_return"]
        >= baseline["metrics_0_65pct"]["cumulative_return"]
        and candidate["metrics_0_30pct"]["max_drawdown"]
        <= baseline["metrics_0_30pct"]["max_drawdown"]
    )


def build_cases(context, policy: dict) -> tuple[dict, np.ndarray, np.ndarray]:
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

    cases = {}
    cases["full_current"] = (copy.deepcopy(policy), extra_age, None, priority_matrix)

    pressure_off = copy.deepcopy(policy)
    pressure_off["score_sell_pressure_limit"] = 1
    cases["remove_pressure_second_exit"] = (
        pressure_off,
        extra_age,
        None,
        priority_matrix,
    )

    cases["remove_regime_trigger_use_uniform4"] = (
        copy.deepcopy(policy),
        extra_age,
        4,
        priority_matrix,
    )
    cases["remove_weak_extra_age_slowdown"] = (
        copy.deepcopy(policy),
        4,
        None,
        priority_matrix,
    )
    cases["remove_weak_volatility_sell_priority"] = (
        copy.deepcopy(policy),
        extra_age,
        None,
        context.score,
    )

    maintenance_off = copy.deepcopy(policy)
    maintenance_off["maintenance_topup_requires_score"] = None
    cases["remove_maintenance_score_gate"] = (
        maintenance_off,
        extra_age,
        None,
        priority_matrix,
    )

    renewal_off = copy.deepcopy(policy)
    renewal_off["renewal_policy"] = "none"
    renewal_off["max_hold_renewal_score"] = None
    cases["remove_max_hold_renewal"] = (
        renewal_off,
        extra_age,
        None,
        priority_matrix,
    )

    rebalance_off = copy.deepcopy(policy)
    rebalance_off["portfolio_rebalance_interval_days"] = 10000
    cases["remove_periodic_equalweight_maintenance"] = (
        rebalance_off,
        extra_age,
        None,
        priority_matrix,
    )
    return cases, extra_age, priority_matrix


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered current-rule ablation")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)
    cases, extra_age, priority_matrix = build_cases(context, policy)

    results = {}
    cache = {}
    for case_id, (case_policy, case_age, pressure_trigger, sell_priority) in cases.items():
        result, daily, actions = age_guard.run_policy(
            context,
            case_policy,
            case_age,
            pressure_trigger_override=pressure_trigger,
            score_sell_priority_override=sell_priority,
        )
        results[case_id] = result
        cache[case_id] = (daily, actions)

    baseline = results["full_current"]
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
        raise RuntimeError("current-rule ablation baseline drifted")
    ablations = {
        case_id: {
            "metrics": result,
            "delta_vs_full_current": deltas(result, baseline),
            "removal_dominates_full_current": removal_dominates(result, baseline),
        }
        for case_id, result in results.items()
        if case_id != "full_current"
    }
    dominating_removals = [
        case_id
        for case_id, item in ablations.items()
        if item["removal_dominates_full_current"]
    ]
    repeat, repeat_daily, repeat_actions = age_guard.run_policy(
        context,
        policy,
        extra_age,
        score_sell_priority_override=priority_matrix,
    )
    deterministic = {
        "daily": round1.frame_hash(cache["full_current"][0])
        == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(cache["full_current"][1])
        == round1.frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("current-rule ablation replay failed")
    result = {
        "status": "current_rule_ablation_complete_2026_not_opened",
        "method": (
            "remove one accepted module at a time while leaving all remaining rules "
            "unchanged; a module is removable only if its removal dominates on training, "
            "2025 forward confirmation, stress and drawdown"
        ),
        "full_current": baseline,
        "ablations": ablations,
        "dominating_removals": dominating_removals,
        "simplification_decision": (
            "remove_dominating_modules"
            if dominating_removals
            else "retain_current_modules_no_dominating_removal"
        ),
        "baseline_equivalence": equivalence,
        "deterministic_replay": deterministic,
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "rule_ablation.json", result)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
