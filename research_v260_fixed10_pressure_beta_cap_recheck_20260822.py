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

import research_v260_fixed10_entry_beta_guard_20260822 as beta_guard
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_portfolio_rebalance_cadence_20260822 as cadence
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness
import research_v260_lowrisk_score_tuning_20260821 as research_base
from research_v260_runtime import fixed10_risk_event_v110 as runtime


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_pressure_beta_cap_recheck_20260822"
)
CHECKPOINT_PATH = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_position_pressure_exit_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
CANDIDATES = ("beta_cap_off", "beta_cap_150")
BETA_CAP = 1.50


def run_with_mask(context, policy, selection_mask):
    threshold = policy.get("maintenance_topup_requires_score")
    maintenance_block = (
        context.empty_block
        if threshold is None
        else ~np.isfinite(context.score) | (context.score < float(threshold))
    )
    common = dict(
        simulator=runtime.simulate,
        portfolio_rebalance_active_override=cadence.rebalance_schedule(
            len(context.arrays["dates"]),
            int(policy["portfolio_rebalance_interval_days"]),
        ),
        portfolio_rebalance_min_weight_deviation_override=policy[
            "portfolio_rebalance_min_weight_deviation"
        ],
        entry_block_mask_override=context.empty_block,
        maintenance_buy_block_mask_override=maintenance_block,
        selection_mask_override=selection_mask,
        score_sell_pressure_trigger_override=policy["score_sell_pressure_trigger"],
        score_sell_pressure_limit_override=policy["score_sell_pressure_limit"],
        score_sell_pressure_confirmation_days_override=policy.get(
            "score_sell_pressure_confirmation_days", 1
        ),
    )
    daily, actions = round1.run_fixed10(
        context.harness,
        context.arrays,
        context.protocol,
        context.definition,
        context.score,
        context.order,
        policy,
        round1.DEVELOPMENT_END,
        slip=round1.BASELINE_COST,
        record_actions=True,
        **common,
    )
    stress_daily, stress_actions = round1.run_fixed10(
        context.harness,
        context.arrays,
        context.protocol,
        context.definition,
        context.score,
        context.order,
        policy,
        round1.DEVELOPMENT_END,
        slip=round1.STRESS_COST,
        record_actions=True,
        **common,
    )
    return {
        "policy": copy.deepcopy(policy),
        "metrics_0_30pct": round1.evaluate_run(
            daily, actions, research_base.FIRST_BUY, round1.DEVELOPMENT_END
        ),
        "train_2022_2024": round1.evaluate_run(
            daily, actions, research_base.FIRST_BUY, "20241231"
        ),
        "holdout_2025": round1.evaluate_run(
            daily, actions, "20250102", round1.DEVELOPMENT_END
        ),
        "metrics_0_65pct": round1.evaluate_run(
            stress_daily,
            stress_actions,
            research_base.FIRST_BUY,
            round1.DEVELOPMENT_END,
        ),
    }, daily, actions


def select_candidate(results):
    baseline = results["beta_cap_off"]
    candidate = results["beta_cap_150"]
    training_improved = harness.training_key(candidate) > harness.training_key(baseline)
    gates = harness.confirmation_gates(candidate, baseline)
    selected = "beta_cap_150" if training_improved and all(gates.values()) else "beta_cap_off"
    return selected, {"training_improved": training_improved, **gates}


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered pressure beta-cap recheck")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)
    production_mask = context.harness.v174.selection_mask(
        context.arrays, context.definition["max_rank_deterioration"]
    )
    beta = beta_guard.trailing_market_beta(
        context.arrays["close_qfq"],
        beta_guard.BETA_LOOKBACK,
        beta_guard.BETA_MIN_OBSERVATIONS,
    )
    masks = {
        "beta_cap_off": production_mask,
        "beta_cap_150": production_mask
        & (~np.isfinite(beta) | (beta <= BETA_CAP)),
    }
    results, cache = {}, {}
    for candidate_id in CANDIDATES:
        item, daily, actions = run_with_mask(context, policy, masks[candidate_id])
        item["policy"] = {
            **item["policy"],
            "new_entry_market_beta_max": (
                None if candidate_id == "beta_cap_off" else BETA_CAP
            ),
            "market_beta_lookback": beta_guard.BETA_LOOKBACK,
            "market_beta_min_observations": beta_guard.BETA_MIN_OBSERVATIONS,
        }
        item["blocked_entry_key_count"] = int(
            np.sum(production_mask & ~masks[candidate_id])
        )
        results[candidate_id] = item
        cache[candidate_id] = (daily, actions)

    selected, gates = select_candidate(results)
    repeat_result, repeat_daily, repeat_actions = run_with_mask(
        context, policy, masks[selected]
    )
    deterministic = {
        "daily": round1.frame_hash(cache[selected][0])
        == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(cache[selected][1])
        == round1.frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("pressure beta-cap replay failed")
    result = {
        "status": "pressure_beta_cap_recheck_complete_2026_not_opened",
        "source_strategy": context.rules["strategy_id"],
        "development_boundary": [research_base.FIRST_BUY, round1.DEVELOPMENT_END],
        "candidate_budget": list(CANDIDATES),
        "only_change": (
            "skip new entries with trailing PIT market beta above 1.50; "
            "pressure exits, full investment, equal weights and all other rules stay fixed"
        ),
        "results": results,
        "selection_gates": gates,
        "selected_candidate": selected,
        "selected_policy": results[selected]["policy"],
        "changed_from_checkpoint": selected != "beta_cap_off",
        "repeat_metrics_equal": all(
            results[selected]["metrics_0_30pct"][key]
            == repeat_result["metrics_0_30pct"][key]
            for key in ("cagr", "sharpe", "max_drawdown")
        ),
        "deterministic_replay": deterministic,
        "data_access": context.access,
        "source_manifests": context.manifests,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "development_result.json", result)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
