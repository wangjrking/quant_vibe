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
import research_v260_fixed10_entry_beta_tail_guard_20260822 as beta_tail
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_portfolio_rebalance_cadence_20260822 as cadence
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness
import research_v260_lowrisk_score_tuning_20260821 as research_base
from research_v260_runtime import fixed10_risk_event_v110 as runtime


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_pressure_beta_score_blend_20260822"
)
CHECKPOINT_PATH = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_position_pressure_exit_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
CANDIDATES = ("production_score", "production95_low_beta05")
PRODUCTION_WEIGHT = 0.95
LOW_BETA_WEIGHT = 0.05


def blended_score(production_score, beta_percentile):
    production = np.asarray(production_score, dtype=np.float64)
    beta_rank = np.asarray(beta_percentile, dtype=np.float64)
    if production.shape != beta_rank.shape:
        raise ValueError("production score and beta percentile do not align")
    result = production.copy()
    usable = np.isfinite(production) & np.isfinite(beta_rank)
    result[usable] = (
        PRODUCTION_WEIGHT * production[usable]
        + LOW_BETA_WEIGHT * (1.0 - beta_rank[usable])
    )
    return result.astype(np.float32)


def stable_descending_order(score):
    values = np.asarray(score, dtype=np.float64)
    safe = np.where(np.isfinite(values), values, -np.inf)
    return np.argsort(-safe, axis=1, kind="stable")


def run_score(context, policy, score, order):
    threshold = policy.get("maintenance_topup_requires_score")
    maintenance_block = (
        context.empty_block
        if threshold is None
        else ~np.isfinite(score) | (score < float(threshold))
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
        score,
        order,
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
        score,
        order,
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
    baseline = results["production_score"]
    candidate = results["production95_low_beta05"]
    training_improved = harness.training_key(candidate) > harness.training_key(baseline)
    gates = harness.confirmation_gates(candidate, baseline)
    selected = (
        "production95_low_beta05"
        if training_improved and all(gates.values())
        else "production_score"
    )
    return selected, {"training_improved": training_improved, **gates}


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered beta score blend")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)
    beta = beta_guard.trailing_market_beta(
        context.arrays["close_qfq"],
        beta_guard.BETA_LOOKBACK,
        beta_guard.BETA_MIN_OBSERVATIONS,
    )
    beta_percentile = beta_tail.cross_sectional_percent_rank(beta)
    candidate_score = blended_score(context.score, beta_percentile)
    scores = {
        "production_score": context.score,
        "production95_low_beta05": candidate_score,
    }
    orders = {
        "production_score": context.order,
        "production95_low_beta05": stable_descending_order(candidate_score),
    }
    results, cache = {}, {}
    for candidate_id in CANDIDATES:
        item, daily, actions = run_score(
            context, policy, scores[candidate_id], orders[candidate_id]
        )
        item["policy"] = {
            **item["policy"],
            "score_formula": (
                {"production_10d": 1.0, "inverse_market_beta_rank": 0.0}
                if candidate_id == "production_score"
                else {
                    "production_10d": PRODUCTION_WEIGHT,
                    "inverse_market_beta_rank": LOW_BETA_WEIGHT,
                }
            ),
        }
        results[candidate_id] = item
        cache[candidate_id] = (daily, actions)

    selected, gates = select_candidate(results)
    repeat_result, repeat_daily, repeat_actions = run_score(
        context, policy, scores[selected], orders[selected]
    )
    deterministic = {
        "daily": round1.frame_hash(cache[selected][0])
        == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(cache[selected][1])
        == round1.frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("beta score blend replay failed")
    result = {
        "status": "pressure_beta_score_blend_complete_2026_not_opened",
        "source_strategy": context.rules["strategy_id"],
        "development_boundary": [research_base.FIRST_BUY, round1.DEVELOPMENT_END],
        "candidate_budget": list(CANDIDATES),
        "only_change": (
            "replace the shared entry/exit score with 95% production 10D score plus "
            "5% inverse trailing market-beta percentile"
        ),
        "results": results,
        "selection_gates": gates,
        "selected_candidate": selected,
        "selected_policy": results[selected]["policy"],
        "changed_from_checkpoint": selected != "production_score",
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
