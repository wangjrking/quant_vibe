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

import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_portfolio_rebalance_cadence_20260822 as cadence
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness
import research_v260_fixed10_regime_exit_threshold_20260822 as regime
import research_v260_fixed10_weak_market_defensive_rerank_20260822 as defensive
import research_v260_lowrisk_score_tuning_20260821 as research_base
from research_v260_runtime import fixed10_risk_event_v110 as runtime


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_pressure_defensive_sleeve_20260822"
)
CHECKPOINT_PATH = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_position_pressure_exit_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
CANDIDATES = ("production_order", "weak_market_8alpha_2defensive")
ALPHA_SLOTS = 8
DEFENSIVE_SLOTS = 2
DEFENSIVE_POOL_SIZE = 30


def defensive_sleeve_order(
    production_order,
    volatility,
    weak_market,
    alpha_slots=ALPHA_SLOTS,
    defensive_slots=DEFENSIVE_SLOTS,
    pool_size=DEFENSIVE_POOL_SIZE,
):
    original = np.asarray(production_order)
    vol = np.asarray(volatility, dtype=np.float64)
    weak = np.asarray(weak_market, dtype=np.bool_)
    if original.shape != vol.shape or weak.shape != (original.shape[0],):
        raise ValueError("defensive sleeve shapes do not align")
    if alpha_slots < 0 or defensive_slots < 0 or alpha_slots + defensive_slots < 1:
        raise ValueError("defensive sleeve slot counts are invalid")
    result = original.copy()
    pool_limit = min(max(int(pool_size), alpha_slots + defensive_slots), original.shape[1])
    for day in np.flatnonzero(weak):
        alpha = [int(idx) for idx in original[day, :alpha_slots]]
        alpha_set = set(alpha)
        pool = [
            int(idx)
            for idx in original[day, :pool_limit]
            if int(idx) not in alpha_set
        ]
        original_position = {int(idx): pos for pos, idx in enumerate(original[day])}
        pool.sort(
            key=lambda idx: (
                float(vol[day, idx]) if np.isfinite(vol[day, idx]) else np.inf,
                original_position[idx],
            )
        )
        selected_defensive = pool[:defensive_slots]
        front = alpha + selected_defensive
        front_set = set(front)
        remainder = [int(idx) for idx in original[day] if int(idx) not in front_set]
        result[day] = np.asarray(front + remainder, dtype=original.dtype)
    return result


def run_order(context, policy, order):
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
        context.score,
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
    baseline = results["production_order"]
    candidate = results["weak_market_8alpha_2defensive"]
    training_improved = harness.training_key(candidate) > harness.training_key(baseline)
    gates = harness.confirmation_gates(candidate, baseline)
    selected = (
        "weak_market_8alpha_2defensive"
        if training_improved and all(gates.values())
        else "production_order"
    )
    return selected, {"training_improved": training_improved, **gates}


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered defensive sleeve development")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)
    volatility = defensive.trailing_log_volatility(
        context.arrays["close_qfq"], defensive.VOLATILITY_WINDOW
    )
    weak_market = ~regime.strong_market_mask(context.score, context.protocol)
    sleeve_order = defensive_sleeve_order(
        context.order, volatility, weak_market
    )
    orders = {
        "production_order": context.order,
        "weak_market_8alpha_2defensive": sleeve_order,
    }
    results, cache = {}, {}
    for candidate_id in CANDIDATES:
        item, daily, actions = run_order(context, policy, orders[candidate_id])
        item["policy"] = {
            **item["policy"],
            "weak_market_defensive_sleeve": (
                None
                if candidate_id == "production_order"
                else {
                    "alpha_slots": ALPHA_SLOTS,
                    "defensive_slots": DEFENSIVE_SLOTS,
                    "production_top_pool": DEFENSIVE_POOL_SIZE,
                    "defensive_sort": "ascending_trailing_20_session_volatility",
                }
            ),
        }
        results[candidate_id] = item
        cache[candidate_id] = (daily, actions)

    selected, gates = select_candidate(results)
    repeat_result, repeat_daily, repeat_actions = run_order(
        context, policy, orders[selected]
    )
    deterministic = {
        "daily": round1.frame_hash(cache[selected][0])
        == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(cache[selected][1])
        == round1.frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("defensive sleeve replay failed")
    result = {
        "status": "pressure_defensive_sleeve_complete_2026_not_opened",
        "source_strategy": context.rules["strategy_id"],
        "development_boundary": [research_base.FIRST_BUY, round1.DEVELOPMENT_END],
        "candidate_budget": list(CANDIDATES),
        "only_change": (
            "on frozen production weak-market days, reserve two of ten entry slots "
            "for the lowest-volatility names inside the production top-30; keep the "
            "first eight production-ranked names and all exit rules unchanged"
        ),
        "weak_market_day_count": int(weak_market.sum()),
        "results": results,
        "selection_gates": gates,
        "selected_candidate": selected,
        "selected_policy": results[selected]["policy"],
        "changed_from_checkpoint": selected != "production_order",
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
