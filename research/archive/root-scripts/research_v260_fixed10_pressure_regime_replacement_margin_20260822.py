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
import research_v260_fixed10_pressure_regime_trigger_20260822 as regime_trigger
import research_v260_fixed10_regime_exit_threshold_20260822 as regime
import research_v260_lowrisk_score_tuning_20260821 as research_base
from research_v260_runtime import fixed10_risk_event_v110 as runtime


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_pressure_regime_replacement_margin_20260822"
)
CHECKPOINT_PATH = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_regime_pressure_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
CANDIDATES = (
    "margin005_all_markets",
    "margin005_strong_margin007_weak",
)


def replacement_margin_schedule(strong_market_mask) -> np.ndarray:
    strong = np.asarray(strong_market_mask, dtype=np.bool_)
    return np.where(strong, 0.05, 0.07).astype(np.float64)


def run_policy(context, policy: dict, replacement_advantage_override):
    strong = regime.strong_market_mask(context.score, context.protocol)
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
        replacement_advantage_override=replacement_advantage_override,
        score_sell_pressure_trigger_override=(
            regime_trigger.pressure_trigger_schedule(strong)
        ),
        score_sell_pressure_limit_override=policy["score_sell_pressure_limit"],
        score_sell_pressure_confirmation_days_override=1,
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


def simplified_gates(candidate: dict, baseline: dict) -> dict[str, bool]:
    candidate_metrics = candidate["metrics_0_30pct"]
    baseline_metrics = baseline["metrics_0_30pct"]
    return {
        "cumulative_return_improved": candidate_metrics["cumulative_return"]
        > baseline_metrics["cumulative_return"],
        "cagr_improved": candidate_metrics["cagr"] > baseline_metrics["cagr"],
        "sharpe_improved": candidate_metrics["sharpe"] > baseline_metrics["sharpe"],
        "drawdown_not_worse": candidate_metrics["max_drawdown"]
        <= baseline_metrics["max_drawdown"],
        "stress_not_worse": candidate["metrics_0_65pct"]["cumulative_return"]
        >= baseline["metrics_0_65pct"]["cumulative_return"],
        "all_years_positive": min(candidate_metrics["annual_returns"].values())
        > 0.0,
        "full_10_positions": candidate_metrics["full_10_position_ratio"] == 1.0,
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 validation is not available for margin research")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)
    strong = regime.strong_market_mask(context.score, context.protocol)
    schedules = {
        "margin005_all_markets": 0.05,
        "margin005_strong_margin007_weak": replacement_margin_schedule(strong),
    }
    results, cache = {}, {}
    for candidate_id in CANDIDATES:
        results[candidate_id], daily, actions = run_policy(
            context, policy, schedules[candidate_id]
        )
        cache[candidate_id] = (daily, actions)

    baseline_id, candidate_id = CANDIDATES
    baseline, candidate = results[baseline_id], results[candidate_id]
    expected = checkpoint["current_best_equalweight"]
    checkpoint_equivalence = {
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
    if not all(checkpoint_equivalence.values()):
        raise RuntimeError("regime-margin baseline drifted")
    gates = simplified_gates(candidate, baseline)
    selected = candidate_id if all(gates.values()) else baseline_id
    repeat, repeat_daily, repeat_actions = run_policy(
        context, policy, schedules[selected]
    )
    deterministic = {
        "daily": round1.frame_hash(cache[selected][0])
        == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(cache[selected][1])
        == round1.frame_hash(repeat_actions),
        "metrics": all(
            results[selected]["metrics_0_30pct"][key]
            == repeat["metrics_0_30pct"][key]
            for key in ("cumulative_return", "cagr", "sharpe", "max_drawdown")
        ),
    }
    if not all(deterministic.values()):
        raise RuntimeError("regime-margin replay failed")

    selected_policy = copy.deepcopy(policy)
    selected_policy["replacement_advantage_rule"] = (
        "0.05_all_markets"
        if selected == baseline_id
        else "0.05_strong_market_0.07_weak_market"
    )
    result = {
        "status": "pressure_regime_replacement_margin_complete_2026_not_opened",
        "source_strategy": context.rules["strategy_id"],
        "development_boundary": [
            research_base.FIRST_BUY,
            round1.DEVELOPMENT_END,
        ],
        "candidate_budget": list(CANDIDATES),
        "only_change": (
            "a weak-market replacement must beat the held score by 0.07 rather "
            "than 0.05; strong-market replacements remain unchanged"
        ),
        "results": results,
        "selection_gates": gates,
        "selected_candidate": selected,
        "selected_policy": selected_policy,
        "changed_from_checkpoint": selected != baseline_id,
        "checkpoint_equivalence": checkpoint_equivalence,
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
