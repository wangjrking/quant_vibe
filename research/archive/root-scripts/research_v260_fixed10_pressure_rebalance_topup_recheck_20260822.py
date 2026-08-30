from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_pressure_hold_renewal_recheck_20260822 as coordinate
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_pressure_rebalance_topup_recheck_20260822"
)
CHECKPOINT_PATH = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_position_pressure_exit_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
REBALANCE_INTERVALS = (10, 20, 40)
TOPUP_SCORES = (0.75, 0.80, 0.85)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered rebalance-topup recheck")
    base_policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(base_policy)

    cadence_results = {}
    for interval in REBALANCE_INTERVALS:
        policy = copy.deepcopy(base_policy)
        policy["portfolio_rebalance_interval_days"] = int(interval)
        cadence_results[f"rebalance_{interval}d"] = harness.run_policy(
            context, policy
        )[0]
    cadence_ranked, cadence_selected, cadence_gates = coordinate.select_coordinate(
        cadence_results, "rebalance_20d"
    )
    selected_policy = copy.deepcopy(cadence_results[cadence_selected]["policy"])

    topup_results = {}
    final_cache = {}
    for threshold in TOPUP_SCORES:
        policy = copy.deepcopy(selected_policy)
        policy["maintenance_topup_requires_score"] = float(threshold)
        candidate_id = f"topup_{int(round(threshold * 100)):03d}"
        item, daily, actions = harness.run_policy(context, policy)
        topup_results[candidate_id] = item
        final_cache[candidate_id] = (daily, actions)
    topup_ranked, topup_selected, topup_gates = coordinate.select_coordinate(
        topup_results, "topup_080"
    )
    selected_result = topup_results[topup_selected]
    selected_daily, selected_actions = final_cache[topup_selected]
    repeat_result, repeat_daily, repeat_actions = harness.run_policy(
        context, selected_result["policy"]
    )
    deterministic = {
        "daily": round1.frame_hash(selected_daily) == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(selected_actions)
        == round1.frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("pressure rebalance-topup replay failed")

    result = {
        "status": "sequential_rebalance_topup_recheck_complete_2026_not_opened",
        "source_strategy": context.rules["strategy_id"],
        "development_boundary": ["20220607", round1.DEVELOPMENT_END],
        "method": "rebalance cadence then top-up quality; no cross-product grid",
        "rebalance_budget": list(REBALANCE_INTERVALS),
        "rebalance_results": cadence_results,
        "rebalance_training_ranked": cadence_ranked,
        "rebalance_confirmation_gates": cadence_gates,
        "selected_rebalance": cadence_selected,
        "topup_score_budget": list(TOPUP_SCORES),
        "topup_results": topup_results,
        "topup_training_ranked": topup_ranked,
        "topup_confirmation_gates": topup_gates,
        "selected_topup": topup_selected,
        "selected_policy": selected_result["policy"],
        "changed_from_checkpoint": selected_result["policy"] != base_policy,
        "repeat_metrics_equal": all(
            selected_result["metrics_0_30pct"][key]
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
