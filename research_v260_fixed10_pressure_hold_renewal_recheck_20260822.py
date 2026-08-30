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
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_pressure_hold_renewal_recheck_20260822"
)
CHECKPOINT_PATH = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_position_pressure_exit_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
MIN_HOLDS = (3, 4, 5)
RENEWAL_SCORES = (0.75, 0.80, 0.85)


def select_coordinate(results: dict, baseline_id: str) -> tuple[str, str, dict]:
    best_key = max(harness.training_key(item) for item in results.values())
    ranked_id = (
        baseline_id
        if harness.training_key(results[baseline_id]) == best_key
        else max(results, key=lambda key: harness.training_key(results[key]))
    )
    gates = harness.confirmation_gates(results[ranked_id], results[baseline_id])
    selected_id = ranked_id if all(gates.values()) else baseline_id
    return ranked_id, selected_id, gates


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered hold-renewal recheck")
    base_policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(base_policy)

    min_hold_results = {}
    for min_hold in MIN_HOLDS:
        policy = copy.deepcopy(base_policy)
        policy["min_hold_mode"] = int(min_hold)
        min_hold_results[f"min_hold_{min_hold}"] = harness.run_policy(
            context, policy
        )[0]
    min_ranked, min_selected, min_gates = select_coordinate(
        min_hold_results, "min_hold_4"
    )
    selected_policy = copy.deepcopy(min_hold_results[min_selected]["policy"])

    renewal_results = {}
    final_cache = {}
    for renewal_score in RENEWAL_SCORES:
        policy = copy.deepcopy(selected_policy)
        policy["max_hold_renewal_score"] = float(renewal_score)
        candidate_id = f"renewal_{int(round(renewal_score * 100)):03d}"
        item, daily, actions = harness.run_policy(context, policy)
        renewal_results[candidate_id] = item
        final_cache[candidate_id] = (daily, actions)
    renewal_ranked, renewal_selected, renewal_gates = select_coordinate(
        renewal_results, "renewal_080"
    )
    selected_result = renewal_results[renewal_selected]
    selected_daily, selected_actions = final_cache[renewal_selected]
    repeat_result, repeat_daily, repeat_actions = harness.run_policy(
        context, selected_result["policy"]
    )
    deterministic = {
        "daily": round1.frame_hash(selected_daily) == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(selected_actions)
        == round1.frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("pressure hold-renewal replay failed")

    result = {
        "status": "sequential_hold_renewal_recheck_complete_2026_not_opened",
        "source_strategy": context.rules["strategy_id"],
        "development_boundary": ["20220607", round1.DEVELOPMENT_END],
        "method": "min hold then renewal threshold; no cross-product grid",
        "min_hold_budget": list(MIN_HOLDS),
        "min_hold_results": min_hold_results,
        "min_hold_training_ranked": min_ranked,
        "min_hold_confirmation_gates": min_gates,
        "selected_min_hold": min_selected,
        "renewal_score_budget": list(RENEWAL_SCORES),
        "renewal_results": renewal_results,
        "renewal_training_ranked": renewal_ranked,
        "renewal_confirmation_gates": renewal_gates,
        "selected_renewal_score": renewal_selected,
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
