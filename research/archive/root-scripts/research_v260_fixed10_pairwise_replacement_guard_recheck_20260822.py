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
    "strategy_agent_v260_fixed10_pairwise_replacement_guard_recheck_20260822"
)
CHECKPOINT_PATH = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_position_pressure_exit_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
CANDIDATES = (False, True)


def select_candidate(results: dict) -> tuple[str, dict[str, bool]]:
    baseline = results["guard_off"]
    candidate = results["guard_on"]
    training_improved = harness.training_key(candidate) > harness.training_key(baseline)
    gates = harness.confirmation_gates(candidate, baseline)
    selected = "guard_on" if training_improved and all(gates.values()) else "guard_off"
    return selected, {"training_improved": training_improved, **gates}


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered pairwise replacement recheck")
    base_policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(base_policy)
    results, cache = {}, {}
    for enabled in CANDIDATES:
        policy = copy.deepcopy(base_policy)
        policy["pairwise_replacement_guard"] = enabled
        candidate_id = "guard_on" if enabled else "guard_off"
        item, daily, actions = harness.run_policy(context, policy)
        results[candidate_id] = item
        cache[candidate_id] = (daily, actions)

    selected, gates = select_candidate(results)
    selected_daily, selected_actions = cache[selected]
    repeat_result, repeat_daily, repeat_actions = harness.run_policy(
        context, results[selected]["policy"]
    )
    deterministic = {
        "daily": round1.frame_hash(selected_daily) == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(selected_actions)
        == round1.frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("pairwise replacement guard replay failed")

    off_actions = cache["guard_off"][1]
    on_actions = cache["guard_on"][1]
    result = {
        "status": "pairwise_replacement_guard_recheck_complete_2026_not_opened",
        "source_strategy": context.rules["strategy_id"],
        "development_boundary": ["20220607", round1.DEVELOPMENT_END],
        "only_change": (
            "each optional score exit consumes its corresponding ranked replacement; "
            "mandatory exits remain outside this optional-exit quality gate"
        ),
        "candidate_budget": ["guard_off", "guard_on"],
        "results": results,
        "selection_gates": gates,
        "selected_candidate": selected,
        "selected_policy": results[selected]["policy"],
        "changed_from_checkpoint": selected == "guard_on",
        "action_row_delta": int(len(on_actions) - len(off_actions)),
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
