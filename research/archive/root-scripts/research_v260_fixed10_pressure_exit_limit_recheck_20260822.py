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
    "strategy_agent_v260_fixed10_pressure_exit_limit_recheck_20260822"
)
CHECKPOINT_PATH = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_position_pressure_exit_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
EXIT_LIMITS = (1, 2, 3)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered pressure limit recheck")
    base_policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(base_policy)
    results, cache = {}, {}
    for limit in EXIT_LIMITS:
        policy = copy.deepcopy(base_policy)
        policy["score_sell_pressure_limit"] = int(limit)
        candidate_id = f"pressure_limit_{limit}"
        item, daily, actions = harness.run_policy(context, policy)
        results[candidate_id] = item
        cache[candidate_id] = (daily, actions)

    ranked, selected, gates = coordinate.select_coordinate(
        results, "pressure_limit_2"
    )
    selected_result = results[selected]
    selected_daily, selected_actions = cache[selected]
    repeat_result, repeat_daily, repeat_actions = harness.run_policy(
        context, selected_result["policy"]
    )
    deterministic = {
        "daily": round1.frame_hash(selected_daily) == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(selected_actions)
        == round1.frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("pressure limit replay failed")

    result = {
        "status": "pressure_exit_limit_recheck_complete_2026_not_opened",
        "source_strategy": context.rules["strategy_id"],
        "development_boundary": ["20220607", round1.DEVELOPMENT_END],
        "only_change": "maximum score exits on a trigger-4 pressure day",
        "candidate_budget": list(EXIT_LIMITS),
        "results": results,
        "training_ranked": ranked,
        "confirmation_gates": gates,
        "selected_candidate": selected,
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
