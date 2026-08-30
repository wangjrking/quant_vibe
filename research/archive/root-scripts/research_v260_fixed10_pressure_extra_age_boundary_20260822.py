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
import research_v260_fixed10_pressure_extra_age_guard_20260822 as age_guard
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_pressure_extra_age_boundary_20260822"
)
CHECKPOINT_PATH = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_pressure_extra_age_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
AGES = (6, 8, 10)


def dominates(candidate: dict, baseline: dict) -> dict[str, bool]:
    current = candidate["metrics_0_30pct"]
    reference = baseline["metrics_0_30pct"]
    return {
        "cumulative_return_improved": current["cumulative_return"] > reference["cumulative_return"],
        "cagr_improved": current["cagr"] > reference["cagr"],
        "sharpe_improved": current["sharpe"] > reference["sharpe"],
        "drawdown_not_worse": current["max_drawdown"] <= reference["max_drawdown"],
        "stress_not_worse": candidate["metrics_0_65pct"]["cumulative_return"]
        >= baseline["metrics_0_65pct"]["cumulative_return"],
        "all_years_positive": min(current["annual_returns"].values()) > 0.0,
        "full_10_positions": current["full_10_position_ratio"] == 1.0,
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 validation is not available for age boundary")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)
    results, cache = {}, {}
    for age in AGES:
        result, daily, actions = age_guard.run_policy(context, policy, age)
        key = f"extra_sell_min_age{age}"
        results[key] = result
        cache[key] = (daily, actions)

    baseline_id = "extra_sell_min_age8"
    gates = {
        key: dominates(value, results[baseline_id])
        for key, value in results.items()
        if key != baseline_id
    }
    eligible = [key for key, value in gates.items() if all(value.values())]
    selected = (
        max(eligible, key=lambda key: results[key]["metrics_0_30pct"]["sharpe"])
        if eligible
        else baseline_id
    )
    repeat, repeat_daily, repeat_actions = age_guard.run_policy(
        context, policy, int(selected.rsplit("age", 1)[1])
    )
    deterministic = {
        "daily": round1.frame_hash(cache[selected][0]) == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(cache[selected][1]) == round1.frame_hash(repeat_actions),
        "metrics": all(
            results[selected]["metrics_0_30pct"][key]
            == repeat["metrics_0_30pct"][key]
            for key in ("cumulative_return", "cagr", "sharpe", "max_drawdown")
        ),
    }
    if not all(deterministic.values()):
        raise RuntimeError("extra-age boundary replay failed")
    checkpoint_metrics = checkpoint["current_best_equalweight"]
    checkpoint_equivalence = all(
        results[baseline_id]["metrics_0_30pct"][key] == checkpoint_metrics[key]
        for key in ("cumulative_return", "cagr", "sharpe", "max_drawdown")
    )
    if not checkpoint_equivalence:
        raise RuntimeError("extra-age boundary checkpoint drifted")
    result = {
        "status": "pressure_extra_age_boundary_complete_2026_not_opened",
        "candidate_budget": list(AGES),
        "selection_rule": "age 6 and 10 are robustness controls; replace age 8 only on broad dominance",
        "results": results,
        "dominance_gates_vs_age8": gates,
        "selected_candidate": selected,
        "changed_from_checkpoint": selected != baseline_id,
        "checkpoint_equivalence": checkpoint_equivalence,
        "deterministic_replay": deterministic,
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "development_result.json", result)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
