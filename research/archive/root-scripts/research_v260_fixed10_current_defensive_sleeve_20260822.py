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
import research_v260_fixed10_pressure_defensive_sleeve_20260822 as sleeve
import research_v260_fixed10_pressure_extra_age_guard_20260822 as age_guard
import research_v260_fixed10_pressure_extra_age_regime_decomposition_20260822 as decomposition
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness
import research_v260_fixed10_regime_exit_threshold_20260822 as regime
import research_v260_fixed10_weak_market_defensive_rerank_20260822 as defensive


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_current_defensive_sleeve_20260822"
)
CHECKPOINT_PATH = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_weak_market_pressure_age_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
CANDIDATES = ("production_order", "weak_market_8alpha_2defensive")


def dominates(candidate: dict, baseline: dict) -> dict[str, bool]:
    current = candidate["metrics_0_30pct"]
    reference = baseline["metrics_0_30pct"]
    return {
        "return_improved": current["cumulative_return"] > reference["cumulative_return"],
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
        raise PermissionError("2026 validation is not available for sleeve research")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)
    strong = regime.strong_market_mask(context.score, context.protocol)
    weak = ~strong
    extra_age = decomposition.age_schedule(strong, 4, 8)
    volatility = defensive.trailing_log_volatility(
        context.arrays["close_qfq"], defensive.VOLATILITY_WINDOW
    )
    defensive_order = sleeve.defensive_sleeve_order(context.order, volatility, weak)
    orders = {"production_order": context.order, "weak_market_8alpha_2defensive": defensive_order}
    results, cache = {}, {}
    for candidate_id in CANDIDATES:
        case_context = copy.copy(context)
        case_context.order = orders[candidate_id]
        results[candidate_id], daily, actions = age_guard.run_policy(
            case_context, policy, extra_age
        )
        cache[candidate_id] = (daily, actions)

    baseline_id, candidate_id = CANDIDATES
    expected = checkpoint["current_best_equalweight"]
    if results[baseline_id]["metrics_0_30pct"]["cumulative_return"] != expected["cumulative_return"]:
        raise RuntimeError("current defensive sleeve baseline drifted")
    gates = dominates(results[candidate_id], results[baseline_id])
    selected = candidate_id if all(gates.values()) else baseline_id
    case_context = copy.copy(context)
    case_context.order = orders[selected]
    repeat, repeat_daily, repeat_actions = age_guard.run_policy(
        case_context, policy, extra_age
    )
    deterministic = {
        "daily": round1.frame_hash(cache[selected][0]) == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(cache[selected][1]) == round1.frame_hash(repeat_actions),
        "metrics": results[selected]["metrics_0_30pct"]["cumulative_return"]
        == repeat["metrics_0_30pct"]["cumulative_return"],
    }
    if not all(deterministic.values()):
        raise RuntimeError("current defensive sleeve replay failed")
    result = {
        "status": "current_defensive_sleeve_complete_2026_not_opened",
        "candidate_budget": list(CANDIDATES),
        "only_change": (
            "on weak-market days keep the first eight production-ranked names and "
            "fill two entry slots with the lowest-volatility names in the top 30"
        ),
        "weak_market_days": int(np.sum(weak)),
        "results": results,
        "selection_gates": gates,
        "selected_candidate": selected,
        "changed_from_checkpoint": selected != baseline_id,
        "deterministic_replay": deterministic,
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "development_result.json", result)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
