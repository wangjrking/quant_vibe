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
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness
import research_v260_fixed10_pressure_regime_trigger_20260822 as pressure
import research_v260_fixed10_regime_exit_threshold_20260822 as regime
import research_v260_lowrisk_score_tuning_20260821 as research_base


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_regime_pressure_trigger_boundary_20260822"
)
CHECKPOINT_PATH = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_regime_pressure_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
CANDIDATES = ("trigger4_strong_trigger5_weak", "trigger4_strong_trigger6_weak")


def pressure_trigger_schedule(strong_market_mask, weak_trigger: int) -> np.ndarray:
    strong = np.asarray(strong_market_mask, dtype=np.bool_)
    if weak_trigger not in {5, 6}:
        raise ValueError("weak trigger is outside the frozen boundary")
    return np.where(strong, 4, int(weak_trigger)).astype(np.int16)


def selection_gates(candidate: dict, baseline: dict) -> dict[str, bool]:
    current = candidate["metrics_0_30pct"]
    reference = baseline["metrics_0_30pct"]
    return {
        "cumulative_return_improved": current["cumulative_return"]
        > reference["cumulative_return"],
        "cagr_improved": current["cagr"] > reference["cagr"],
        "sharpe_improved": current["sharpe"] > reference["sharpe"],
        "drawdown_not_worse": current["max_drawdown"]
        <= reference["max_drawdown"],
        "stress_not_worse": candidate["metrics_0_65pct"]["cumulative_return"]
        >= baseline["metrics_0_65pct"]["cumulative_return"],
        "all_years_positive": all(
            value > 0 for value in current["annual_returns"].values()
        ),
        "full_10_positions": current["full_10_position_ratio"] == 1.0,
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 validation is not available for trigger research")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)
    strong = regime.strong_market_mask(context.score, context.protocol)
    schedules = {
        "trigger4_strong_trigger5_weak": pressure_trigger_schedule(strong, 5),
        "trigger4_strong_trigger6_weak": pressure_trigger_schedule(strong, 6),
    }
    results, cache = {}, {}
    for candidate_id in CANDIDATES:
        results[candidate_id], daily, actions = pressure.run_policy(
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
        raise RuntimeError("regime trigger-boundary baseline drifted")
    gates = selection_gates(candidate, baseline)
    selected = candidate_id if all(gates.values()) else baseline_id
    repeat, repeat_daily, repeat_actions = pressure.run_policy(
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
        raise RuntimeError("regime trigger-boundary replay failed")

    selected_policy = copy.deepcopy(policy)
    selected_policy["score_sell_pressure_trigger_rule"] = (
        "4_strong_market_5_weak_market"
        if selected == baseline_id
        else "4_strong_market_6_weak_market"
    )
    result = {
        "status": "regime_pressure_trigger_boundary_complete_2026_not_opened",
        "source_strategy": context.rules["strategy_id"],
        "development_boundary": [
            research_base.FIRST_BUY,
            round1.DEVELOPMENT_END,
        ],
        "candidate_budget": list(CANDIDATES),
        "only_change": "raise the weak-market pressure trigger from five to six",
        "strong_market_days": int(np.sum(strong)),
        "weak_market_days": int(np.sum(~strong)),
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
