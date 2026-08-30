from __future__ import annotations

import copy
import functools
import json
import sys
from pathlib import Path

import numpy as np


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_entry_rank_sizing_profit_optimization_20260822 as sizing
import research_v260_fixed10_hold_optimization_20260822 as round1


CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_rank_sizing_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_friction_compensated_gross_20260823"
)
BUY_COMMISSION = 0.0003


def friction_compensated_gross(slippage_ratio: float) -> float:
    if not 0.0 <= slippage_ratio < 1.0:
        raise ValueError("slippage ratio must be in [0, 1)")
    return float((1.0 + slippage_ratio) * (1.0 + BUY_COMMISSION))


def friction_compensated_entry_multipliers(
    order: np.ndarray,
    slippage_ratio: float,
    positions: int = sizing.TARGET_POSITIONS,
) -> np.ndarray:
    return sizing.entry_rank_multipliers(
        order, positions=positions
    ) * friction_compensated_gross(slippage_ratio)


def run_case(context, policy: dict, cost: float, compensate: bool):
    extra_age, sell_priority = sizing.width_tools.build_overrides(context, policy)
    multipliers = (
        friction_compensated_entry_multipliers(context.order, cost)
        if compensate
        else sizing.entry_rank_multipliers(context.order)
    )
    simulator = functools.partial(
        sizing.runtime.simulate,
        candidate_target_multiplier_override=multipliers,
    )
    return sizing.age_guard.run_policy_at_cost(
        context,
        policy,
        extra_age,
        cost,
        simulator=simulator,
        score_sell_priority_override=sell_priority,
        target_positions_override=sizing.TARGET_POSITIONS,
        target_gross_override=1.0,
    )


def exposure_diagnostics(daily) -> dict:
    invested = daily["invested_ratio"].to_numpy(dtype=np.float64)
    if not np.isfinite(invested).all():
        raise RuntimeError("invested ratio is nonfinite")
    return {
        "mean": float(invested.mean()),
        "minimum": float(invested.min()),
        "maximum": float(invested.max()),
        "above_one_count": int((invested > 1.0 + 1e-12).sum()),
        "below_zero_count": int((invested < -1e-12).sum()),
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint.get("validation_2026_opened") is not False:
        raise PermissionError("2026 entered friction-compensated gross research")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = sizing.width_tools.harness.load_context(policy)
    if context.access["logical_max_date"] != round1.DEVELOPMENT_END:
        raise PermissionError("pre-2026 development boundary drift")

    cases = {}
    frames = {}
    for name, compensate in (
        ("current_nominal_gross_1", False),
        ("friction_compensated_gross", True),
    ):
        daily, actions = run_case(context, policy, sizing.BASELINE_COST, compensate)
        stress_daily, stress_actions = run_case(
            context, policy, sizing.STRESS_COST, compensate
        )
        cases[name] = {
            "new_entry_budget_multiplier_0_30pct": (
                friction_compensated_gross(sizing.BASELINE_COST)
                if compensate
                else 1.0
            ),
            "new_entry_budget_multiplier_0_65pct": (
                friction_compensated_gross(sizing.STRESS_COST)
                if compensate
                else 1.0
            ),
            "metrics_0_30pct": sizing.evaluate(daily, actions),
            "metrics_0_65pct": sizing.evaluate(stress_daily, stress_actions),
            "exposure_0_30pct": exposure_diagnostics(daily),
            "exposure_0_65pct": exposure_diagnostics(stress_daily),
        }
        frames[name] = (daily, actions)

    repeat_daily, repeat_actions = run_case(
        context, policy, sizing.BASELINE_COST, True
    )
    candidate_daily, candidate_actions = frames["friction_compensated_gross"]
    deterministic = {
        "daily": round1.frame_hash(candidate_daily) == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(candidate_actions)
        == round1.frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("friction-compensated gross replay is not deterministic")

    current = cases["current_nominal_gross_1"]
    candidate = cases["friction_compensated_gross"]
    delta = {
        cost: sizing.checkpoint_tools.metric_delta(
            candidate[f"metrics_{cost}"], current[f"metrics_{cost}"]
        )
        for cost in ("0_30pct", "0_65pct")
    }
    no_leverage = all(
        candidate[key]["above_one_count"] == 0
        and candidate[key]["below_zero_count"] == 0
        for key in ("exposure_0_30pct", "exposure_0_65pct")
    )
    profit_upgrade = bool(
        delta["0_30pct"]["cumulative_return"] > 0.0
        and delta["0_65pct"]["cumulative_return"] > 0.0
        and no_leverage
    )
    result = {
        "status": "friction_compensated_gross_complete_2026_not_opened",
        "single_formula": (
            "new_entry_budget_multiplier=(1+slippage_ratio)*"
            "(1+0.0003_buy_commission); account gross cap remains 1.0"
        ),
        "parameter_search_count": 0,
        "portfolio_full_investment_is_soft": True,
        "cases": cases,
        "candidate_minus_current": delta,
        "actual_exposure_never_levered": no_leverage,
        "profit_upgrade_at_both_costs": profit_upgrade,
        "selection_decision": (
            "eligible_for_further_robustness"
            if profit_upgrade
            else "reject_keep_current_frozen_candidate"
        ),
        "deterministic_replay": deterministic,
        "development_boundary": [
            sizing.research_base.FIRST_BUY,
            round1.DEVELOPMENT_END,
        ],
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "development_result.json", result)
    print(json.dumps({
        "status": result["status"],
        "baseline_cumulative_delta": delta["0_30pct"]["cumulative_return"],
        "stress_cumulative_delta": delta["0_65pct"]["cumulative_return"],
        "mean_invested_delta": (
            candidate["exposure_0_30pct"]["mean"]
            - current["exposure_0_30pct"]["mean"]
        ),
        "selection_decision": result["selection_decision"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
