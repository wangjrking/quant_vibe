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
import research_v260_fixed10_weak_market_defensive_rerank_20260822 as defensive


CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_rank_sizing_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_low_vol_entry_sizing_profit_optimization_20260823"
)
VOLATILITY_WINDOW = 20


def low_volatility_multipliers(
    score_order: np.ndarray,
    volatility: np.ndarray,
    positions: int = 10,
) -> tuple[np.ndarray, dict]:
    if score_order.ndim != 2 or volatility.shape != score_order.shape:
        raise ValueError("score order and volatility must share a two-dimensional shape")
    if positions <= 1 or positions > score_order.shape[1]:
        raise ValueError("positions must fit the ranking matrix")
    result = np.ones(score_order.shape, dtype=np.float64)
    dates_with_missing = 0
    missing_observations = 0
    dates_with_fewer_than_two_available = 0
    for row in range(score_order.shape[0]):
        selected = np.asarray(score_order[row, :positions], dtype=np.int64)
        values = volatility[row, selected]
        finite = np.isfinite(values)
        available = selected[finite]
        missing = int((~finite).sum())
        missing_observations += missing
        dates_with_missing += int(missing > 0)
        if len(available) < 2:
            dates_with_fewer_than_two_available += 1
            continue
        available_values = values[finite]
        low_to_high = available[
            np.lexsort((available, available_values))
        ]
        schedule = np.linspace(
            1.10, 0.90, len(low_to_high), dtype=np.float64
        )
        result[row, low_to_high] = schedule
    return result, {
        "dates_with_any_missing_top10_volatility": dates_with_missing,
        "missing_top10_volatility_observations": missing_observations,
        "dates_with_fewer_than_two_available": dates_with_fewer_than_two_available,
    }


def run_low_vol(context, policy: dict, cost: float, multipliers: np.ndarray):
    extra_age, sell_priority = sizing.width_tools.build_overrides(context, policy)
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


def select_by_pre2026_net_return(results: dict[str, dict]) -> str:
    allowed = {
        "equalweight",
        "global_score_rank_sizing",
        "low_volatility_rank_sizing",
    }
    if set(results) != allowed:
        raise ValueError("low-vol sizing comparison arms drifted")
    return max(
        results,
        key=lambda name: (float(results[name]["cumulative_return"]), name),
    )


def main() -> None:
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint.get("validation_2026_opened") is not False:
        raise PermissionError("2026 entered low-vol entry sizing optimization")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = sizing.width_tools.harness.load_context(policy)
    volatility = defensive.trailing_log_volatility(
        context.arrays["close_qfq"], VOLATILITY_WINDOW
    )
    multipliers, availability = low_volatility_multipliers(
        context.order, volatility, sizing.TARGET_POSITIONS
    )
    results = {}
    deterministic = {}
    for cost_name, cost in (
        ("0_30pct", sizing.BASELINE_COST),
        ("0_65pct", sizing.STRESS_COST),
    ):
        equal_daily, equal_actions = sizing.run_case(
            context, policy, cost, None, None
        )
        score_daily, score_actions = sizing.run_case(
            context,
            policy,
            cost,
            sizing.TOP_WEIGHT_MULTIPLIER,
            sizing.BOTTOM_WEIGHT_MULTIPLIER,
        )
        low_vol_daily, low_vol_actions = run_low_vol(
            context, policy, cost, multipliers
        )
        arms = {
            "equalweight": sizing.evaluate(equal_daily, equal_actions),
            "global_score_rank_sizing": sizing.evaluate(
                score_daily, score_actions
            ),
            "low_volatility_rank_sizing": sizing.evaluate(
                low_vol_daily, low_vol_actions
            ),
        }
        results[cost_name] = {
            "arms": arms,
            "selected_by_net_return": select_by_pre2026_net_return(arms),
            "low_vol_minus_score": sizing.checkpoint_tools.metric_delta(
                arms["low_volatility_rank_sizing"],
                arms["global_score_rank_sizing"],
            ),
            "low_vol_minus_equalweight": sizing.checkpoint_tools.metric_delta(
                arms["low_volatility_rank_sizing"], arms["equalweight"]
            ),
        }
        replay_daily, replay_actions = run_low_vol(
            context, policy, cost, multipliers
        )
        deterministic[cost_name] = {
            "daily": replay_daily.equals(low_vol_daily),
            "actions": replay_actions.equals(low_vol_actions),
        }
    if not all(all(item.values()) for item in deterministic.values()):
        raise RuntimeError("low-vol entry sizing replay drifted")

    selected = results["0_30pct"]["selected_by_net_return"]
    result = {
        "status": "low_vol_entry_sizing_profit_comparison_complete_2026_not_opened",
        "selection_rule": "maximize_pre2026_net_cumulative_return_at_0_30pct_cost",
        "single_new_rule": {
            "application": "new_entries_only",
            "selected_universe": "unchanged global score top10",
            "ranking": "ascending trailing 20-session PIT log volatility",
            "target_schedule": "1.10 down to 0.90",
            "unavailable_behavior": "all multipliers remain 1.00 for that date",
            "parameter_search": False,
        },
        "volatility_availability": availability,
        "results": results,
        "selected_arm": selected,
        "selected_changes_frozen_candidate": selected
        == "low_volatility_rank_sizing",
        "deterministic_replay": deterministic,
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    round1.atomic_json(OUTPUT_ROOT / "development_result.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_arm": selected,
                "baseline_low_vol_minus_score": results["0_30pct"][
                    "low_vol_minus_score"
                ],
                "stress_low_vol_minus_score": results["0_65pct"][
                    "low_vol_minus_score"
                ],
                "volatility_availability": availability,
                "validation_2026_opened": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
