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
import research_v260_fixed10_paired_block_bootstrap_20260822 as bootstrap


CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_rank_sizing_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_rank_sizing_neutral_placebo_20260823"
)
NEUTRAL_PERMUTATION = (4, 0, 2, 8, 9, 7, 6, 5, 3, 1)
BLOCK_LENGTHS = (5, 20, 60)
BOOTSTRAP_SEED = 2_602_026_082_3


def neutral_schedule() -> np.ndarray:
    descending = np.linspace(1.10, 0.90, 10, dtype=np.float64)
    values = descending[np.asarray(NEUTRAL_PERMUTATION, dtype=np.int64)]
    correlation = float(np.corrcoef(np.arange(10), values)[0, 1])
    if abs(correlation) > 0.007:
        raise RuntimeError("neutral placebo rank correlation drifted")
    if not np.isclose(values.sum(), 10.0, rtol=0.0, atol=1e-12):
        raise RuntimeError("neutral placebo reference gross drifted")
    return values


def schedule_multipliers(order: np.ndarray, schedule: np.ndarray) -> np.ndarray:
    if order.ndim != 2 or schedule.shape != (10,):
        raise ValueError("neutral placebo shapes are invalid")
    result = np.ones(order.shape, dtype=np.float64)
    rows = np.arange(order.shape[0])[:, None]
    result[rows, order[:, :10]] = schedule[None, :]
    return result


def run_neutral(context, policy: dict, cost: float):
    extra_age, sell_priority = sizing.width_tools.build_overrides(context, policy)
    multipliers = schedule_multipliers(context.order, neutral_schedule())
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


def temporal_direction_attribution(
    dates: np.ndarray,
    forward_returns: np.ndarray,
    neutral_returns: np.ndarray,
) -> dict:
    if len(dates) != len(forward_returns) or len(dates) != len(neutral_returns):
        raise ValueError("temporal direction inputs do not align")
    years = np.asarray([str(value)[:4] for value in dates])
    annual = {}
    leave_one_year_out = {}
    for year in sorted(set(years)):
        in_year = years == year
        forward_year = float(np.prod(1.0 + forward_returns[in_year]) - 1.0)
        neutral_year = float(np.prod(1.0 + neutral_returns[in_year]) - 1.0)
        annual[year] = {
            "forward_return": forward_year,
            "neutral_return": neutral_year,
            "forward_minus_neutral": forward_year - neutral_year,
        }
        keep = ~in_year
        forward_keep = float(np.prod(1.0 + forward_returns[keep]) - 1.0)
        neutral_keep = float(np.prod(1.0 + neutral_returns[keep]) - 1.0)
        leave_one_year_out[year] = {
            "forward_cumulative": forward_keep,
            "neutral_cumulative": neutral_keep,
            "forward_minus_neutral": forward_keep - neutral_keep,
        }
    return {
        "annual": annual,
        "positive_annual_count": int(
            sum(item["forward_minus_neutral"] > 0.0 for item in annual.values())
        ),
        "leave_one_year_out": leave_one_year_out,
        "forward_beats_neutral_after_each_year_removed": all(
            item["forward_minus_neutral"] > 0.0
            for item in leave_one_year_out.values()
        ),
    }


def main() -> None:
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint.get("validation_2026_opened") is not False:
        raise PermissionError("2026 entered neutral rank sizing placebo")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = sizing.width_tools.harness.load_context(policy)
    results = {}
    deterministic = {}
    baseline_frames = None
    for cost_name, cost in (
        ("0_30pct", sizing.BASELINE_COST),
        ("0_65pct", sizing.STRESS_COST),
    ):
        equal_daily, equal_actions = sizing.run_case(
            context, policy, cost, None, None
        )
        forward_daily, forward_actions = sizing.run_case(
            context,
            policy,
            cost,
            sizing.TOP_WEIGHT_MULTIPLIER,
            sizing.BOTTOM_WEIGHT_MULTIPLIER,
        )
        neutral_daily, neutral_actions = run_neutral(context, policy, cost)
        equal = sizing.evaluate(equal_daily, equal_actions)
        forward = sizing.evaluate(forward_daily, forward_actions)
        neutral = sizing.evaluate(neutral_daily, neutral_actions)
        results[cost_name] = {
            "equalweight": equal,
            "forward_high_score_overweight": forward,
            "rank_neutral_dispersion_placebo": neutral,
            "neutral_minus_equalweight": sizing.checkpoint_tools.metric_delta(
                neutral, equal
            ),
            "forward_minus_neutral": sizing.checkpoint_tools.metric_delta(
                forward, neutral
            ),
        }
        replay_daily, replay_actions = run_neutral(context, policy, cost)
        deterministic[cost_name] = {
            "daily": replay_daily.equals(neutral_daily),
            "actions": replay_actions.equals(neutral_actions),
        }
        if cost_name == "0_30pct":
            baseline_frames = (forward_daily, neutral_daily)
    if not all(all(value.values()) for value in deterministic.values()):
        raise RuntimeError("neutral placebo replay drifted")
    schedule = neutral_schedule()
    if baseline_frames is None:
        raise RuntimeError("baseline frames were not retained")
    forward_returns = baseline_frames[0]["return"].to_numpy(dtype=np.float64)
    neutral_returns = baseline_frames[1]["return"].to_numpy(dtype=np.float64)
    direction_bootstrap = {
        str(block): bootstrap.paired_bootstrap(
            forward_returns,
            neutral_returns,
            block,
            BOOTSTRAP_SEED + block,
        )
        for block in BLOCK_LENGTHS
    }
    temporal = temporal_direction_attribution(
        baseline_frames[0]["date"].astype(str).to_numpy(),
        forward_returns,
        neutral_returns,
    )
    result = {
        "status": "rank_sizing_neutral_placebo_complete_2026_not_opened",
        "role": "rank_neutral_dispersion_diagnostic_not_selectable",
        "design": {
            "schedule": schedule.tolist(),
            "schedule_sum": float(schedule.sum()),
            "rank_correlation": float(
                np.corrcoef(np.arange(10), schedule)[0, 1]
            ),
            "permutation": list(NEUTRAL_PERMUTATION),
            "chosen_without_return_results": True,
            "parameter_search": False,
        },
        "results": results,
        "forward_vs_neutral_paired_block_bootstrap": direction_bootstrap,
        "forward_vs_neutral_temporal_attribution": temporal,
        "forward_beats_neutral_at_both_costs": all(
            item["forward_minus_neutral"]["cumulative_return"] > 0.0
            for item in results.values()
        ),
        "neutral_beats_equalweight_at_both_costs": all(
            item["neutral_minus_equalweight"]["cumulative_return"] > 0.0
            for item in results.values()
        ),
        "deterministic_replay": deterministic,
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    round1.atomic_json(OUTPUT_ROOT / "neutral_placebo.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "forward_beats_neutral_at_both_costs": result[
                    "forward_beats_neutral_at_both_costs"
                ],
                "neutral_beats_equalweight_at_both_costs": result[
                    "neutral_beats_equalweight_at_both_costs"
                ],
                "results": {
                    cost: {
                        "neutral_minus_equalweight": item[
                            "neutral_minus_equalweight"
                        ]["cumulative_return"],
                        "forward_minus_neutral": item[
                            "forward_minus_neutral"
                        ]["cumulative_return"],
                    }
                    for cost, item in results.items()
                },
                "validation_2026_opened": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
