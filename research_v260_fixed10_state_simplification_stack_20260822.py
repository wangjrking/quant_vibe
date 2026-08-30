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

import research_v260_fixed10_confirmed_severe_weak_defensive_sleeve_20260822 as confirmed
import research_v260_fixed10_current_weak_pressure_age_boundary_20260822 as age_boundary
import research_v260_fixed10_entry_beta_tail_guard_20260822 as percentile
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_pressure_extra_age_guard_20260822 as age_guard
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness
import research_v260_fixed10_regime_exit_threshold_20260822 as regime
import research_v260_fixed10_vol_priority_action_concentration_20260822 as concentration
import research_v260_fixed10_weak_market_defensive_rerank_20260822 as defensive
import research_v260_fixed10_weak_market_volatility_sell_priority_20260822 as priority


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_state_simplification_stack_20260822"
)
CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)


def metrics_delta(left: dict, right: dict) -> dict[str, float]:
    return {
        key: float(left[key] - right[key])
        for key in (
            "cumulative_return",
            "cagr",
            "sharpe",
            "max_drawdown",
            "turnover_annualized",
            "average_invested_ratio",
        )
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered state-simplification stack")

    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)
    strong = regime.strong_market_mask(context.score, context.protocol)
    weak2 = confirmed.confirmed_weak_mask(strong, 2)
    current_age = age_boundary.pressure_age_schedule(strong, 10)
    volatility = defensive.trailing_log_volatility(
        context.arrays["close_qfq"], defensive.VOLATILITY_WINDOW
    )
    volatility_rank = percentile.cross_sectional_percent_rank(volatility)
    current_priority = priority.sell_priority_matrix(
        context.score, volatility_rank, ~weak2, 0.05
    )
    all_day_priority = priority.sell_priority_matrix(
        context.score,
        volatility_rank,
        np.zeros(strong.shape, dtype=np.bool_),
        0.05,
    )
    uniform5 = np.full(strong.shape, 5, dtype=np.int16)
    maintenance_off = copy.deepcopy(policy)
    maintenance_off["maintenance_topup_requires_score"] = None

    cases = {
        "current": (policy, current_age, np.where(strong, 4, 5), current_priority),
        "uniform5_trigger": (policy, current_age, uniform5, current_priority),
        "uniform5_constant_age4": (policy, 4, uniform5, current_priority),
        "uniform5_constant_age4_all_day_vol_priority": (
            policy,
            4,
            uniform5,
            all_day_priority,
        ),
        "fully_simplified": (
            maintenance_off,
            4,
            uniform5,
            all_day_priority,
        ),
    }

    runs = {}
    cache = {}
    for case_id, (case_policy, extra_age, trigger, sell_priority) in cases.items():
        metrics, daily, actions = age_guard.run_policy(
            context,
            case_policy,
            extra_age,
            pressure_trigger_override=trigger,
            score_sell_priority_override=sell_priority,
        )
        runs[case_id] = metrics
        cache[case_id] = (daily, actions)

    expected = checkpoint["current_best_equalweight"]
    current_equivalent = bool(all(
        np.isclose(
            runs["current"]["metrics_0_30pct"][key],
            expected[key],
            rtol=0.0,
            atol=1e-12,
        )
        for key in ("cumulative_return", "cagr", "sharpe", "max_drawdown")
    ))
    if not current_equivalent:
        raise RuntimeError("state-simplification baseline drifted")

    sequence = list(cases)
    transitions = {}
    for previous, current in zip(sequence, sequence[1:]):
        summary = concentration.concentration_summary(
            concentration.daily_log_excess(
                cache[current][0], cache[previous][0]
            )
        )
        transitions[f"{previous}_to_{current}"] = {
            "metrics_delta_0_30pct": metrics_delta(
                runs[current]["metrics_0_30pct"],
                runs[previous]["metrics_0_30pct"],
            ),
            "stress_cumulative_return_delta": float(
                runs[current]["metrics_0_65pct"]["cumulative_return"]
                - runs[previous]["metrics_0_65pct"]["cumulative_return"]
            ),
            "action_difference": {
                key: value
                for key, value in concentration.action_difference(
                    cache[current][1], cache[previous][1]
                ).items()
                if key != "changed_execution_dates"
            },
            "daily_log_excess": {
                key: summary[key]
                for key in (
                    "total_log_excess",
                    "equivalent_relative_wealth_gain",
                    "log_excess_after_removing_top5_positive_days",
                    "positive_month_fraction",
                    "yearly_log_excess",
                )
            },
        }

    production = checkpoint["production_baseline"]
    simplified = runs["fully_simplified"]
    simplified_metrics = simplified["metrics_0_30pct"]
    simplified_vs_production = {
        "metrics_delta_0_30pct": metrics_delta(
            simplified_metrics, production
        ),
        "stress_cumulative_return": simplified["metrics_0_65pct"][
            "cumulative_return"
        ],
        "all_years_positive": min(simplified_metrics["annual_returns"].values()) > 0.0,
        "full_10_positions": simplified_metrics["full_10_position_ratio"] == 1.0,
    }

    repeat, repeat_daily, repeat_actions = age_guard.run_policy(
        context,
        maintenance_off,
        4,
        pressure_trigger_override=uniform5,
        score_sell_priority_override=all_day_priority,
    )
    deterministic = {
        "metrics": all(
            repeat["metrics_0_30pct"][key]
            == simplified["metrics_0_30pct"][key]
            for key in ("cumulative_return", "cagr", "sharpe", "max_drawdown")
        ),
        "daily": round1.frame_hash(repeat_daily)
        == round1.frame_hash(cache["fully_simplified"][0]),
        "actions": round1.frame_hash(repeat_actions)
        == round1.frame_hash(cache["fully_simplified"][1]),
    }
    if not all(deterministic.values()):
        raise RuntimeError("fully simplified candidate replay failed")

    result = {
        "status": "state_simplification_stack_complete_2026_not_opened",
        "diagnostic_only_no_automatic_policy_change": True,
        "ordered_simplifications": [
            "use uniform second-exit trigger=5",
            "use constant extra-pressure minimum age=4",
            "apply volatility sell ordering on every day",
            "remove maintenance top-up score gate",
        ],
        "runs": runs,
        "transitions": transitions,
        "fully_simplified_vs_production": simplified_vs_production,
        "current_equivalence": current_equivalent,
        "deterministic_replay": deterministic,
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "state_simplification_stack.json", result)
    print(json.dumps({
        "status": result["status"],
        "runs": {
            case_id: {
                key: values["metrics_0_30pct"][key]
                for key in ("cagr", "sharpe", "max_drawdown", "turnover_annualized")
            }
            for case_id, values in runs.items()
        },
        "fully_simplified_vs_production": simplified_vs_production,
        "deterministic_replay": deterministic,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
