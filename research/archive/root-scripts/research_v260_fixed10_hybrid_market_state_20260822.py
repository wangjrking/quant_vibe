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
import research_v260_fixed10_hold_robustness_20260822 as robustness
import research_v260_fixed10_market_state_semantic_diagnostic_20260822 as state_diag
import research_v260_fixed10_pressure_extra_age_guard_20260822 as age_guard
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness
import research_v260_fixed10_pressure_regime_trigger_20260822 as trigger
import research_v260_fixed10_regime_exit_threshold_20260822 as regime
import research_v260_fixed10_weak_market_defensive_rerank_20260822 as defensive
import research_v260_fixed10_weak_market_volatility_sell_priority_20260822 as priority
import research_v260_lowrisk_score_tuning_20260821 as research_base


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/strategy_agent_v260_fixed10_hybrid_market_state_20260822"
)
CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
PRICE_TREND_WINDOW = 20
SEVERE_DECLINE_THRESHOLD = -0.05
CONFIRMATION_DAYS = 2
VOLATILITY_PENALTY = 0.05
COST_LEVELS = (0.0030, 0.0040, 0.0065)


def build_state(context, use_price_confirmation: bool) -> dict:
    production_strong = regime.strong_market_mask(context.score, context.protocol)
    market_daily = state_diag.cross_sectional_median_return(
        context.arrays["close_qfq"]
    )
    trend = state_diag.trailing_compound_return(market_daily, PRICE_TREND_WINDOW)
    severe_decline = np.isfinite(trend) & (trend < SEVERE_DECLINE_THRESHOLD)
    strong = (
        production_strong & ~severe_decline
        if use_price_confirmation
        else production_strong
    )
    confirmed_weak = confirmed.confirmed_weak_mask(strong, CONFIRMATION_DAYS)
    volatility = defensive.trailing_log_volatility(
        context.arrays["close_qfq"], defensive.VOLATILITY_WINDOW
    )
    volatility_rank = percentile.cross_sectional_percent_rank(volatility)
    return {
        "strong": strong,
        "production_strong": production_strong,
        "trend": trend,
        "severe_decline": severe_decline,
        "confirmed_weak": confirmed_weak,
        "extra_age": age_boundary.pressure_age_schedule(strong, 10),
        "pressure_trigger": trigger.pressure_trigger_schedule(strong),
        "sell_priority": priority.sell_priority_matrix(
            context.score,
            volatility_rank,
            ~confirmed_weak,
            VOLATILITY_PENALTY,
        ),
    }


def run_candidate(context, policy: dict, state: dict, cost: float):
    return age_guard.run_policy_at_cost(
        context,
        policy,
        state["extra_age"],
        cost,
        pressure_trigger_override=state["pressure_trigger"],
        score_sell_priority_override=state["sell_priority"],
    )


def evaluate(daily, actions, start: str, end: str) -> dict:
    return round1.evaluate_run(daily, actions, start, end)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered hybrid market-state development")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)
    states = {
        "production_breadth_state": build_state(context, False),
        "breadth_plus_20d_minus5pct_guard": build_state(context, True),
    }
    results, cache = {}, {}
    for candidate_id, state in states.items():
        cost_metrics = {}
        for cost in COST_LEVELS:
            daily, actions = run_candidate(context, policy, state, cost)
            cost_metrics[f"{cost:.4f}"] = evaluate(
                daily, actions, research_base.FIRST_BUY, round1.DEVELOPMENT_END
            )
            if cost == 0.0030:
                cache[candidate_id] = (daily, actions)
        baseline_daily, baseline_actions = cache[candidate_id]
        results[candidate_id] = {
            "cost_metrics": cost_metrics,
            "train_2022_2024": evaluate(
                baseline_daily, baseline_actions, research_base.FIRST_BUY, "20241231"
            ),
            "holdout_2025": evaluate(
                baseline_daily, baseline_actions, "20250102", round1.DEVELOPMENT_END
            ),
            "start_offset_metrics": {
                str(offset): evaluate(
                    baseline_daily,
                    baseline_actions,
                    research_base.FIRST_BUY
                    if offset == 0
                    else robustness.window_start_date(context.arrays, offset),
                    round1.DEVELOPMENT_END,
                )
                for offset in (0, 5, 20, 60)
            },
            "state_counts": {
                "strong_days": int(np.sum(state["strong"])),
                "confirmed_weak_days": int(np.sum(state["confirmed_weak"])),
                "severe_decline_days": int(np.sum(state["severe_decline"])),
                "production_strong_reclassified": int(
                    np.sum(state["production_strong"] & ~state["strong"])
                ),
            },
        }

    baseline_id = "production_breadth_state"
    candidate_id = "breadth_plus_20d_minus5pct_guard"
    baseline = results[baseline_id]
    candidate = results[candidate_id]
    expected = checkpoint["current_best_equalweight"]
    checkpoint_equivalence = {
        key: bool(
            np.isclose(
                baseline["cost_metrics"]["0.0030"][key],
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
        raise RuntimeError("hybrid market-state baseline drifted from checkpoint")

    current = candidate["cost_metrics"]["0.0030"]
    reference = baseline["cost_metrics"]["0.0030"]
    gates = {
        "training_sharpe_improved": candidate["train_2022_2024"]["sharpe"]
        > baseline["train_2022_2024"]["sharpe"],
        "training_drawdown_not_worse": candidate["train_2022_2024"]["max_drawdown"]
        <= baseline["train_2022_2024"]["max_drawdown"],
        "full_period_cumulative_not_worse": current["cumulative_return"]
        >= reference["cumulative_return"],
        "full_period_sharpe_not_worse": current["sharpe"] >= reference["sharpe"],
        "full_period_drawdown_not_worse": current["max_drawdown"]
        <= reference["max_drawdown"],
        "forward_2025_not_worse": candidate["holdout_2025"]["cumulative_return"]
        >= baseline["holdout_2025"]["cumulative_return"],
        "all_cost_levels_not_worse": all(
            candidate["cost_metrics"][key]["cumulative_return"]
            >= baseline["cost_metrics"][key]["cumulative_return"]
            for key in candidate["cost_metrics"]
        ),
        "all_start_offsets_profitable": min(
            item["cumulative_return"]
            for item in candidate["start_offset_metrics"].values()
        )
        > 0.0,
        "all_years_positive": min(current["annual_returns"].values()) > 0.0,
        "exactly10_full_period": current["full_10_position_ratio"] == 1.0,
    }
    selected = candidate_id if all(gates.values()) else baseline_id
    repeat_daily, repeat_actions = run_candidate(
        context, policy, states[selected], 0.0030
    )
    deterministic = {
        "daily": round1.frame_hash(cache[selected][0])
        == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(cache[selected][1])
        == round1.frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("hybrid market-state replay failed")

    result = {
        "status": "hybrid_market_state_test_complete_2026_not_opened",
        "source_strategy": context.rules["strategy_id"],
        "only_change": (
            "when trailing 20-session compounded cross-sectional median return is "
            "below -5%, production breadth cannot classify the day as strong"
        ),
        "portfolio_and_score_changed": False,
        "results": results,
        "candidate_gates": gates,
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
