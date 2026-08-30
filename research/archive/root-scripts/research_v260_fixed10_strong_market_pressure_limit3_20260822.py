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
import research_v260_fixed10_pressure_regime_trigger_20260822 as trigger
import research_v260_fixed10_regime_exit_threshold_20260822 as regime
import research_v260_fixed10_weak_market_defensive_rerank_20260822 as defensive
import research_v260_fixed10_weak_market_volatility_sell_priority_20260822 as priority
import research_v260_lowrisk_score_tuning_20260821 as research_base


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_strong_market_pressure_limit3_20260822"
)
CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
COSTS = (0.0030, 0.0065)


def priority_matrix(context, strong: np.ndarray) -> np.ndarray:
    weak2 = confirmed.confirmed_weak_mask(strong, 2)
    volatility = defensive.trailing_log_volatility(
        context.arrays["close_qfq"], defensive.VOLATILITY_WINDOW
    )
    volatility_rank = percentile.cross_sectional_percent_rank(volatility)
    return priority.sell_priority_matrix(
        context.score, volatility_rank, ~weak2, 0.05
    )


def run(context, policy: dict, pressure_limit: np.ndarray, cost: float):
    strong = regime.strong_market_mask(context.score, context.protocol)
    return age_guard.run_policy_at_cost(
        context,
        policy,
        age_boundary.pressure_age_schedule(strong, 10),
        cost,
        pressure_trigger_override=trigger.pressure_trigger_schedule(strong),
        pressure_limit_override=pressure_limit,
        score_sell_priority_override=priority_matrix(context, strong),
    )


def evaluate(daily, actions, start: str, end: str) -> dict:
    return round1.evaluate_run(daily, actions, start, end)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered strong-market pressure test")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)
    strong = regime.strong_market_mask(context.score, context.protocol)
    schedules = {
        "current_limit2_all_markets": np.full(strong.shape, 2, dtype=np.int64),
        "strong_limit3_weak_limit2": np.where(strong, 3, 2).astype(np.int64),
    }
    results = {}
    cache = {}
    for candidate_id, schedule in schedules.items():
        cost_metrics = {}
        for cost in COSTS:
            daily, actions = run(context, policy, schedule, cost)
            cost_metrics[f"{cost:.4f}"] = evaluate(
                daily, actions, research_base.FIRST_BUY, round1.DEVELOPMENT_END
            )
            if cost == 0.0030:
                cache[candidate_id] = (daily, actions)
        daily, actions = cache[candidate_id]
        results[candidate_id] = {
            "cost_metrics": cost_metrics,
            "train_2022_2024": evaluate(
                daily, actions, research_base.FIRST_BUY, "20241231"
            ),
            "holdout_2025": evaluate(
                daily, actions, "20250102", round1.DEVELOPMENT_END
            ),
        }

    baseline_id = "current_limit2_all_markets"
    candidate_id = "strong_limit3_weak_limit2"
    baseline = results[baseline_id]
    candidate = results[candidate_id]
    expected = checkpoint["current_best_equalweight"]
    baseline_equivalence = all(
        np.isclose(
            baseline["cost_metrics"]["0.0030"][key], expected[key], rtol=0.0, atol=1e-12
        )
        for key in (
            "cumulative_return", "cagr", "sharpe", "max_drawdown",
            "turnover_annualized", "average_invested_ratio",
        )
    )
    if not baseline_equivalence:
        raise RuntimeError("strong-market pressure baseline drifted")

    train = candidate["train_2022_2024"]
    train_base = baseline["train_2022_2024"]
    holdout = candidate["holdout_2025"]
    holdout_base = baseline["holdout_2025"]
    full = candidate["cost_metrics"]["0.0030"]
    full_base = baseline["cost_metrics"]["0.0030"]
    gates = {
        "training_cagr_improved": train["cagr"] > train_base["cagr"],
        "training_sharpe_not_worse": train["sharpe"] >= train_base["sharpe"],
        "holdout_2025_cagr_not_worse": holdout["cagr"] >= holdout_base["cagr"],
        "holdout_2025_sharpe_not_worse": holdout["sharpe"] >= holdout_base["sharpe"],
        "full_cumulative_improved": full["cumulative_return"] > full_base["cumulative_return"],
        "full_drawdown_not_worse": full["max_drawdown"] <= full_base["max_drawdown"],
        "stress_not_worse": (
            candidate["cost_metrics"]["0.0065"]["cumulative_return"]
            >= baseline["cost_metrics"]["0.0065"]["cumulative_return"]
        ),
        "all_years_positive": min(full["annual_returns"].values()) > 0.0,
        "exactly10": full["full_10_position_ratio"] == 1.0,
    }
    accepted = all(gates.values())
    selected = candidate_id if accepted else baseline_id
    repeat_daily, repeat_actions = run(
        context, policy, schedules[selected], 0.0030
    )
    deterministic = {
        "daily": round1.frame_hash(repeat_daily) == round1.frame_hash(cache[selected][0]),
        "actions": round1.frame_hash(repeat_actions) == round1.frame_hash(cache[selected][1]),
    }
    if not all(deterministic.values()):
        raise RuntimeError("strong-market pressure replay failed")

    result = {
        "status": (
            "candidate_accepted_pre2026_validation_not_opened"
            if accepted
            else "candidate_rejected_pre2026_validation_not_opened"
        ),
        "rule": (
            "when the frozen strong-market state has at least four eligible score exits, "
            "allow at most three score exits; weak-market limit remains two"
        ),
        "results": results,
        "acceptance_gates": gates,
        "selected_candidate": selected,
        "baseline_equivalence": baseline_equivalence,
        "deterministic_replay": deterministic,
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "development_result.json", result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
