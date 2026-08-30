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
import research_v260_fixed10_current_replacement_margin_recheck_20260822 as current
import research_v260_fixed10_entry_beta_tail_guard_20260822 as percentile
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_hold_robustness_20260822 as robustness
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness
import research_v260_fixed10_regime_exit_threshold_20260822 as regime
import research_v260_fixed10_weak_market_defensive_rerank_20260822 as defensive
import research_v260_fixed10_weak_market_volatility_sell_priority_20260822 as priority
import research_v260_lowrisk_score_tuning_20260821 as research_base


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_current_exit_confirmation_recheck_20260822"
)
CHECKPOINT = current.CHECKPOINT
COST_LEVELS = current.COST_LEVELS
BASELINE_ID = "score_exit_confirmation_1d"
CANDIDATE_ID = "score_exit_confirmation_2d"


def build_policies(base_policy: dict) -> dict[str, dict]:
    policies = {}
    for days in (1, 2):
        policy = copy.deepcopy(base_policy)
        policy["sell_confirmation_days"] = days
        policies[f"score_exit_confirmation_{days}d"] = policy
    return policies


def selection_gates(candidate: dict, baseline: dict) -> dict[str, bool]:
    current_metrics = candidate["cost_metrics"]["0.0030"]
    reference = baseline["cost_metrics"]["0.0030"]
    return {
        "training_cagr_improved": candidate["train_2022_2024"]["cagr"]
        > baseline["train_2022_2024"]["cagr"],
        "training_sharpe_improved": candidate["train_2022_2024"]["sharpe"]
        > baseline["train_2022_2024"]["sharpe"],
        "full_period_cumulative_not_worse": current_metrics["cumulative_return"]
        >= reference["cumulative_return"],
        "full_period_sharpe_not_worse": current_metrics["sharpe"]
        >= reference["sharpe"],
        "drawdown_not_worse": current_metrics["max_drawdown"]
        <= reference["max_drawdown"],
        "forward_2025_not_worse": candidate["holdout_2025"]["cumulative_return"]
        >= baseline["holdout_2025"]["cumulative_return"],
        "stress_not_worse": candidate["cost_metrics"]["0.0065"][
            "cumulative_return"
        ]
        >= baseline["cost_metrics"]["0.0065"]["cumulative_return"],
        "turnover_not_higher": current_metrics["turnover_annualized"]
        <= reference["turnover_annualized"],
        "all_offsets_profitable": min(
            item["cumulative_return"]
            for item in candidate["start_offset_metrics"].values()
        )
        > 0.0,
        "all_years_positive": min(current_metrics["annual_returns"].values())
        > 0.0,
        "exactly10": current_metrics["full_10_position_ratio"] == 1.0,
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered exit-confirmation recheck")
    base_policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(base_policy)
    strong = regime.strong_market_mask(context.score, context.protocol)
    confirmed_weak = confirmed.confirmed_weak_mask(strong, 2)
    volatility = defensive.trailing_log_volatility(
        context.arrays["close_qfq"], defensive.VOLATILITY_WINDOW
    )
    volatility_rank = percentile.cross_sectional_percent_rank(volatility)
    priority_matrix = priority.sell_priority_matrix(
        context.score, volatility_rank, ~confirmed_weak, 0.05
    )
    policies = build_policies(base_policy)

    results, cache = {}, {}
    for candidate_id, policy in policies.items():
        cost_metrics = {}
        for cost in COST_LEVELS:
            daily, actions = current.run(context, policy, priority_matrix, cost)
            cost_metrics[f"{cost:.4f}"] = round1.evaluate_run(
                daily, actions, research_base.FIRST_BUY, round1.DEVELOPMENT_END
            )
            if cost == 0.0030:
                cache[candidate_id] = (daily, actions)
        daily, actions = cache[candidate_id]
        results[candidate_id] = {
            "cost_metrics": cost_metrics,
            "train_2022_2024": round1.evaluate_run(
                daily, actions, research_base.FIRST_BUY, "20241231"
            ),
            "holdout_2025": round1.evaluate_run(
                daily, actions, "20250102", round1.DEVELOPMENT_END
            ),
            "start_offset_metrics": {
                str(offset): round1.evaluate_run(
                    daily,
                    actions,
                    research_base.FIRST_BUY
                    if offset == 0
                    else robustness.window_start_date(context.arrays, offset),
                    round1.DEVELOPMENT_END,
                )
                for offset in (0, 5, 20, 60)
            },
        }

    baseline = results[BASELINE_ID]
    candidate = results[CANDIDATE_ID]
    expected = checkpoint["current_best_equalweight"]
    equivalence = {
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
    if not all(equivalence.values()):
        raise RuntimeError("exit-confirmation baseline drifted")
    gates = selection_gates(candidate, baseline)
    selected = CANDIDATE_ID if all(gates.values()) else BASELINE_ID
    repeat_daily, repeat_actions = current.run(
        context, policies[selected], priority_matrix, 0.0030
    )
    deterministic = {
        "daily": round1.frame_hash(cache[selected][0])
        == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(cache[selected][1])
        == round1.frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("exit-confirmation replay failed")

    baseline_metrics = baseline["cost_metrics"]["0.0030"]
    candidate_metrics = candidate["cost_metrics"]["0.0030"]
    result = {
        "status": "current_exit_confirmation_recheck_complete_2026_not_opened",
        "only_change": (
            "require two consecutive signal sessions below the unchanged 0.85 score "
            "threshold instead of one before a score exit"
        ),
        "candidate_budget": [CANDIDATE_ID],
        "results": results,
        "candidate_deltas": {
            key: candidate_metrics[key] - baseline_metrics[key]
            for key in (
                "cumulative_return",
                "cagr",
                "sharpe",
                "max_drawdown",
                "turnover_annualized",
            )
        },
        "candidate_gates": gates,
        "selected_candidate": selected,
        "changed_from_checkpoint": selected != BASELINE_ID,
        "checkpoint_equivalence": equivalence,
        "deterministic_replay": deterministic,
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "development_result.json", result)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
