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
import research_v260_fixed10_weak_market_defensive_rerank_20260822 as defensive
import research_v260_fixed10_weak_market_size_tilt_20260822 as size_tilt
import research_v260_fixed10_weak_market_volatility_sell_priority_20260822 as priority


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_global_microcap_tail_guard_20260822"
)
CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
CASES = {
    "guard_off": 0.0,
    "control_bottom_025pct": 0.025,
    "candidate_bottom_05pct": 0.05,
    "control_bottom_10pct": 0.10,
}


def gates(candidate: dict, baseline: dict, controls: list[dict]) -> dict[str, bool]:
    return {
        "training_sharpe_improved": candidate["train_2022_2024"]["sharpe"]
        > baseline["train_2022_2024"]["sharpe"],
        "training_cagr_improved": candidate["train_2022_2024"]["cagr"]
        > baseline["train_2022_2024"]["cagr"],
        "overall_drawdown_lower": candidate["metrics_0_30pct"]["max_drawdown"]
        < baseline["metrics_0_30pct"]["max_drawdown"],
        "forward_2025_not_worse": candidate["holdout_2025"]["cumulative_return"]
        >= baseline["holdout_2025"]["cumulative_return"],
        "stress_not_worse": candidate["metrics_0_65pct"]["cumulative_return"]
        >= baseline["metrics_0_65pct"]["cumulative_return"],
        "all_calendar_years_positive": min(
            candidate["metrics_0_30pct"]["annual_returns"].values()
        )
        > 0.0,
        "full_10_positions": candidate["metrics_0_30pct"]["full_10_position_ratio"]
        == 1.0,
        "both_neighbor_controls_improve_training_sharpe": all(
            item["train_2022_2024"]["sharpe"]
            > baseline["train_2022_2024"]["sharpe"]
            for item in controls
        ),
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered global microcap-tail guard")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)
    strong = regime.strong_market_mask(context.score, context.protocol)
    confirmed_weak = confirmed.confirmed_weak_mask(strong, 2)
    extra_age = age_boundary.pressure_age_schedule(strong, 10)
    volatility = defensive.trailing_log_volatility(
        context.arrays["close_qfq"], defensive.VOLATILITY_WINDOW
    )
    volatility_rank = percentile.cross_sectional_percent_rank(volatility)
    priority_matrix = priority.sell_priority_matrix(
        context.score, volatility_rank, ~confirmed_weak, 0.05
    )
    reference_universe = np.asarray(
        context.arrays["signal_clean"], dtype=np.bool_
    ) & np.isfinite(context.arrays["total_mv"])
    size_percentile = size_tilt.size_percentile_matrix(
        context.arrays["total_mv"], reference_universe
    )
    results = {}
    cache = {}
    for case_id, minimum in CASES.items():
        entry_block = ~np.isfinite(size_percentile) | (size_percentile < minimum)
        result, daily, actions = age_guard.run_policy(
            context,
            policy,
            extra_age,
            entry_block_mask_override=entry_block,
            score_sell_priority_override=priority_matrix,
        )
        result["minimum_entry_size_percentile"] = float(minimum)
        result["blocked_stock_dates"] = int(entry_block.sum())
        results[case_id] = result
        cache[case_id] = (daily, actions)

    baseline = results["guard_off"]
    expected = checkpoint["current_best_equalweight"]
    equivalence = {
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
    if not all(equivalence.values()):
        raise RuntimeError("global microcap-tail baseline drifted")
    candidate_id = "candidate_bottom_05pct"
    acceptance = gates(
        results[candidate_id],
        baseline,
        [results["control_bottom_025pct"], results["control_bottom_10pct"]],
    )
    selected = candidate_id if all(acceptance.values()) else "guard_off"
    repeat, repeat_daily, repeat_actions = age_guard.run_policy(
        context,
        policy,
        extra_age,
        entry_block_mask_override=(
            ~np.isfinite(size_percentile)
            | (size_percentile < CASES[selected])
        ),
        score_sell_priority_override=priority_matrix,
    )
    deterministic = {
        "daily": round1.frame_hash(cache[selected][0])
        == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(cache[selected][1])
        == round1.frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("global microcap-tail replay failed")
    selected_policy = copy.deepcopy(policy)
    selected_policy["minimum_entry_size_percentile"] = CASES[selected]
    result = {
        "status": (
            "candidate_accepted_pre2026_validation_not_opened"
            if selected == candidate_id
            else "candidate_rejected_pre2026_validation_not_opened"
        ),
        "only_change": (
            "exclude only the same-day cross-sectional extreme microcap tail from "
            "new entries; existing holdings and all exit rules remain unchanged"
        ),
        "candidate_budget": CASES,
        "results": results,
        "acceptance_gates": acceptance,
        "selected_candidate": selected,
        "selected_policy": selected_policy,
        "baseline_equivalence": equivalence,
        "deterministic_replay": deterministic,
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "development_result.json", result)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
