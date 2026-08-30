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
import research_v260_fixed10_weak_market_volatility_sell_priority_20260822 as priority


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_current_open_gap_guard_recheck_20260822"
)
CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)


def utility(metrics: dict) -> float:
    return float(
        metrics["cagr"]
        + 0.25 * metrics["sharpe"]
        - 0.50 * metrics["max_drawdown"]
        - 0.0025 * metrics["turnover_annualized"]
    )


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered open-gap recheck")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)
    strong = regime.strong_market_mask(context.score, context.protocol)
    confirmed_weak = confirmed.confirmed_weak_mask(strong, 2)
    extra_age = age_boundary.pressure_age_schedule(strong, 10)
    volatility = defensive.trailing_log_volatility(
        context.arrays["close_qfq"], defensive.VOLATILITY_WINDOW
    )
    volatility_rank = percentile.cross_sectional_percent_rank(volatility)
    sell_priority = priority.sell_priority_matrix(
        context.score, volatility_rank, ~confirmed_weak, 0.05
    )

    cases = {"no_gap_guard": (None, None)}
    cases.update({f"max_gap_{value:02d}pct": (value / 100.0, None) for value in (3, 5, 7)})
    cases.update({f"min_gap_m{value:02d}pct": (None, -value / 100.0) for value in (3, 5, 7)})
    results = {}
    cache = {}
    for case_id, (maximum, minimum) in cases.items():
        result, daily, actions = age_guard.run_policy(
            context,
            policy,
            extra_age,
            score_sell_priority_override=sell_priority,
            max_entry_open_gap_override=maximum,
            min_entry_open_gap_override=minimum,
        )
        results[case_id] = {
            "maximum_entry_open_gap": maximum,
            "minimum_entry_open_gap": minimum,
            "metrics_0_30pct": result["metrics_0_30pct"],
            "train_2022_2024": result["train_2022_2024"],
            "holdout_2025": result["holdout_2025"],
            "metrics_0_65pct": result["metrics_0_65pct"],
            "train_utility": utility(result["train_2022_2024"]),
        }
        cache[case_id] = (daily, actions)

    baseline = results["no_gap_guard"]
    expected = checkpoint["current_best_equalweight"]
    equivalence = {
        key: bool(np.isclose(baseline["metrics_0_30pct"][key], expected[key], rtol=0.0, atol=1e-12))
        for key in (
            "cumulative_return", "cagr", "sharpe", "max_drawdown",
            "turnover_annualized", "average_invested_ratio",
        )
    }
    if not all(equivalence.values()):
        raise RuntimeError("open-gap baseline drifted")

    deltas = {}
    for case_id, item in results.items():
        if case_id == "no_gap_guard":
            continue
        deltas[case_id] = {
            "train_utility": float(item["train_utility"] - baseline["train_utility"]),
            "holdout_2025_return": float(item["holdout_2025"]["cumulative_return"] - baseline["holdout_2025"]["cumulative_return"]),
            "stress_return": float(item["metrics_0_65pct"]["cumulative_return"] - baseline["metrics_0_65pct"]["cumulative_return"]),
            "full_cagr": float(item["metrics_0_30pct"]["cagr"] - baseline["metrics_0_30pct"]["cagr"]),
            "full_sharpe": float(item["metrics_0_30pct"]["sharpe"] - baseline["metrics_0_30pct"]["sharpe"]),
            "full_drawdown": float(item["metrics_0_30pct"]["max_drawdown"] - baseline["metrics_0_30pct"]["max_drawdown"]),
        }

    family_gates = {}
    for family, candidate, controls in (
        ("maximum_gap", "max_gap_05pct", ("max_gap_03pct", "max_gap_07pct")),
        ("minimum_gap", "min_gap_m05pct", ("min_gap_m03pct", "min_gap_m07pct")),
    ):
        candidate_item = results[candidate]
        family_gates[family] = {
            "candidate_train_utility_improved": deltas[candidate]["train_utility"] > 0.0,
            "candidate_holdout_not_worse": deltas[candidate]["holdout_2025_return"] >= 0.0,
            "candidate_stress_not_worse": deltas[candidate]["stress_return"] >= 0.0,
            "candidate_drawdown_not_worse": deltas[candidate]["full_drawdown"] <= 0.0,
            "at_least_one_neighbor_confirms": any(
                deltas[control]["train_utility"] > 0.0
                and deltas[control]["holdout_2025_return"] >= 0.0
                for control in controls
            ),
            "all_years_positive": all(
                value > 0.0
                for value in candidate_item["metrics_0_30pct"]["annual_returns"].values()
            ),
            "full_10_positions": candidate_item["metrics_0_30pct"]["full_10_position_ratio"] == 1.0,
        }
    passing = [family for family, gates in family_gates.items() if all(gates.values())]
    selected = (
        "max_gap_05pct" if passing == ["maximum_gap"]
        else "min_gap_m05pct" if passing == ["minimum_gap"]
        else "no_gap_guard"
    )

    maximum = results[selected]["maximum_entry_open_gap"]
    minimum = results[selected]["minimum_entry_open_gap"]
    repeat, repeat_daily, repeat_actions = age_guard.run_policy(
        context,
        policy,
        extra_age,
        score_sell_priority_override=sell_priority,
        max_entry_open_gap_override=maximum,
        min_entry_open_gap_override=minimum,
    )
    deterministic = {
        "daily": round1.frame_hash(cache[selected][0]) == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(cache[selected][1]) == round1.frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("open-gap recheck replay failed")

    payload = {
        "status": "current_open_gap_guard_recheck_complete_2026_not_opened",
        "rule_scope": "new entries only; ranking refills to ten positions",
        "results": results,
        "deltas_vs_no_guard": deltas,
        "family_gates": family_gates,
        "selected_candidate": selected,
        "changed_from_checkpoint": selected != "no_gap_guard",
        "baseline_equivalence": equivalence,
        "deterministic_replay": deterministic,
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "development_result.json", payload)
    print(json.dumps({
        "status": payload["status"],
        "selected_candidate": selected,
        "family_gates": family_gates,
        "deltas_vs_no_guard": deltas,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
