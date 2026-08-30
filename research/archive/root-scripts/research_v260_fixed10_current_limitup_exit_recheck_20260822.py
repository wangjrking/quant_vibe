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
    "strategy_agent_v260_fixed10_current_limitup_exit_recheck_20260822"
)
CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered limit-up exit recheck")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)
    strong = regime.strong_market_mask(context.score, context.protocol)
    confirmed_weak = confirmed.confirmed_weak_mask(strong, 2)
    extra_age = age_boundary.pressure_age_schedule(strong, 10)
    volatility_rank = percentile.cross_sectional_percent_rank(
        defensive.trailing_log_volatility(
            context.arrays["close_qfq"], defensive.VOLATILITY_WINDOW
        )
    )
    sell_priority = priority.sell_priority_matrix(
        context.score, volatility_rank, ~confirmed_weak, 0.05
    )

    results = {}
    cache = {}
    for case_id, defer in (("sell_as_planned", False), ("defer_open_limit_up_sell", True)):
        result, daily, actions = age_guard.run_policy(
            context,
            policy,
            extra_age,
            score_sell_priority_override=sell_priority,
            defer_sell_on_limit_up_override=defer,
        )
        results[case_id] = result
        cache[case_id] = (daily, actions)

    baseline = results["sell_as_planned"]
    candidate = results["defer_open_limit_up_sell"]
    expected = checkpoint["current_best_equalweight"]
    equivalence = {
        key: bool(np.isclose(baseline["metrics_0_30pct"][key], expected[key], rtol=0.0, atol=1e-12))
        for key in (
            "cumulative_return", "cagr", "sharpe", "max_drawdown",
            "turnover_annualized", "average_invested_ratio",
        )
    }
    if not all(equivalence.values()):
        raise RuntimeError("limit-up exit baseline drifted")
    gates = {
        "train_cagr_improved": candidate["train_2022_2024"]["cagr"] > baseline["train_2022_2024"]["cagr"],
        "train_sharpe_improved": candidate["train_2022_2024"]["sharpe"] > baseline["train_2022_2024"]["sharpe"],
        "holdout_not_worse": candidate["holdout_2025"]["cumulative_return"] >= baseline["holdout_2025"]["cumulative_return"],
        "stress_not_worse": candidate["metrics_0_65pct"]["cumulative_return"] >= baseline["metrics_0_65pct"]["cumulative_return"],
        "drawdown_not_worse": candidate["metrics_0_30pct"]["max_drawdown"] <= baseline["metrics_0_30pct"]["max_drawdown"],
        "full_10_positions": candidate["metrics_0_30pct"]["full_10_position_ratio"] == 1.0,
    }
    selected = "defer_open_limit_up_sell" if all(gates.values()) else "sell_as_planned"
    repeat, repeat_daily, repeat_actions = age_guard.run_policy(
        context,
        policy,
        extra_age,
        score_sell_priority_override=sell_priority,
        defer_sell_on_limit_up_override=selected == "defer_open_limit_up_sell",
    )
    deterministic = {
        "daily": round1.frame_hash(cache[selected][0]) == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(cache[selected][1]) == round1.frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("limit-up exit replay failed")
    payload = {
        "status": "current_limitup_exit_recheck_complete_2026_not_opened",
        "market_logic": (
            "a limit-up open is sellable and has available buyers; deferral is an "
            "optional strategy choice, not an execution necessity"
        ),
        "results": results,
        "candidate_deltas": {
            section: {
                key: float(candidate[section][key] - baseline[section][key])
                for key in ("cumulative_return", "cagr", "sharpe", "max_drawdown", "turnover_annualized")
            }
            for section in ("metrics_0_30pct", "train_2022_2024", "holdout_2025", "metrics_0_65pct")
        },
        "selection_gates": gates,
        "selected_candidate": selected,
        "changed_from_checkpoint": selected != "sell_as_planned",
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
        "selection_gates": gates,
        "candidate_deltas": payload["candidate_deltas"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
