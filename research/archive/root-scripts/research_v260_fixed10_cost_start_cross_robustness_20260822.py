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
import research_v260_fixed10_pressure_extra_age_guard_20260822 as age_guard
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness
import research_v260_fixed10_regime_exit_threshold_20260822 as regime
import research_v260_fixed10_weak_market_defensive_rerank_20260822 as defensive
import research_v260_fixed10_weak_market_volatility_sell_priority_20260822 as priority
import research_v260_lowrisk_score_tuning_20260821 as research_base


REPORTS = REPO / "quant/data_file/reports"
CHECKPOINT = (
    REPORTS
    / "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
OUTPUT_ROOT = (
    REPORTS / "strategy_agent_v260_fixed10_cost_start_cross_robustness_20260822"
)
COSTS = (0.0030, 0.0040, 0.0065)
START_OFFSETS = (0, 5, 20, 60)


def matrix_summary(matrix: dict) -> dict:
    cells = [cell for costs in matrix.values() for cell in costs.values()]
    return {
        "cell_count": len(cells),
        "all_cumulative_returns_positive": all(
            cell["cumulative_return"] > 0.0 for cell in cells
        ),
        "all_cagrs_positive": all(cell["cagr"] > 0.0 for cell in cells),
        "minimum_cumulative_return": float(
            min(cell["cumulative_return"] for cell in cells)
        ),
        "minimum_cagr": float(min(cell["cagr"] for cell in cells)),
        "maximum_drawdown": float(max(cell["max_drawdown"] for cell in cells)),
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint.get("validation_2026_opened") is not False:
        raise PermissionError("cost-start audit cannot consume 2026 validation")
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

    matrix = {}
    baseline_daily = baseline_actions = None
    for cost in COSTS:
        daily, actions = age_guard.run_policy_at_cost(
            context,
            policy,
            extra_age,
            cost,
            score_sell_priority_override=sell_priority,
        )
        if cost == 0.0030:
            baseline_daily, baseline_actions = daily, actions
        matrix[f"{cost:.4f}"] = {
            str(offset): round1.evaluate_run(
                daily,
                actions,
                (
                    research_base.FIRST_BUY
                    if offset == 0
                    else robustness.window_start_date(context.arrays, offset)
                ),
                round1.DEVELOPMENT_END,
            )
            for offset in START_OFFSETS
        }

    expected = checkpoint["current_best_equalweight"]
    baseline = matrix["0.0030"]["0"]
    equivalence = {
        key: bool(np.isclose(baseline[key], expected[key], rtol=0.0, atol=1e-12))
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
        raise RuntimeError("cost-start audit baseline drifted")
    summary = matrix_summary(matrix)
    if not summary["all_cumulative_returns_positive"]:
        raise RuntimeError("cost-start audit found a non-positive cumulative return")

    repeat_daily, repeat_actions = age_guard.run_policy_at_cost(
        context,
        policy,
        extra_age,
        0.0030,
        score_sell_priority_override=sell_priority,
    )
    deterministic = {
        "daily": round1.frame_hash(baseline_daily) == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(baseline_actions)
        == round1.frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("cost-start audit replay failed")

    result = {
        "status": "pre2026_cost_start_cross_robustness_complete",
        "method": (
            "cross the frozen 0.30/0.40/0.65 percent cost assumptions with frozen "
            "0/5/20/60-session evaluation starts; no rule or parameter selection"
        ),
        "matrix": matrix,
        "summary": summary,
        "baseline_equivalence": equivalence,
        "deterministic_replay": deterministic,
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "cost_start_cross_robustness.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "summary": summary,
                "validation_2026_opened": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
