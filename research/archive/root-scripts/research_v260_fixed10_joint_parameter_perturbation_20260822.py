from __future__ import annotations

import copy
import itertools
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
import research_v260_lowrisk_score_tuning_20260821 as research_base


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_joint_parameter_perturbation_20260822"
)
CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)


def fractional_factorial_cases() -> list[dict]:
    cases = []
    for a, b, c in itertools.product((-1, 1), repeat=3):
        d = a * b * c
        cases.append({
            "case_id": f"A{a:+d}_B{b:+d}_C{c:+d}_D{d:+d}",
            "min_hold_days": 4 + a,
            "sell_score_below": 0.85 + 0.01 * b,
            "replacement_advantage": 0.05 + 0.01 * c,
            "volatility_penalty": 0.05 + 0.01 * d,
        })
    return cases


def summarize(results: dict, production: dict) -> dict:
    baseline = [value["metrics_0_30pct"] for value in results.values()]
    stress = [value["metrics_0_65pct"] for value in results.values()]

    def distribution(key: str, rows: list[dict]) -> dict:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        return {
            "min": float(values.min()),
            "median": float(np.median(values)),
            "max": float(values.max()),
        }

    return {
        "case_count": len(results),
        "cagr": distribution("cagr", baseline),
        "sharpe": distribution("sharpe", baseline),
        "max_drawdown": distribution("max_drawdown", baseline),
        "stress_cumulative_return": distribution("cumulative_return", stress),
        "all_cases_all_calendar_years_positive": bool(all(
            min(row["annual_returns"].values()) > 0.0 for row in baseline
        )),
        "all_cases_exactly10": bool(all(
            row["full_10_position_ratio"] == 1.0 for row in baseline
        )),
        "fraction_cagr_above_production": float(np.mean([
            row["cagr"] > production["cagr"] for row in baseline
        ])),
        "fraction_cumulative_return_above_production": float(np.mean([
            row["cumulative_return"] > production["cumulative_return"]
            for row in baseline
        ])),
        "fraction_stress_positive": float(np.mean([
            row["cumulative_return"] > 0.0 for row in stress
        ])),
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered joint perturbation diagnostic")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)
    strong = regime.strong_market_mask(context.score, context.protocol)
    weak2 = confirmed.confirmed_weak_mask(strong, 2)
    extra_age = age_boundary.pressure_age_schedule(strong, 10)
    volatility_rank = percentile.cross_sectional_percent_rank(
        defensive.trailing_log_volatility(
            context.arrays["close_qfq"], defensive.VOLATILITY_WINDOW
        )
    )

    cases = fractional_factorial_cases()
    results = {}
    for case in cases:
        sell_priority = priority.sell_priority_matrix(
            context.score,
            volatility_rank,
            ~weak2,
            case["volatility_penalty"],
        )
        daily, actions = age_guard.run_policy_at_cost(
            context,
            policy,
            extra_age,
            round1.BASELINE_COST,
            min_hold_days_override=case["min_hold_days"],
            sell_score_below_override=case["sell_score_below"],
            replacement_advantage_override=case["replacement_advantage"],
            score_sell_priority_override=sell_priority,
        )
        stress_daily, stress_actions = age_guard.run_policy_at_cost(
            context,
            policy,
            extra_age,
            round1.STRESS_COST,
            min_hold_days_override=case["min_hold_days"],
            sell_score_below_override=case["sell_score_below"],
            replacement_advantage_override=case["replacement_advantage"],
            score_sell_priority_override=sell_priority,
        )
        results[case["case_id"]] = {
            "parameters": {key: value for key, value in case.items() if key != "case_id"},
            "metrics_0_30pct": round1.evaluate_run(
                daily, actions, research_base.FIRST_BUY, round1.DEVELOPMENT_END
            ),
            "metrics_0_65pct": round1.evaluate_run(
                stress_daily,
                stress_actions,
                research_base.FIRST_BUY,
                round1.DEVELOPMENT_END,
            ),
        }

    summary = summarize(results, checkpoint["production_baseline"])
    result = {
        "status": "joint_parameter_perturbation_complete_2026_not_opened",
        "diagnostic_only_no_parameter_selection": True,
        "design": {
            "type": "balanced_eight_run_fractional_factorial_D_equals_ABC",
            "parameters": {
                "A_min_hold_days": [3, 5],
                "B_sell_score_below": [0.84, 0.86],
                "C_replacement_advantage": [0.04, 0.06],
                "D_volatility_penalty": [0.04, 0.06],
            },
        },
        "results": results,
        "robustness_summary": summary,
        "interpretation": (
            "The selected candidate is not replaced by a corner. The diagnostic asks "
            "whether simultaneous small perturbations preserve broad profitability."
        ),
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "joint_parameter_perturbation.json", result)
    print(json.dumps({
        "status": result["status"],
        "robustness_summary": summary,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
