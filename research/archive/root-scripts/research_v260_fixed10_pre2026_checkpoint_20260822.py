from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_portfolio_rebalance_band_20260822 as final_round
import research_v260_fixed10_maintenance_quality_gate_20260822 as maintenance_round
import research_v260_lowrisk_score_tuning_20260821 as research_base
from research_v260_runtime import fixed10_risk_event_v110 as maintenance_runtime_v110
from research_v260_runtime import fixed10_safe_v109 as safe_v109


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/strategy_agent_v260_fixed10_pre2026_checkpoint_20260822"
)


def key_deltas(candidate: dict, baseline: dict) -> dict:
    return {
        key: float(candidate[key] - baseline[key])
        for key in (
            "cumulative_return",
            "cagr",
            "sharpe",
            "max_drawdown",
            "turnover_annualized",
            "average_invested_ratio",
            "average_positions",
        )
    }


def beats_production(candidate: dict, production: dict) -> bool:
    return bool(
        candidate["cagr"] > production["cagr"]
        and candidate["sharpe"] >= production["sharpe"]
        and candidate["max_drawdown"] <= production["max_drawdown"]
    )


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    source = json.loads(
        (
            maintenance_round.OUTPUT_ROOT
            / "development_result.json"
        ).read_text(encoding="utf-8")
    )
    if source["validation_2026_opened"]:
        raise PermissionError("selected rule was contaminated by 2026 validation")
    selected_policy = source["selected_policy"]

    harness, protocol, rules, manifests, arrays, access = round1.load_arrays(
        round1.DEVELOPMENT_END
    )
    if access["logical_max_date"] != round1.DEVELOPMENT_END:
        raise PermissionError("2026 data entered pre-2026 checkpoint")
    definition = harness.production_definition(protocol)
    score, order = harness.v95.score_pair(arrays, 0.0, 7, 0.1)

    production_daily, production_actions = research_base.run_shell(
        harness,
        arrays,
        protocol,
        score,
        order,
        definition,
        "production_shell",
        round1.DEVELOPMENT_END,
        actions=True,
    )
    naive_daily, naive_actions = round1.run_fixed10(
        harness,
        arrays,
        protocol,
        definition,
        score,
        order,
        round1.HOLD_POLICIES["production_exit_fixed10"],
        round1.DEVELOPMENT_END,
        slip=round1.BASELINE_COST,
        record_actions=True,
        simulator=safe_v109.simulate,
    )
    candidate_daily, candidate_actions = round1.run_fixed10(
        harness,
        arrays,
        protocol,
        definition,
        score,
        order,
        selected_policy,
        round1.DEVELOPMENT_END,
        slip=round1.BASELINE_COST,
        record_actions=True,
        simulator=maintenance_runtime_v110.simulate,
        portfolio_rebalance_active_override=final_round.cadence.rebalance_schedule(
            len(arrays["dates"]),
            selected_policy["portfolio_rebalance_interval_days"],
        ),
        portfolio_rebalance_min_weight_deviation_override=selected_policy[
            "portfolio_rebalance_min_weight_deviation"
        ],
        entry_block_mask_override=np.zeros(score.shape, dtype=np.bool_),
        maintenance_buy_block_mask_override=maintenance_round.maintenance_quality_block(
            score,
            selected_policy.get("maintenance_topup_requires_score") is not None,
        ),
    )

    production_metrics = round1.evaluate_run(
        production_daily,
        production_actions,
        research_base.FIRST_BUY,
        round1.DEVELOPMENT_END,
    )
    naive_metrics = round1.evaluate_run(
        naive_daily,
        naive_actions,
        research_base.FIRST_BUY,
        round1.DEVELOPMENT_END,
    )
    candidate_metrics = round1.evaluate_run(
        candidate_daily,
        candidate_actions,
        research_base.FIRST_BUY,
        round1.DEVELOPMENT_END,
    )
    superior = beats_production(candidate_metrics, production_metrics)
    result = {
        "status": (
            "pre2026_equalweight_candidate_superior_to_production"
            if superior
            else "pre2026_equalweight_family_best_but_not_superior_to_production"
        ),
        "source_strategy": rules["strategy_id"],
        "development_boundary": [research_base.FIRST_BUY, round1.DEVELOPMENT_END],
        "score_rule_changed": False,
        "market_timing_added": False,
        "selected_policy": selected_policy,
        "production_baseline": production_metrics,
        "naive_fixed10_baseline": naive_metrics,
        "current_best_equalweight": candidate_metrics,
        "candidate_minus_production": key_deltas(candidate_metrics, production_metrics),
        "candidate_minus_naive_fixed10": key_deltas(candidate_metrics, naive_metrics),
        "superior_to_production": superior,
        "development_source_result": str(
            maintenance_round.OUTPUT_ROOT / "development_result.json"
        ),
        "data_access": access,
        "source_manifests": manifests,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "pre2026_checkpoint.json", result)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
