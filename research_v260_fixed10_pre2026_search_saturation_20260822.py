from __future__ import annotations

import json
import sys
from pathlib import Path


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_hold_optimization_20260822 as round1


REPORTS = REPO / "quant/data_file/reports"
OUTPUT_ROOT = (
    REPORTS / "strategy_agent_v260_fixed10_pre2026_search_saturation_20260822"
)
CANDIDATE_ID = "fixed10_global_rank11_to_9_entry_buy_day_cash_sweep"


def read(relative: str) -> dict:
    value = json.loads((REPORTS / relative).read_text(encoding="utf-8"))
    if value.get("validation_2026_opened") is not False:
        raise PermissionError(f"source is not explicitly pre-2026: {relative}")
    if value.get("production_modified") is not False:
        raise PermissionError(f"source modified production: {relative}")
    return value


def saturation_gates(
    checkpoint: dict,
    historical_checkpoint: dict,
    rule_generalization: dict,
    turnover: dict,
    exit_confirmation: dict,
    event_policy: dict,
    pairwise: dict,
) -> dict[str, bool]:
    attribution = turnover["turnover_attribution"]
    return {
        "candidate_is_frozen": checkpoint["selected_candidate"] == CANDIDATE_ID,
        "portfolio_direction_recorded_as_soft": bool(
            checkpoint["selected_policy"]["entry_rank_sizing"][
                "equalweight_is_soft_reference"
            ]
            and checkpoint["selected_policy"]["residual_cash_sweep"][
                "equalweight_and_full_investment_are_soft_directions"
            ]
        ),
        "all_generalization_rules_classified": all(
            item.get("classification")
            for item in rule_generalization["rules"].values()
        ),
        "no_unsupported_rule_left_open": not rule_generalization[
            "unsupported_rules"
        ],
        "replacement_turnover_dominates": attribution[
            "replacement_turnover_share"
        ]
        >= 0.95,
        "maintenance_turnover_is_minor": attribution[
            "equalweight_maintenance_turnover_share"
        ]
        < 0.05,
        "two_day_exit_confirmation_rejected": exit_confirmation[
            "selected_candidate"
        ]
        == "score_exit_confirmation_1d",
        "event_overlay_is_diagnostic_only": event_policy[
            "event_overlay_role"
        ]
        == "diagnostic_only_not_selection_candidate",
        "no_dominating_pair_removal": not pairwise["dominating_pair_removals"],
        "validation_2026_unopened": all(
            item["validation_2026_opened"] is False
            for item in (
                checkpoint,
                historical_checkpoint,
                rule_generalization,
                turnover,
                exit_confirmation,
                event_policy,
                pairwise,
            )
        ),
        "production_unchanged": all(
            item["production_modified"] is False
            for item in (
                checkpoint,
                historical_checkpoint,
                rule_generalization,
                turnover,
                exit_confirmation,
                event_policy,
                pairwise,
            )
        ),
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = read(
        "strategy_agent_v260_fixed10_residual_cash_sweep_checkpoint_20260823/"
        "pre2026_checkpoint.json"
    )
    historical_checkpoint = read(
        "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
        "pre2026_checkpoint.json"
    )
    rule_generalization = read(
        "strategy_agent_v260_fixed10_rule_generalization_audit_20260822/"
        "rule_generalization_audit.json"
    )
    turnover = read(
        "strategy_agent_v260_fixed10_turnover_source_attribution_20260822/"
        "turnover_source_attribution.json"
    )
    exit_confirmation = read(
        "strategy_agent_v260_fixed10_current_exit_confirmation_recheck_20260822/"
        "development_result.json"
    )
    event_policy = read(
        "strategy_agent_v260_fixed10_tushare_event_overlay_freeze_20260822/"
        "event_overlay_contract.json"
    )
    pairwise = read(
        "strategy_agent_v260_fixed10_pairwise_rule_interaction_audit_20260822/"
        "pairwise_rule_interaction_audit.json"
    )
    gates = saturation_gates(
        checkpoint,
        historical_checkpoint,
        rule_generalization,
        turnover,
        exit_confirmation,
        event_policy,
        pairwise,
    )
    if not all(gates.values()):
        raise RuntimeError(
            "pre-2026 search is not saturated: "
            + ", ".join(name for name, passed in gates.items() if not passed)
        )
    result = {
        "status": "pre2026_broad_rule_search_saturated_2026_not_opened",
        "candidate": CANDIDATE_ID,
        "current_metrics": checkpoint["metrics_0_30pct"],
        "production_metrics": historical_checkpoint["production_baseline"],
        "tested_rule_families": {
            "holding_horizon_and_renewal": "closed",
            "score_exit_threshold_and_confirmation": "closed",
            "replacement_advantage_and_pressure_exit": "closed",
            "equalweight_maintenance_and_topup": "closed",
            "market_state_and_volatility_priority": "closed_with_validation_risk",
            "limit_up_and_open_gap_handling": "closed",
            "abnormal_severe_and_exchange_alert_events": "diagnostic_only_for_2026",
            "cost_capital_scale_and_start_offset": "closed",
        },
        "saturation_gates": gates,
        "remaining_pre2026_selection_candidates": [],
        "search_decision": (
            "Stop adding pre-2026 rule branches. Continue reproducibility and boundary "
            "checks until the requested 24-hour duration, then execute the frozen "
            "one-shot 2026 validation exactly once."
        ),
        "known_validation_risks": [
            "candidate improves return but not Sharpe or drawdown versus production",
            "relative outperformance is weaker in 2022 and concentrated in a few top days",
            "residual cash sweep adds only a small part of drawdown; fixed10 exposure dominates risk",
            "official Tushare event history is concentrated in 2026 and remains diagnostic",
        ],
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "search_saturation.json", result)
    print(json.dumps({
        "status": result["status"],
        "candidate": result["candidate"],
        "all_gates_passed": all(gates.values()),
        "remaining_candidates": 0,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
