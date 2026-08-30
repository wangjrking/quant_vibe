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
    REPORTS / "strategy_agent_v260_fixed10_pre2026_optimization_ledger_20260822"
)


def read(relative: str) -> dict:
    value = json.loads((REPORTS / relative).read_text(encoding="utf-8"))
    if value.get("validation_2026_opened") is not False:
        raise PermissionError(f"ledger source is not explicitly pre-2026: {relative}")
    if value.get("production_modified") is not False:
        raise PermissionError(f"ledger source modified production: {relative}")
    return value


def compact_profit_table(report: dict) -> dict:
    return {
        name: {
            "cumulative_return_0_30pct": item["metrics_0_30pct"][
                "cumulative_return"
            ],
            "sharpe_0_30pct": item["metrics_0_30pct"]["sharpe"],
            "max_drawdown_0_30pct": item["metrics_0_30pct"][
                "max_drawdown"
            ],
            "cumulative_return_0_65pct": item["metrics_0_65pct"][
                "cumulative_return"
            ],
        }
        for name, item in report["candidates"].items()
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = read(
        "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
        "pre2026_checkpoint.json"
    )
    rolling = read(
        "strategy_agent_v260_fixed10_rolling_relative_robustness_20260822/"
        "rolling_relative_robustness.json"
    )
    bootstrap = read(
        "strategy_agent_v260_fixed10_paired_block_bootstrap_20260822/"
        "paired_block_bootstrap.json"
    )
    concentration = read(
        "strategy_agent_v260_fixed10_return_concentration_diagnostic_20260822/"
        "return_concentration.json"
    )
    cost = read(
        "strategy_agent_v260_fixed10_cost_frontier_20260822/cost_frontier.json"
    )
    ablation = read(
        "strategy_agent_v260_fixed10_current_rule_ablation_20260822/"
        "rule_ablation.json"
    )
    rank_noise = read(
        "strategy_agent_v260_fixed10_score_noise_rank_calibration_20260822/"
        "score_noise_rank_calibration.json"
    )
    small_noise = read(
        "strategy_agent_v260_fixed10_small_score_noise_robustness_20260822/"
        "small_score_noise_robustness.json"
    )
    event_overlay = read(
        "strategy_agent_v260_fixed10_tushare_event_overlay_freeze_20260822/"
        "event_overlay_contract.json"
    )
    minimal_event_policy = read(
        "strategy_agent_v260_fixed10_risk_event_minimal_policy_20260822/"
        "development_result.json"
    )
    event_score_tiebreak = read(
        "strategy_agent_v260_fixed10_risk_event_score_tiebreak_20260822/"
        "development_result.json"
    )
    local_stability = read(
        "strategy_agent_v260_fixed10_local_parameter_stability_20260822/"
        "local_parameter_stability.json"
    )
    vol_priority_bootstrap = read(
        "strategy_agent_v260_fixed10_vol_priority_increment_bootstrap_20260822/"
        "vol_priority_increment_bootstrap.json"
    )
    vol_priority_action_concentration = read(
        "strategy_agent_v260_fixed10_vol_priority_action_concentration_20260822/"
        "action_concentration.json"
    )
    vol_priority_binary = read(
        "strategy_agent_v260_fixed10_vol_priority_binary_simplification_20260822/"
        "development_result.json"
    )
    noise_production_margin = read(
        "strategy_agent_v260_fixed10_score_noise_production_margin_20260822/"
        "score_noise_production_margin.json"
    )
    rank_attribution = read(
        "strategy_agent_v260_fixed10_rank_membership_order_attribution_20260822/"
        "rank_membership_order_attribution.json"
    )
    action_rank = read(
        "strategy_agent_v260_fixed10_action_rank_diagnostic_20260822/"
        "action_rank_diagnostic.json"
    )
    leave_one_year_out = read(
        "strategy_agent_v260_fixed10_leave_one_year_out_parameter_stability_20260822/"
        "leave_one_year_out_parameter_stability.json"
    )
    core_neighborhood = read(
        "strategy_agent_v260_fixed10_current_core_parameter_neighborhood_20260822/"
        "current_core_parameter_neighborhood.json"
    )
    staleness = read(
        "strategy_agent_v260_fixed10_decision_staleness_robustness_20260822/"
        "decision_staleness_robustness.json"
    )
    exit_timing = read(
        "strategy_agent_v260_fixed10_exit_timing_robustness_tradeoff_20260822/"
        "exit_timing_tradeoff.json"
    )
    event_block_robustness = read(
        "strategy_agent_v260_fixed10_event_entry_block_robustness_20260822/"
        "event_entry_block_robustness.json"
    )
    score_precision = read(
        "strategy_agent_v260_fixed10_score_precision_robustness_20260822/"
        "score_precision_robustness.json"
    )
    precision_volatility_tiebreak = read(
        "strategy_agent_v260_fixed10_score_precision_volatility_tiebreak_20260822/"
        "development_result.json"
    )
    weak_size_guard = read(
        "strategy_agent_v260_fixed10_current_weak_market_size_guard_recheck_20260822/"
        "development_result.json"
    )
    open_gap_guard = read(
        "strategy_agent_v260_fixed10_current_open_gap_guard_recheck_20260822/"
        "development_result.json"
    )
    limitup_exit = read(
        "strategy_agent_v260_fixed10_current_limitup_exit_recheck_20260822/"
        "development_result.json"
    )
    vol_size_priority = read(
        "strategy_agent_v260_fixed10_current_weak_market_vol_size_sell_priority_20260822/"
        "development_result.json"
    )
    candidate_frontier = read(
        "strategy_agent_v260_fixed10_pre2026_candidate_frontier_20260822/"
        "candidate_frontier.json"
    )
    validation_protocol = read(
        "strategy_agent_v260_fixed10_one_shot_2026_validation_protocol_20260822/"
        "one_shot_validation_protocol.json"
    )
    exposure_attribution = read(
        "strategy_agent_v260_fixed10_current_vs_production_exposure_attribution_20260822/"
        "exposure_attribution.json"
    )
    joint_perturbation = read(
        "strategy_agent_v260_fixed10_joint_parameter_perturbation_20260822/"
        "joint_parameter_perturbation.json"
    )
    min_hold_fragility = read(
        "strategy_agent_v260_fixed10_min_hold4_fragility_diagnostic_20260822/"
        "min_hold4_fragility_diagnostic.json"
    )
    core_parameter_support = read(
        "strategy_agent_v260_fixed10_core_parameter_support_matrix_20260822/"
        "core_parameter_support_matrix.json"
    )
    weak_confirmation_support = read(
        "strategy_agent_v260_fixed10_weak_confirmation_support_diagnostic_20260822/"
        "weak_confirmation_support_diagnostic.json"
    )
    rule_ablation_support = read(
        "strategy_agent_v260_fixed10_rule_ablation_support_matrix_20260822/"
        "rule_ablation_support_matrix.json"
    )
    capital_scale = read(
        "strategy_agent_v260_fixed10_capital_scale_robustness_20260822/"
        "capital_scale_robustness.json"
    )
    market_state_branch = read(
        "strategy_agent_v260_fixed10_market_state_branch_support_20260822/"
        "market_state_branch_support.json"
    )
    state_simplification = read(
        "strategy_agent_v260_fixed10_state_simplification_stack_20260822/"
        "state_simplification_stack.json"
    )
    prevalidation_risk = read(
        "strategy_agent_v260_fixed10_prevalidation_risk_scorecard_20260822/"
        "prevalidation_risk_scorecard.json"
    )
    full_investment_tradeoff = read(
        "strategy_agent_v260_fixed10_full_investment_tradeoff_20260822/"
        "full_investment_tradeoff.json"
    )
    rule_generalization = read(
        "strategy_agent_v260_fixed10_rule_generalization_audit_20260822/"
        "rule_generalization_audit.json"
    )
    execution_timing = read(
        "strategy_agent_v260_fixed10_execution_timing_audit_20260822/"
        "execution_timing_audit.json"
    )
    pairwise_rules = read(
        "strategy_agent_v260_fixed10_pairwise_rule_interaction_audit_20260822/"
        "pairwise_rule_interaction_audit.json"
    )
    cost_start_cross = read(
        "strategy_agent_v260_fixed10_cost_start_cross_robustness_20260822/"
        "cost_start_cross_robustness.json"
    )
    drawdown_position_age = read(
        "strategy_agent_v260_fixed10_drawdown_position_age_attribution_20260822/"
        "position_age_attribution.json"
    )
    drawdown_score_state = read(
        "strategy_agent_v260_fixed10_drawdown_score_state_attribution_20260822/"
        "score_state_attribution.json"
    )
    current_candidate_drawdown = read(
        "strategy_agent_v260_fixed10_buy_day_cash_sweep_drawdown_attribution_20260823/"
        "drawdown_attribution.json"
    )
    current_candidate_vs_production = read(
        "strategy_agent_v260_fixed10_buy_day_cash_sweep_vs_production_robustness_20260823/"
        "robustness.json"
    )
    turnover_source = read(
        "strategy_agent_v260_fixed10_turnover_source_attribution_20260822/"
        "turnover_source_attribution.json"
    )
    replacement_counterfactual = read(
        "strategy_agent_v260_fixed10_replacement_counterfactual_diagnostic_20260822/"
        "replacement_diagnostic.json"
    )
    replacement_failure = read(
        "strategy_agent_v260_fixed10_replacement_failure_segmentation_20260822/"
        "segmentation.json"
    )
    exit_confirmation = read(
        "strategy_agent_v260_fixed10_current_exit_confirmation_recheck_20260822/"
        "development_result.json"
    )
    search_saturation = read(
        "strategy_agent_v260_fixed10_pre2026_search_saturation_20260822/"
        "search_saturation.json"
    )
    freeze_integrity = read(
        "strategy_agent_v260_fixed10_pre2026_freeze_integrity_20260822/"
        "freeze_integrity.json"
    )
    goal_readiness = read(
        "strategy_agent_v260_fixed10_goal_completion_readiness_20260822/"
        "goal_completion_readiness.json"
    )
    soft_width = read(
        "strategy_agent_v260_fixed10_soft_width_profit_optimization_20260822/"
        "soft_width_results.json"
    )
    soft_gross = read(
        "strategy_agent_v260_fixed10_soft_gross_profit_optimization_20260822/"
        "soft_gross_results.json"
    )
    entry_1d = read(
        "strategy_agent_v260_fixed10_current_entry_score_profit_optimization_20260822/"
        "development_result.json"
    )
    entry_3d = read(
        "strategy_agent_v260_fixed10_current_entry_3d_profit_optimization_20260822/"
        "development_result.json"
    )
    entry_5d = read(
        "strategy_agent_v260_fixed10_current_entry_5d_profit_optimization_20260822/"
        "development_result.json"
    )
    entry_smoothing = read(
        "strategy_agent_v260_fixed10_current_smoothing_profit_optimization_20260822/"
        "development_result.json"
    )
    entry_latest_alpha = read(
        "strategy_agent_v260_fixed10_current_raw_alpha_profit_optimization_20260822/"
        "development_result.json"
    )
    exit_smoothing = read(
        "strategy_agent_v260_fixed10_current_exit_smoothing_profit_optimization_20260822/"
        "development_result.json"
    )
    exit_latest_alpha = read(
        "strategy_agent_v260_fixed10_current_exit_raw_alpha_profit_optimization_20260822/"
        "development_result.json"
    )
    exit_horizon = read(
        "strategy_agent_v260_fixed10_current_exit_horizon_profit_optimization_20260822/"
        "development_result.json"
    )
    partial_trim = read(
        "strategy_agent_v260_fixed10_current_partial_trim_profit_optimization_20260822/"
        "development_result.json"
    )
    score_transfer = read(
        "strategy_agent_v260_fixed10_current_score_transfer_profit_optimization_20260822/"
        "development_result.json"
    )
    holdout_consistency = read(
        "strategy_agent_v260_fixed10_profit_first_holdout_consistency_20260822/"
        "profit_first_holdout_consistency.json"
    )
    entry_rank_sizing = read(
        "strategy_agent_v260_fixed10_entry_rank_sizing_profit_optimization_20260822/"
        "development_result.json"
    )
    rank_sizing_checkpoint = read(
        "strategy_agent_v260_fixed10_residual_cash_sweep_checkpoint_20260823/"
        "pre2026_checkpoint.json"
    )
    rank_sizing_leave_one_year_out = read(
        "strategy_agent_v260_fixed10_rank_sizing_leave_one_year_out_20260822/"
        "leave_one_year_out.json"
    )
    rank_sizing_scale_concentration = read(
        "strategy_agent_v260_fixed10_rank_sizing_scale_concentration_20260823/"
        "scale_concentration.json"
    )
    rank_sizing_market_state = read(
        "strategy_agent_v260_fixed10_rank_sizing_market_state_attribution_20260823/"
        "market_state_attribution.json"
    )
    rank_sizing_mechanism = read(
        "strategy_agent_v260_fixed10_rank_sizing_mechanism_attribution_20260823/"
        "mechanism_attribution.json"
    )
    rank_sizing_maintenance = read(
        "strategy_agent_v260_fixed10_rank_sizing_maintenance_profit_optimization_20260823/"
        "development_result.json"
    )
    rank_sizing_reverse_placebo = read(
        "strategy_agent_v260_fixed10_rank_sizing_reverse_placebo_20260823/"
        "reverse_placebo.json"
    )
    rank_sizing_neutral_placebo = read(
        "strategy_agent_v260_fixed10_rank_sizing_neutral_placebo_20260823/"
        "neutral_placebo.json"
    )
    low_vol_entry_sizing = read(
        "strategy_agent_v260_fixed10_low_vol_entry_sizing_profit_optimization_20260823/"
        "development_result.json"
    )
    score_distance_entry_sizing = read(
        "strategy_agent_v260_fixed10_score_distance_entry_sizing_20260823/"
        "development_result.json"
    )
    refill_haircut = read(
        "strategy_agent_v260_fixed10_refill_haircut_profit_optimization_20260823/"
        "development_result.json"
    )
    friction_compensated_gross = read(
        "strategy_agent_v260_fixed10_friction_compensated_gross_20260823/"
        "development_result.json"
    )
    cash_aware_batch_allocation = read(
        "strategy_agent_v260_fixed10_cash_aware_batch_allocation_20260823/"
        "development_result.json"
    )
    rank_sizing_midpoint = read(
        "strategy_agent_v260_fixed10_rank_sizing_midpoint_profit_check_20260823/"
        "development_result.json"
    )
    rank_sizing_walkforward = read(
        "strategy_agent_v260_fixed10_rank_sizing_walkforward_amplitude_20260823/"
        "walkforward_amplitude.json"
    )
    rank_sizing_shape = read(
        "strategy_agent_v260_fixed10_rank_sizing_shape_profit_optimization_20260823/"
        "development_result.json"
    )
    entry_rank_realization = read(
        "strategy_agent_v260_fixed10_entry_rank_realization_diagnostic_20260823/"
        "entry_rank_realization.json"
    )
    rank_sizing_topup_simplification = read(
        "strategy_agent_v260_fixed10_rank_sizing_topup_gate_simplification_20260823/"
        "development_result.json"
    )
    rank_sizing_sell_priority_simplification = read(
        "strategy_agent_v260_fixed10_rank_sizing_sell_priority_simplification_20260823/"
        "development_result.json"
    )
    rank_sizing_pressure_trigger_simplification = read(
        "strategy_agent_v260_fixed10_rank_sizing_pressure_trigger_simplification_20260823/"
        "development_result.json"
    )
    rank_sizing_pressure_age_simplification = read(
        "strategy_agent_v260_fixed10_rank_sizing_pressure_age_simplification_20260823/"
        "development_result.json"
    )
    residual_cash_sweep = read(
        "strategy_agent_v260_fixed10_rank_sizing_residual_cash_sweep_20260823/"
        "development_result.json"
    )
    residual_cash_sweep_robustness = read(
        "strategy_agent_v260_fixed10_residual_cash_sweep_robustness_20260823/"
        "robustness.json"
    )
    residual_cash_sweep_scale = read(
        "strategy_agent_v260_fixed10_residual_cash_sweep_capital_scale_20260823/"
        "capital_scale.json"
    )
    sweep_rank_interaction = read(
        "strategy_agent_v260_fixed10_sweep_rank_interaction_20260823/"
        "development_result.json"
    )
    cash_sweep_order_simplification = read(
        "strategy_agent_v260_fixed10_cash_sweep_order_simplification_20260823/"
        "development_result.json"
    )
    cash_sweep_cost_frontier = read(
        "strategy_agent_v260_fixed10_residual_cash_sweep_cost_frontier_20260823/"
        "cost_frontier.json"
    )
    cash_sweep_mechanism = read(
        "strategy_agent_v260_fixed10_residual_cash_sweep_mechanism_attribution_20260823/"
        "mechanism_attribution.json"
    )
    cash_sweep_trigger = read(
        "strategy_agent_v260_fixed10_cash_sweep_trigger_simplification_20260823/"
        "development_result.json"
    )
    cash_sweep_trigger_robustness = read(
        "strategy_agent_v260_fixed10_cash_sweep_trigger_robustness_20260823/"
        "robustness.json"
    )
    cash_sweep_recipient_score = read(
        "strategy_agent_v260_fixed10_cash_sweep_recipient_score_20260823/"
        "development_result.json"
    )

    rejected_sources = {
        "entry_rank_persistence": (
            "strategy_agent_v260_fixed10_entry_rank_persistence_20260822/"
            "development_result.json"
        ),
        "prior_day_entry_score_confirmation": (
            "strategy_agent_v260_fixed10_entry_score_confirmation_20260822/"
            "development_result.json"
        ),
        "liquidity_tiebreak": (
            "strategy_agent_v260_fixed10_liquidity_tiebreak_20260822/"
            "development_result.json"
        ),
        "score_precision_volatility_tiebreak": (
            "strategy_agent_v260_fixed10_score_precision_volatility_tiebreak_20260822/"
            "development_result.json"
        ),
        "strong_market_three_exit_limit": (
            "strategy_agent_v260_fixed10_strong_market_pressure_limit3_20260822/"
            "development_result.json"
        ),
        "tushare_event_entry_gate": (
            "strategy_agent_v260_fixed10_tushare_risk_event_entry_gate_20260822/"
            "development_result.json"
        ),
        "weak_market_size_guard": (
            "strategy_agent_v260_fixed10_current_weak_market_size_guard_recheck_20260822/"
            "development_result.json"
        ),
        "entry_open_gap_guard": (
            "strategy_agent_v260_fixed10_current_open_gap_guard_recheck_20260822/"
            "development_result.json"
        ),
        "defer_open_limit_up_sell": (
            "strategy_agent_v260_fixed10_current_limitup_exit_recheck_20260822/"
            "development_result.json"
        ),
        "confirmed_weak_microcap_sell_priority": (
            "strategy_agent_v260_fixed10_current_weak_market_vol_size_sell_priority_20260822/"
            "development_result.json"
        ),
    }
    rejected = {}
    for name, source in rejected_sources.items():
        result = read(source)
        rejected[name] = {
            "status": result["status"],
            "selected_candidate": result.get("selected_candidate"),
            "reason": (
                "no pre-2026 event rows; leave rule inactive"
                if name == "tushare_event_entry_gate"
                else "failed frozen development/holdout robustness gates"
            ),
            "source": source,
        }

    metrics = rank_sizing_checkpoint["metrics_0_30pct"]
    production = checkpoint["production_baseline"]
    result = {
        "status": "pre2026_optimization_ledger_current_2026_validation_unopened",
        "current_candidate": {
            "candidate_id": rank_sizing_checkpoint["selected_candidate"],
            "policy": rank_sizing_checkpoint["selected_policy"],
            "metrics_0_30pct": metrics,
            "delta_vs_production": {
                key: float(metrics[key] - production[key])
                for key in (
                    "cumulative_return", "cagr", "sharpe", "max_drawdown",
                    "turnover_annualized", "average_invested_ratio",
                )
            },
        },
        "profit_first_soft_experiments": {
            "selection_policy": (
                "maximize pre-2026 net cumulative return; treat position count, "
                "equal target weight and invested ratio as soft diagnostics; disclose "
                "drawdown, cost and turnover without using 10/10pct/full-investment "
                "as automatic rejection gates"
            ),
            "hard_safety_only": [
                "PIT_and_no_future_data",
                "T_plus_1_and_real_execution_prices",
                "cost_capacity_cash_conservation",
                "finite_reproducible_results",
                "production_assets_unchanged",
            ],
            "portfolio_width": {
                "selected": soft_width["selected_width"],
                "table": compact_profit_table(soft_width),
                "interpretation": (
                    "10 positions won the tested 5/7/8/9/10/11/12/15/20 profit "
                    "comparison; "
                    "it was selected by return rather than enforced by contract"
                ),
            },
            "target_gross": {
                "selected": soft_gross["selected_gross"],
                "table": compact_profit_table(soft_gross),
                "interpretation": (
                    "100% gross won the tested 90/95/100% profit comparison; "
                    "cash level remains a tunable economic choice"
                ),
            },
            "entry_score": {
                "one_day_blends": compact_profit_table(entry_1d),
                "three_day_blends": compact_profit_table(entry_3d),
                "five_day_blends": compact_profit_table(entry_5d),
                "selected": "pure_10d",
            },
            "entry_smoothing": {
                "selected": entry_smoothing["selected_smoothing_window"],
                "table": compact_profit_table(entry_smoothing),
            },
            "entry_latest_session_weight": {
                "selected": entry_latest_alpha["selected_raw_alpha"],
                "table": compact_profit_table(entry_latest_alpha),
            },
            "exit_score": {
                "selected_smoothing": exit_smoothing[
                    "selected_exit_smoothing_window"
                ],
                "smoothing_table": compact_profit_table(exit_smoothing),
                "selected_latest_session_weight": exit_latest_alpha[
                    "selected_exit_raw_alpha"
                ],
                "latest_session_table": compact_profit_table(exit_latest_alpha),
                "selected_horizon": exit_horizon["selected_exit_horizon"],
                "horizon_table": compact_profit_table(exit_horizon),
            },
            "rejected_weight_management": {
                "partial_trim_selected": partial_trim["selected_candidate"],
                "partial_trim_table": compact_profit_table(partial_trim),
                "score_transfer_selected": score_transfer["selected_candidate"],
                "score_transfer_table": compact_profit_table(score_transfer),
                "implementation_status": (
                    "rejected experimental hooks removed from the shared runtime; "
                    "reports retained only as evidence"
                ),
            },
            "development_vs_2025_holdout_consistency": {
                "role": holdout_consistency["role"],
                "stable_experiments": holdout_consistency[
                    "stable_experiments"
                ],
                "divergent_experiments": holdout_consistency[
                    "divergent_experiments"
                ],
                "interpretation": (
                    "portfolio width, gross exposure, entry horizon blends, latest-"
                    "session weight and pure-10d exit agree across development and "
                    "2025; smoothing and the rejected trim/transfer ideas vary by "
                    "period and remain disclosed rather than promoted"
                ),
            },
            "entry_rank_sizing": {
                "selected": entry_rank_sizing["selected_candidate"],
                "table": compact_profit_table(entry_rank_sizing),
                "selected_minus_equalweight": entry_rank_sizing[
                    "selected_minus_equalweight"
                ],
                "annual_return_delta": entry_rank_sizing[
                    "annual_return_delta_tilted_minus_equalweight"
                ],
                "neighbor_direction_support": entry_rank_sizing[
                    "neighbor_direction_support"
                ],
                "paired_block_bootstrap": entry_rank_sizing[
                    "paired_block_bootstrap"
                ],
                "path_diagnostics": entry_rank_sizing[
                    "relative_path_diagnostics"
                ],
                "invested_exposure_attribution": entry_rank_sizing[
                    "invested_exposure_attribution"
                ],
                "leave_one_year_out": rank_sizing_leave_one_year_out[
                    "leave_one_year_out"
                ],
                "beats_equalweight_after_each_year_removed": (
                    rank_sizing_leave_one_year_out[
                        "all_removed_year_cases_positive"
                    ]
                ),
                "capital_scale_and_concentration": {
                    "capital_results": rank_sizing_scale_concentration[
                        "capital_results"
                    ],
                    "rank_beats_equalweight_at_every_capital": (
                        rank_sizing_scale_concentration[
                            "rank_beats_equalweight_at_every_capital"
                        ]
                    ),
                    "paired_excess_concentration": (
                        rank_sizing_scale_concentration[
                            "paired_excess_concentration"
                        ]
                    ),
                },
                "market_state_attribution": {
                    "overall": rank_sizing_market_state["overall"],
                    "states": rank_sizing_market_state["states"],
                    "interpretation": (
                        "most rank-sizing excess is earned in the frozen strong-market "
                        "state; confirmed-weak excess remains slightly positive while "
                        "the short weak-transition state is slightly negative; this is "
                        "disclosed without adding a new state switch"
                    ),
                },
                "mechanism_attribution": {
                    "entry_target_buckets": rank_sizing_mechanism[
                        "attribution"
                    ]["bucket_summary"],
                    "same_holdings_weight_effect": rank_sizing_mechanism[
                        "attribution"
                    ]["same_holdings_weight_effect"],
                    "paired_candidate_vs_equalweight_path": (
                        rank_sizing_mechanism[
                            "paired_candidate_vs_equalweight_path_attribution"
                        ]
                    ),
                    "interpretation": (
                        "candidate and equalweight hold the same stock set on every "
                        "observed day; the net uplift is attributable to position-size "
                        "differences after trading/cash residuals, but entry target "
                        "buckets are not monotonically ordered by subsequent return"
                    ),
                },
                "maintenance_extension_test": {
                    "selected_arm": rank_sizing_maintenance["selected_arm"],
                    "baseline_delta_maintained_minus_entry_only": (
                        rank_sizing_maintenance[
                            "baseline_delta_maintained_minus_entry_only"
                        ]
                    ),
                    "stress_delta_maintained_minus_entry_only": (
                        rank_sizing_maintenance[
                            "stress_delta_maintained_minus_entry_only"
                        ]
                    ),
                    "decision": (
                        "keep new-entry-only sizing; extending the same tilt into "
                        "the existing 20-session maintenance rebalance lowers net "
                        "pre-2026 return"
                    ),
                },
                "reverse_direction_placebo": {
                    "role": rank_sizing_reverse_placebo["role"],
                    "results": rank_sizing_reverse_placebo["results"],
                    "forward_beats_reverse_at_both_costs": (
                        rank_sizing_reverse_placebo[
                            "forward_beats_reverse_at_both_costs"
                        ]
                    ),
                    "reverse_beats_equalweight_at_both_costs": (
                        rank_sizing_reverse_placebo[
                            "reverse_beats_equalweight_at_both_costs"
                        ]
                    ),
                    "interpretation": (
                        "breaking exact equalweight contributes some path benefit, "
                        "while the forward score direction adds further return over "
                        "the reverse placebo at both disclosed costs"
                    ),
                },
                "rank_neutral_dispersion_placebo": {
                    "role": rank_sizing_neutral_placebo["role"],
                    "design": rank_sizing_neutral_placebo["design"],
                    "results": rank_sizing_neutral_placebo["results"],
                    "forward_vs_neutral_paired_block_bootstrap": (
                        rank_sizing_neutral_placebo[
                            "forward_vs_neutral_paired_block_bootstrap"
                        ]
                    ),
                    "forward_beats_neutral_at_both_costs": (
                        rank_sizing_neutral_placebo[
                            "forward_beats_neutral_at_both_costs"
                        ]
                    ),
                    "neutral_beats_equalweight_at_both_costs": (
                        rank_sizing_neutral_placebo[
                            "neutral_beats_equalweight_at_both_costs"
                        ]
                    ),
                    "interpretation": (
                        "pre-2026 uplift contains both a generic modest weight-"
                        "dispersion component and a separate score-direction "
                        "component; this placebo is diagnostic only"
                    ),
                },
                "low_volatility_entry_sizing_test": {
                    "single_new_rule": low_vol_entry_sizing["single_new_rule"],
                    "volatility_availability": low_vol_entry_sizing[
                        "volatility_availability"
                    ],
                    "results": low_vol_entry_sizing["results"],
                    "selected_arm": low_vol_entry_sizing["selected_arm"],
                    "decision": (
                        "do not adopt: low-volatility sizing lowers drawdown and "
                        "turnover slightly but earns less net pre-2026 return than "
                        "the frozen global-score sizing"
                    ),
                },
                "score_distance_entry_sizing_test": {
                    "only_change": score_distance_entry_sizing["only_change"],
                    "parameter_search_count": score_distance_entry_sizing[
                        "parameter_search_count"
                    ],
                    "score_distance_minus_current": score_distance_entry_sizing[
                        "score_distance_minus_current"
                    ],
                    "multiplier_diagnostics": score_distance_entry_sizing[
                        "multiplier_diagnostics"
                    ],
                    "selection_decision": score_distance_entry_sizing[
                        "selection_decision"
                    ],
                    "decision": (
                        "do not adopt: actual score-distance sizing improves the "
                        "0.65% stress result but earns less at the normal 0.30% "
                        "cost, so the profit-first objective keeps rank sizing"
                    ),
                },
                "outside_top10_refill_haircut_test": {
                    "single_new_rule": refill_haircut["single_new_rule"],
                    "parameter_search_count": refill_haircut[
                        "parameter_search_count"
                    ],
                    "candidate_minus_current": refill_haircut[
                        "candidate_minus_current"
                    ],
                    "selection_decision": refill_haircut[
                        "selection_decision"
                    ],
                    "decision": (
                        "do not adopt: holding extra cash on outside-top10 refill "
                        "reduces normal-cost profit; its small stress improvement is "
                        "a lower-exposure effect rather than better stock selection"
                    ),
                },
                "friction_compensated_full_investment_test": {
                    "single_formula": friction_compensated_gross[
                        "single_formula"
                    ],
                    "parameter_search_count": friction_compensated_gross[
                        "parameter_search_count"
                    ],
                    "candidate_minus_current": friction_compensated_gross[
                        "candidate_minus_current"
                    ],
                    "actual_exposure_never_levered": friction_compensated_gross[
                        "actual_exposure_never_levered"
                    ],
                    "selection_decision": friction_compensated_gross[
                        "selection_decision"
                    ],
                    "decision": (
                        "do not adopt: nominally compensating buy friction lowers "
                        "normal-cost profit and slightly lowers realized investment "
                        "because earlier orders consume cash and later board-lot "
                        "orders are skipped"
                    ),
                },
                "cash_aware_batch_allocation_test": {
                    "single_new_rule": cash_aware_batch_allocation[
                        "single_new_rule"
                    ],
                    "parameter_search_count": cash_aware_batch_allocation[
                        "parameter_search_count"
                    ],
                    "candidate_minus_current": cash_aware_batch_allocation[
                        "candidate_minus_current"
                    ],
                    "selection_decision": cash_aware_batch_allocation[
                        "selection_decision"
                    ],
                    "decision": (
                        "do not adopt: batch cash allocation improves visual target "
                        "symmetry but slightly lowers profit at both disclosed costs "
                        "and does not improve realized investment"
                    ),
                },
                "rank_sizing_midpoint_check": {
                    "single_new_probe": rank_sizing_midpoint[
                        "single_new_probe"
                    ],
                    "parameter_search_count": rank_sizing_midpoint[
                        "parameter_search_count"
                    ],
                    "midpoint_minus_current": rank_sizing_midpoint[
                        "midpoint_minus_current"
                    ],
                    "selection_decision": rank_sizing_midpoint[
                        "selection_decision"
                    ],
                    "decision": (
                        "retain 1.10..0.90: the single 1.05..0.95 midpoint earns "
                        "less at both disclosed costs; together with the existing "
                        "1.20..0.80 and 1.30..0.70 controls this closes the local "
                        "tilt-amplitude neighborhood without a fine grid"
                    ),
                },
                "rank_sizing_walkforward_amplitude": {
                    "role": rank_sizing_walkforward["role"],
                    "splits": rank_sizing_walkforward["splits"],
                    "all_early_sample_choices_win_following_period": (
                        rank_sizing_walkforward[
                            "all_early_sample_choices_win_following_period"
                        ]
                    ),
                    "interpretation": rank_sizing_walkforward[
                        "interpretation"
                    ],
                },
                "rank_sizing_shape_test": {
                    "only_change": rank_sizing_shape["only_change"],
                    "selection_rule": rank_sizing_shape["selection_rule"],
                    "selected_arm": rank_sizing_shape["selected_arm"],
                    "challenger_minus_current": rank_sizing_shape[
                        "challenger_minus_current"
                    ],
                    "decision": (
                        "retain the linear 1.10..0.90 schedule: the single "
                        "top-five/bottom-five step shape earns less at both costs; "
                        "no shape grid or new rejection gate is introduced"
                    ),
                },
                "entry_rank_realization_diagnostic": {
                    "role": entry_rank_realization["role"],
                    "summary": entry_rank_realization["summary"],
                    "interpretation": (
                        "completed lifecycle returns are not monotonic in initial "
                        "score rank; do not derive a more complex rank-weight curve "
                        "from these groups. Keep the frozen linear schedule only for "
                        "its aggregate net-return edge and let 2026 validate it"
                    ),
                },
                "maintenance_topup_gate_simplification": {
                    "only_change": rank_sizing_topup_simplification["only_change"],
                    "selected_arm": rank_sizing_topup_simplification["selected_arm"],
                    "challenger_minus_current": rank_sizing_topup_simplification[
                        "challenger_minus_current"
                    ],
                    "decision": (
                        "retain the existing 0.80 maintenance top-up score rule as "
                        "an economic strategy rule, not a governance gate: removing "
                        "it lowers normal-cost net return"
                    ),
                },
                "confirmed_weak_sell_priority_simplification": {
                    "only_change": rank_sizing_sell_priority_simplification[
                        "only_change"
                    ],
                    "selected_arm": rank_sizing_sell_priority_simplification[
                        "selected_arm"
                    ],
                    "challenger_minus_current": (
                        rank_sizing_sell_priority_simplification[
                            "challenger_minus_current"
                        ]
                    ),
                    "decision": (
                        "retain the confirmed-weak volatility sell ordering because "
                        "removing it materially lowers net return at both costs; it "
                        "remains an economic strategy rule, not a governance gate"
                    ),
                },
                "pressure_trigger_simplification": {
                    "only_change": rank_sizing_pressure_trigger_simplification[
                        "only_change"
                    ],
                    "selected_arm": rank_sizing_pressure_trigger_simplification[
                        "selected_arm"
                    ],
                    "challenger_minus_current": (
                        rank_sizing_pressure_trigger_simplification[
                            "challenger_minus_current"
                        ]
                    ),
                    "decision": (
                        "retain strong4/weak5 because replacing it with uniform4 "
                        "materially lowers net return at both costs; do not expand "
                        "the branch into a trigger grid"
                    ),
                },
                "pressure_age_simplification": {
                    "only_change": rank_sizing_pressure_age_simplification[
                        "only_change"
                    ],
                    "selected_arm": rank_sizing_pressure_age_simplification[
                        "selected_arm"
                    ],
                    "challenger_minus_current": (
                        rank_sizing_pressure_age_simplification[
                            "challenger_minus_current"
                        ]
                    ),
                    "decision": (
                        "retain strong-market 4 / weak-market 10 minimum age for "
                        "the extra pressure sell because uniform4 materially lowers "
                        "net return at both costs; treat it as an economic patience "
                        "rule and do not search an age grid"
                    ),
                },
                "residual_cash_sweep": {
                    "single_new_rule": residual_cash_sweep["single_new_rule"],
                    "selected_arm": residual_cash_sweep["selected_arm"],
                    "sweep_minus_current": residual_cash_sweep[
                        "sweep_minus_current"
                    ],
                    "capital_scale": residual_cash_sweep_scale["summary"],
                    "robustness": {
                        "paired_diagnostics": residual_cash_sweep_robustness[
                            "paired_diagnostics"
                        ],
                        "paired_block_bootstrap": residual_cash_sweep_robustness[
                            "paired_block_bootstrap"
                        ],
                    },
                    "rank_interaction": {
                        "selected_arm": sweep_rank_interaction["selected_arm"],
                        "rank_minus_exact_equalweight": sweep_rank_interaction[
                            "rank_minus_exact_equalweight"
                        ],
                    },
                    "order_simplification": {
                        "selected_arm": cash_sweep_order_simplification[
                            "selected_arm"
                        ],
                        "one_order_minus_unlimited": (
                            cash_sweep_order_simplification[
                                "one_order_minus_unlimited"
                            ]
                        ),
                    },
                    "cost_frontier": cash_sweep_cost_frontier["summary"],
                    "mechanism_attribution": cash_sweep_mechanism["attribution"],
                    "trigger_simplification": {
                        "selected_arm": cash_sweep_trigger["selected_arm"],
                        "buy_trade_minus_any_trade": cash_sweep_trigger[
                            "buy_trade_minus_any_trade"
                        ],
                        "capital_and_cost_sensitivity": cash_sweep_trigger_robustness[
                            "cumulative_return_deltas"
                        ],
                    },
                    "recipient_score_ablation": {
                        "selected_arm": cash_sweep_recipient_score["selected_arm"],
                        "highest_score_minus_most_underweight": (
                            cash_sweep_recipient_score[
                                "highest_score_minus_most_underweight"
                            ]
                        ),
                        "decision": (
                            "reject repeated score chasing for residual cash because it "
                            "reduces net return at both disclosed costs"
                        ),
                    },
                    "decision": (
                        "freeze the buy-day-only residual-cash sweep with the existing "
                        "1.10-to-0.90 new-entry rank tilt: it raises primary-cost net "
                        "return at every tested capital scale and slightly reduces "
                        "turnover versus the any-trade trigger; disclose mixed high-cost "
                        "sensitivity without turning it into another rejection gate"
                    ),
                },
                "interpretation": (
                    "the 11%-to-9% new-entry rank schedule improves pre-2026 net "
                    "return at baseline and stress cost; the daily return advantage "
                    "is not explained by its small invested-ratio difference"
                ),
            },
        },
        "retained_rule_evidence": {
            "ablation": ablation["simplification_decision"],
            "paired_bootstrap_vs_simpler_weak3": bootstrap["comparisons"][
                "current_weak2_vs_simpler_weak3"
            ],
            "highest_equal_cost_beating_production": cost[
                "highest_tested_equal_cost_with_fixed10_return_above_production"
            ],
            "local_parameter_neighborhood": {
                "all_neighbors_profitable": local_stability[
                    "all_dimensions_neighbors_profitable"
                ],
                "all_neighbors_all_years_positive": local_stability[
                    "all_dimensions_neighbors_have_positive_calendar_years"
                ],
                "selected_values_are_interior": local_stability[
                    "selected_values_are_interior_of_tested_neighborhood"
                ],
                "leave_one_year_out": leave_one_year_out["dimensions"],
                "current_core_parameters": core_neighborhood["leave_one_year_out"],
                "all_current_values_top2_in_every_fold": core_neighborhood[
                    "all_current_values_top2_in_every_fold"
                ],
            },
            "volatility_priority_increment": {
                "candidate_minus_no_priority": vol_priority_bootstrap[
                    "candidate_minus_reference"
                ],
                "paired_block_bootstrap": vol_priority_bootstrap[
                    "paired_block_bootstrap"
                ],
                "decision": vol_priority_bootstrap["decision"],
                "action_concentration": vol_priority_action_concentration[
                    "comparisons"
                ]["penalty_0.050_vs_0"],
                "nearby_005_vs_0075": vol_priority_action_concentration[
                    "comparisons"
                ]["penalty_0.050_vs_0.075"],
                "action_concentration_decision": (
                    vol_priority_action_concentration["decision"]
                ),
                "binary_simplification": {
                    "selected_candidate": vol_priority_binary[
                        "selected_candidate"
                    ],
                    "gates": vol_priority_binary["simplification_gates"],
                    "decision": vol_priority_binary["decision"],
                },
            },
        },
        "risk_summary": {
            "prevalidation_classification": prevalidation_risk[
                "pre2026_classification"
            ],
            "prevalidation_decision": prevalidation_risk["decision"],
            "time_breadth": prevalidation_risk["time_breadth"],
            "paired_block_bootstrap_vs_production": prevalidation_risk[
                "paired_block_bootstrap"
            ],
            "full_investment_tradeoff": {
                "return_edge_attribution": full_investment_tradeoff[
                    "return_edge_attribution"
                ],
                "drawdown_structure": full_investment_tradeoff[
                    "drawdown_structure"
                ],
                "constraint_compatibility": full_investment_tradeoff[
                    "constraint_compatibility"
                ],
                "decision": full_investment_tradeoff["decision"],
            },
            "rule_generalization": {
                "classifications": {
                    name: item["classification"]
                    for name, item in rule_generalization["rules"].items()
                },
                "unsupported_rules": rule_generalization["unsupported_rules"],
                "decision": rule_generalization["decision"],
            },
            "quarterly_outperformance_frequency": rolling[
                "quarterly_outperformance_frequency"
            ],
            "annual_outperformance_frequency": rolling[
                "annual_outperformance_frequency"
            ],
            "rolling_252d_outperformance_frequency": rolling[
                "rolling_windows"
            ]["252"]["fixed10_outperformance_frequency"],
            "top1_stock_share_of_positive_mark_pnl": concentration[
                "fixed10_stock_mark_concentration"
            ]["top1_share_of_positive_mark_pnl"],
            "top10_stock_share_of_positive_mark_pnl": concentration[
                "fixed10_stock_mark_concentration"
            ]["top10_share_of_positive_mark_pnl"],
            "monthly_return_concentration": {
                "fixed10_positive_month_hhi": concentration[
                    "fixed10_monthly_concentration"
                ]["positive_month_return_hhi"],
                "production_positive_month_hhi": concentration[
                    "production_monthly_concentration"
                ]["positive_month_return_hhi"],
                "fixed10_top3_positive_month_share": concentration[
                    "fixed10_monthly_concentration"
                ]["without_top_positive_months"]["3"][
                    "top_months_share_of_positive_simple_returns"
                ],
                "production_top3_positive_month_share": concentration[
                    "production_monthly_concentration"
                ]["without_top_positive_months"]["3"][
                    "top_months_share_of_positive_simple_returns"
                ],
            },
            "cost_start_cross_robustness": cost_start_cross["summary"],
            "current_candidate_vs_production_drawdown": {
                "candidate_episode": current_candidate_drawdown[
                    "candidate_maximum_drawdown"
                ],
                "production_episode": current_candidate_drawdown[
                    "production_maximum_drawdown"
                ],
                "candidate_worst_windows": current_candidate_drawdown[
                    "candidate_worst_windows"
                ],
                "production_worst_windows": current_candidate_drawdown[
                    "production_worst_windows"
                ],
                "worst_relative_windows": current_candidate_drawdown[
                    "worst_relative_windows"
                ],
            },
            "current_candidate_vs_production_profit_breadth": {
                "paired_diagnostics": current_candidate_vs_production[
                    "paired_diagnostics"
                ],
                "paired_block_bootstrap": current_candidate_vs_production[
                    "paired_block_bootstrap"
                ],
                "interpretation": (
                    "development return is higher but concentrated enough that one "
                    "independent 2026 validation remains necessary; this disclosure "
                    "does not add a new rejection gate or invite post-hoc repair"
                ),
            },
            "maximum_drawdown_position_age": {
                "episode": drawdown_position_age["maximum_drawdown_episode"],
                "age_bands": drawdown_position_age["position_age_attribution"][
                    "age_bands"
                ],
                "diagnosis": drawdown_position_age["diagnosis"],
                "rule_implication": drawdown_position_age["rule_implication"],
            },
            "maximum_drawdown_score_state": {
                "episode": drawdown_score_state["maximum_drawdown_episode"],
                "support": drawdown_score_state["support"],
                "supported_bounded_rule_tests": drawdown_score_state[
                    "supported_bounded_rule_tests"
                ],
                "decision": drawdown_score_state["decision"],
            },
            "turnover_and_replacement_quality": {
                "source_attribution": turnover_source["turnover_attribution"],
                "replacement_counterfactual_overall": replacement_counterfactual[
                    "overall"
                ],
                "replacement_counterfactual_by_year": replacement_counterfactual[
                    "by_year"
                ],
                "replacement_by_market_state": replacement_failure[
                    "by_market_state"
                ],
                "two_day_exit_confirmation_recheck": {
                    "candidate_deltas": exit_confirmation["candidate_deltas"],
                    "candidate_gates": exit_confirmation["candidate_gates"],
                    "selected_candidate": exit_confirmation["selected_candidate"],
                },
                "decision": (
                    "score-driven replacement contributes 98% of turnover and has "
                    "positive mean 5-day replacement excess in every pre-2026 year; "
                    "equalweight maintenance contributes less than 2%, while a two-day "
                    "exit confirmation sharply reduces return and worsens drawdown; "
                    "retain the immediate score exit and 20-session maintenance rules "
                    "rather than searching for a cosmetic turnover reduction"
                ),
            },
            "small_noise_rank_and_return": {
                "rank_effect": rank_noise["noise_rank_effect"],
                "return_effect": small_noise["sigma_results"],
                "margin_vs_production": noise_production_margin["comparisons"],
            },
            "rank_membership_vs_internal_order": rank_attribution[
                "deltas_vs_baseline"
            ],
            "executed_action_rank_profile": {
                "replacement_buys": action_rank["replacement_buys"],
                "sells": action_rank["sells"],
            },
            "decision_staleness": {
                "one_session_attribution": staleness["robustness_decision"][
                    "one_session_attribution"
                ],
                "one_session_delay_all_years_positive": staleness[
                    "robustness_decision"
                ]["one_session_delay_all_years_positive"],
                "two_session_delay_all_years_positive": staleness[
                    "robustness_decision"
                ]["two_session_delay_all_years_positive"],
                "interpretation": (
                    "the candidate remains profitable by calendar year but exact score "
                    "freshness is operationally material"
                ),
            },
            "exit_timing_tradeoff": {
                "selection": exit_timing["selection"],
                "comparison": exit_timing["comparison"],
            },
            "event_entry_block_robustness": event_block_robustness["summary"],
            "score_precision": {
                "operational_decision": score_precision["operational_decision"],
                "round_5dp": score_precision["summary"]["round_5dp"],
                "round_4dp": score_precision["summary"]["round_4dp"],
                "low_volatility_tiebreak": {
                    "selected_candidate": precision_volatility_tiebreak[
                        "selected_candidate"
                    ],
                    "gates": precision_volatility_tiebreak["candidate_gates"],
                    "order_diagnostics": precision_volatility_tiebreak[
                        "order_diagnostics"
                    ],
                },
            },
            "weak_market_size_guard": {
                "selected_candidate": weak_size_guard["selected_candidate"],
                "changed_from_checkpoint": weak_size_guard[
                    "changed_from_checkpoint"
                ],
            },
            "entry_open_gap_guard": {
                "selected_candidate": open_gap_guard["selected_candidate"],
                "changed_from_checkpoint": open_gap_guard[
                    "changed_from_checkpoint"
                ],
            },
            "limitup_exit": {
                "selected_candidate": limitup_exit["selected_candidate"],
                "changed_from_checkpoint": limitup_exit[
                    "changed_from_checkpoint"
                ],
                "defer_minus_sell_as_planned": limitup_exit[
                    "candidate_deltas"
                ],
            },
            "confirmed_weak_microcap_sell_priority": {
                "selected_candidate": vol_size_priority["selected_candidate"],
                "changed_from_checkpoint": vol_size_priority[
                    "changed_from_checkpoint"
                ],
                "selection_gates": vol_size_priority["selection_gates"],
            },
            "candidate_frontier": {
                "unique_metric_vectors": candidate_frontier[
                    "unique_metric_vectors"
                ],
                "pareto_frontier_size": candidate_frontier[
                    "pareto_frontier_size"
                ],
                "current_candidate": candidate_frontier["current_candidate"],
                "production_baseline": candidate_frontier[
                    "production_baseline"
                ],
            },
            "current_vs_production_exposure_attribution": {
                "exposure_bins": exposure_attribution["exposure_bins"],
                "total_log_excess": exposure_attribution[
                    "candidate_minus_production_total_log_return"
                ],
                "low_and_mid_exposure_log_excess": exposure_attribution[
                    "candidate_minus_production_low_and_mid_exposure_log_return"
                ],
                "high_exposure_log_excess": exposure_attribution[
                    "candidate_minus_production_high_exposure_log_return"
                ],
                "structural_diagnosis": exposure_attribution[
                    "structural_diagnosis"
                ],
            },
            "joint_parameter_perturbation": {
                "design": joint_perturbation["design"],
                "summary": joint_perturbation["robustness_summary"],
                "interpretation": (
                    "simultaneously moving four parameters to nearby corners remains "
                    "profitable in every year but fails to beat production; interaction "
                    "fragility is retained as a validation risk, not used for selection"
                ),
            },
            "minimum_hold_4_fragility": {
                "comparisons": min_hold_fragility["comparisons"],
                "diagnosis": min_hold_fragility["diagnosis"],
                "interpretation": (
                    "four sessions is a sharp discrete optimum, but its advantage over "
                    "three and five sessions is positive in every pre-2026 year, survives "
                    "removal of the five best days, and has broad paired-bootstrap support"
                ),
            },
            "core_parameter_support": {
                "summary": core_parameter_support["summary"],
                "interpretation": core_parameter_support["interpretation"],
            },
            "weak_confirmation_support": {
                "diagnosis": weak_confirmation_support["diagnosis"],
                "comparisons": {
                    key: {
                        "action_difference": value["action_difference"],
                        "concentration": value[
                            "daily_log_excess_concentration"
                        ],
                        "support": value["support"],
                    }
                    for key, value in weak_confirmation_support[
                        "comparisons"
                    ].items()
                },
            },
            "rule_ablation_support": {
                "summary": rule_ablation_support["summary"],
                "simplification_candidates": rule_ablation_support[
                    "simplification_candidates"
                ],
                "maintenance_gate_action_difference": rule_ablation_support[
                    "comparisons"
                ]["remove_maintenance_score_gate"]["action_difference"],
                "interpretation": (
                    "the maintenance top-up score gate changed only one action and is a "
                    "leading final-freeze simplification candidate; weak extra-age "
                    "slowdown remains a risk-motivated rule despite concentrated evidence"
                ),
            },
            "capital_scale_robustness": capital_scale["summary"],
            "market_state_branch_support": {
                "state_semantics": market_state_branch["state_semantics"],
                "state_day_counts": market_state_branch["state_day_counts"],
                "diagnosis": market_state_branch["diagnosis"],
                "comparisons": market_state_branch["comparisons"],
                "interpretation": (
                    "the strong4/weak5 branch is broadly supported against uniform6, "
                    "but not against the adjacent uniform4 and uniform5 rules; retain "
                    "it only as an aggressive candidate with explicit validation risk"
                ),
            },
            "state_simplification_stack": {
                "ordered_simplifications": state_simplification[
                    "ordered_simplifications"
                ],
                "runs": state_simplification["runs"],
                "transitions": state_simplification["transitions"],
                "fully_simplified_vs_production": state_simplification[
                    "fully_simplified_vs_production"
                ],
                "interpretation": (
                    "removing every state-dependent branch makes the fixed10 strategy "
                    "simpler but weaker than production; state-conditional volatility "
                    "ordering remains economically material"
                ),
            },
        },
        "rejected_or_inactive_directions": rejected,
        "event_rule_decision": {
            "contract_id": event_overlay["contract_id"],
            "frozen_rules": event_overlay["rules"],
            "authoritative_strategy_decision": minimal_event_policy[
                "authoritative_strategy_decision"
            ],
            "minimal_diagnostic_policy": minimal_event_policy[
                "minimal_diagnostic_policy"
            ],
            "score_tiebreak_diagnostic": event_score_tiebreak[
                "recommended_predeclared_diagnostic"
            ],
            "score_tiebreak_proxy_deltas": event_score_tiebreak["action_audit"],
            "score_tiebreak_candidate_gates": event_score_tiebreak[
                "candidate_gates"
            ],
            "proxy_direct_interventions": minimal_event_policy[
                "direct_interventions"
            ],
            "ordinary_abnormal_volatility": "observe_only; no trading action",
            "severe_abnormal_volatility": (
                "diagnostic-only: use the existing 0.05 replacement margin as a "
                "new-entry score tiebreak on the first visible session; never change "
                "eligibility or force a sale"
            ),
            "exchange_focus_security": (
                "observe_only; proxy entry tiebreak reduced return and Sharpe"
            ),
            "existing_holdings": "retain original score exit; no event-forced sell",
            "development_evidence_reason": (
                "official production event history starts in 2026, so pre-2026 tuning "
                "cannot estimate these effects without contaminating validation"
            ),
            "one_shot_validation_arms": event_overlay["one_shot_validation_arms"],
            "decision_order": event_overlay["decision_order"],
            "runtime_implementation": {
                "shared_source": str(
                    REPO / "quant/main/research_v260_risk_event_overlay.py"
                ).replace("\\", "/"),
                "ordinary_severe_alert_use_one_mask_builder": True,
                "frozen_contract_checked_before_validation_read": True,
                "event_asset_hash_checked_before_and_after_read": True,
                "pit_future_row_poison_prefix_invariance": True,
                "pre2026_execution_timing_audit": execution_timing["audit"],
                "pre2026_portfolio_invariant_audit": execution_timing[
                    "portfolio_audit"
                ],
                "pre2026_pairwise_rule_interaction_audit": {
                    "pair_count": pairwise_rules["pair_count"],
                    "dominating_pair_removals": pairwise_rules[
                        "dominating_pair_removals"
                    ],
                    "decision": pairwise_rules["decision"],
                },
                "validation_executable_code_binding": validation_protocol[
                    "source_bindings"
                ]["executable_code"],
                "fixed10_regression_tests_passed": 602,
            },
        },
        "simplification_decision": {
            "maintenance_topup_score": (
                "do not expose a separate tunable parameter; reuse the 0.80 "
                "max-hold renewal threshold"
            ),
            "evidence": {
                "threshold_075_080_085_are_execution_equivalent": all(
                    core_neighborhood["results"]["maintenance_topup_score"][value]
                    == core_neighborhood["results"]["maintenance_topup_score"]["0.8000"]
                    for value in ("0.7500", "0.8500")
                ),
                "removing_gate_dominates": ablation["ablations"][
                    "remove_maintenance_score_gate"
                ]["removal_dominates_full_current"],
            },
        },
        "pre2026_search_saturation": {
            "tested_rule_families": search_saturation["tested_rule_families"],
            "gates": search_saturation["saturation_gates"],
            "remaining_selection_candidates": search_saturation[
                "remaining_pre2026_selection_candidates"
            ],
            "decision": search_saturation["search_decision"],
            "known_validation_risks": search_saturation["known_validation_risks"],
        },
        "pre2026_freeze_integrity": {
            "status": freeze_integrity["status"],
            "candidate_checkpoint_sha256": freeze_integrity[
                "candidate_checkpoint_sha256"
            ],
            "protocol_sha256": freeze_integrity["protocol_sha256"],
            "executable_code_bundle_sha256": freeze_integrity[
                "executable_code_bundle_sha256"
            ],
            "executable_code_file_count": freeze_integrity[
                "executable_code_file_count"
            ],
            "validation_asset_metadata": freeze_integrity[
                "validation_asset_metadata"
            ],
            "replay_evidence": freeze_integrity["replay_evidence"],
            "gates": freeze_integrity["gates"],
        },
        "goal_completion_readiness": {
            "status": goal_readiness["status"],
            "elapsed_seconds": goal_readiness["elapsed_seconds"],
            "required_seconds": goal_readiness["required_seconds"],
            "requirements": goal_readiness["requirements"],
            "incomplete_requirements": goal_readiness[
                "incomplete_requirements"
            ],
            "next_action": goal_readiness["next_action"],
        },
        "next_gate": (
            "continue pre-2026 robustness work until the 24-hour objective is met, "
            "then open 2026 exactly once for validation"
        ),
        "one_shot_validation_protocol": {
            "protocol_id": validation_protocol["protocol_id"],
            "validation_boundary": validation_protocol["validation_boundary"],
            "arms": validation_protocol["arms"],
            "base_candidate_decision": validation_protocol[
                "base_candidate_decision"
            ],
            "runtime_contract": validation_protocol["runtime_contract"],
            "event_overlay_decision": validation_protocol[
                "event_overlay_decision"
            ],
            "runner": {
                "path": str(
                    REPO
                    / "quant/main/"
                    "research_v260_fixed10_one_shot_2026_validation_runner_20260822.py"
                ).replace("\\", "/"),
                "default_mode": "build_preflight_only_2026_not_opened",
                "explicit_confirmation_required": True,
                "build_preflight_passed": True,
                "protocol_derivation_exact_match_required": True,
                "decision_thresholds_consumed_from_protocol": True,
                "terminal_failure_record_required": True,
                "pit_future_row_poison_prefix_invariance": True,
                "pre2026_execution_timing_audit_passed": execution_timing["audit"][
                    "all_gates_passed"
                ],
                "pre2026_portfolio_invariant_audit_passed": execution_timing[
                    "portfolio_audit"
                ]["all_gates_passed"],
                "pre2026_pairwise_rule_interaction_audit_passed": not pairwise_rules[
                    "dominating_pair_removals"
                ],
                "pre2026_cost_start_cross_robustness_passed": cost_start_cross[
                    "summary"
                ]["all_cumulative_returns_positive"],
                "executable_code_file_count": validation_protocol["source_bindings"][
                    "executable_code"
                ]["file_count"],
                "executable_code_bundle_sha256": validation_protocol[
                    "source_bindings"
                ]["executable_code"]["bundle_sha256"],
                "fixed10_regression_tests_passed": 602,
                "readonly_result_auditor": str(
                    REPO
                    / "quant/main/"
                    "research_v260_fixed10_one_shot_2026_result_audit_20260822.py"
                ).replace("\\", "/"),
            },
        },
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "optimization_ledger.json", result)
    print(json.dumps({
        "status": result["status"],
        "candidate_id": result["current_candidate"]["candidate_id"],
        "rejected_or_inactive": list(rejected),
        "next_gate": result["next_gate"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
