from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_portfolio_rebalance_cadence_20260822 as cadence
import research_v260_lowrisk_score_tuning_20260821 as research_base
from research_v260_runtime import fixed10_risk_event_v110 as runtime


@dataclass
class PressureContext:
    harness: object
    protocol: dict
    definition: dict
    rules: dict
    manifests: dict
    arrays: dict
    access: dict
    score: np.ndarray
    order: np.ndarray
    active: np.ndarray
    maintenance_block: np.ndarray
    empty_block: np.ndarray


def load_context(policy: dict) -> PressureContext:
    harness, protocol, rules, manifests, arrays, access = round1.load_arrays(
        round1.DEVELOPMENT_END
    )
    if access["logical_max_date"] != round1.DEVELOPMENT_END:
        raise PermissionError("2026 data entered pressure recheck harness")
    score, order = harness.v95.score_pair(arrays, 0.0, 7, 0.1)
    return PressureContext(
        harness=harness,
        protocol=protocol,
        definition=harness.production_definition(protocol),
        rules=rules,
        manifests=manifests,
        arrays=arrays,
        access=access,
        score=score,
        order=order,
        active=cadence.rebalance_schedule(
            len(arrays["dates"]), policy["portfolio_rebalance_interval_days"]
        ),
        maintenance_block=np.zeros(score.shape, dtype=np.bool_),
        empty_block=np.zeros(score.shape, dtype=np.bool_),
    )


def run_policy(context: PressureContext, policy: dict) -> tuple[dict, object, object]:
    maintenance_threshold = policy.get("maintenance_topup_requires_score")
    maintenance_block = (
        context.empty_block
        if maintenance_threshold is None
        else ~np.isfinite(context.score)
        | (context.score < float(maintenance_threshold))
    )
    common = dict(
        simulator=runtime.simulate,
        portfolio_rebalance_active_override=cadence.rebalance_schedule(
            len(context.arrays["dates"]),
            int(policy["portfolio_rebalance_interval_days"]),
        ),
        portfolio_rebalance_min_weight_deviation_override=policy[
            "portfolio_rebalance_min_weight_deviation"
        ],
        entry_block_mask_override=context.empty_block,
        maintenance_buy_block_mask_override=maintenance_block,
        score_sell_pressure_trigger_override=policy["score_sell_pressure_trigger"],
        score_sell_pressure_limit_override=policy["score_sell_pressure_limit"],
        score_sell_pressure_confirmation_days_override=policy.get(
            "score_sell_pressure_confirmation_days", 1
        ),
        pairwise_replacement_guard_override=policy.get(
            "pairwise_replacement_guard", False
        ),
    )
    daily, actions = round1.run_fixed10(
        context.harness,
        context.arrays,
        context.protocol,
        context.definition,
        context.score,
        context.order,
        policy,
        round1.DEVELOPMENT_END,
        slip=round1.BASELINE_COST,
        record_actions=True,
        **common,
    )
    stress_daily, stress_actions = round1.run_fixed10(
        context.harness,
        context.arrays,
        context.protocol,
        context.definition,
        context.score,
        context.order,
        policy,
        round1.DEVELOPMENT_END,
        slip=round1.STRESS_COST,
        record_actions=True,
        **common,
    )
    result = {
        "policy": copy.deepcopy(policy),
        "metrics_0_30pct": round1.evaluate_run(
            daily, actions, research_base.FIRST_BUY, round1.DEVELOPMENT_END
        ),
        "train_2022_2024": round1.evaluate_run(
            daily, actions, research_base.FIRST_BUY, "20241231"
        ),
        "holdout_2025": round1.evaluate_run(
            daily, actions, "20250102", round1.DEVELOPMENT_END
        ),
        "metrics_0_65pct": round1.evaluate_run(
            stress_daily,
            stress_actions,
            research_base.FIRST_BUY,
            round1.DEVELOPMENT_END,
        ),
    }
    return result, daily, actions


def training_key(item: dict) -> tuple:
    train = item["train_2022_2024"]
    return (
        train["sharpe"],
        train["cagr"],
        -train["max_drawdown"],
        -train["turnover_annualized"],
    )


def confirmation_gates(candidate: dict, baseline: dict) -> dict[str, bool]:
    return {
        "forward_2025_not_worse": candidate["holdout_2025"]["cumulative_return"]
        >= baseline["holdout_2025"]["cumulative_return"],
        "stress_not_worse": candidate["metrics_0_65pct"]["cumulative_return"]
        >= baseline["metrics_0_65pct"]["cumulative_return"],
        "full_10_positions": candidate["metrics_0_30pct"]["full_10_position_ratio"]
        == 1.0,
        "all_calendar_years_positive": min(
            candidate["metrics_0_30pct"]["annual_returns"].values()
        )
        > 0.0,
    }
