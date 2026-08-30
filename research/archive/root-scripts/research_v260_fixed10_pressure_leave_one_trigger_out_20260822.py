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

import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_portfolio_rebalance_cadence_20260822 as cadence
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness
import research_v260_lowrisk_score_tuning_20260821 as research_base
from research_v260_runtime import fixed10_risk_event_v110 as runtime


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_pressure_leave_one_trigger_out_20260822"
)
CHECKPOINT_PATH = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_position_pressure_exit_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
ROBUSTNESS_PATH = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_position_pressure_exit_robustness_20260822/"
    "development_result.json"
)


def leave_one_out_schedules(
    dates: np.ndarray,
    trigger_dates: list[str],
    full_limit: int = 2,
    suppressed_limit: int = 1,
) -> dict[str, np.ndarray]:
    values = np.asarray(dates).astype(str)
    if len(set(trigger_dates)) != len(trigger_dates):
        raise ValueError("trigger dates must be unique")
    date_index = {value: index for index, value in enumerate(values)}
    missing = sorted(set(trigger_dates).difference(date_index))
    if missing:
        raise ValueError(f"trigger dates missing from development calendar: {missing}")
    schedules = {}
    for trigger_date in trigger_dates:
        schedule = np.full(len(values), int(full_limit), dtype=np.int16)
        schedule[date_index[trigger_date]] = int(suppressed_limit)
        schedules[trigger_date] = schedule
    return schedules


def run_schedule(context, policy: dict, schedule: np.ndarray):
    threshold = policy.get("maintenance_topup_requires_score")
    maintenance_block = (
        context.empty_block
        if threshold is None
        else ~np.isfinite(context.score) | (context.score < float(threshold))
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
        score_sell_pressure_limit_override=schedule,
        score_sell_pressure_confirmation_days_override=policy.get(
            "score_sell_pressure_confirmation_days", 1
        ),
    )
    metrics = round1.evaluate_run(
        daily, actions, research_base.FIRST_BUY, round1.DEVELOPMENT_END
    )
    return metrics, daily, actions


def concentration_summary(full_metrics: dict, variants: dict[str, dict]) -> dict:
    effects = {
        date: float(full_metrics["cumulative_return"] - item["cumulative_return"])
        for date, item in variants.items()
    }
    positive = {date: value for date, value in effects.items() if value > 0.0}
    total_positive = float(sum(positive.values()))
    largest_date = max(effects, key=lambda key: abs(effects[key]))
    return {
        "trigger_count": len(effects),
        "beneficial_trigger_count": len(positive),
        "harmful_trigger_count": sum(value < 0.0 for value in effects.values()),
        "zero_effect_trigger_count": sum(value == 0.0 for value in effects.values()),
        "median_cumulative_effect": float(np.median(list(effects.values()))),
        "largest_absolute_effect_date": largest_date,
        "largest_absolute_effect": float(effects[largest_date]),
        "largest_positive_effect_share": (
            float(max(positive.values()) / total_positive)
            if total_positive > 0.0 else 0.0
        ),
        "effects": effects,
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 validation is not available for sensitivity analysis")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)
    robustness = json.loads(ROBUSTNESS_PATH.read_text(encoding="utf-8"))
    records = robustness["results"]["control_trigger4_limit2"][
        "pressure_diagnostics"
    ]["records"]
    trigger_dates = [str(item["signal_date"]) for item in records]
    schedules = leave_one_out_schedules(context.arrays["dates"], trigger_dates)

    full_schedule = np.full(
        len(context.arrays["dates"]),
        int(policy["score_sell_pressure_limit"]),
        dtype=np.int16,
    )
    full_metrics, full_daily, full_actions = run_schedule(
        context, policy, full_schedule
    )
    expected = checkpoint["current_best_equalweight"]
    checkpoint_equivalence = {
        key: bool(np.isclose(full_metrics[key], expected[key], rtol=0.0, atol=1e-12))
        for key in (
            "cumulative_return", "cagr", "sharpe", "max_drawdown",
            "turnover_annualized", "average_invested_ratio",
        )
    }
    if not all(checkpoint_equivalence.values()):
        raise RuntimeError("leave-one-trigger-out baseline drifted")

    variants = {}
    for trigger_date, schedule in schedules.items():
        metrics, _, _ = run_schedule(context, policy, schedule)
        variants[trigger_date] = metrics
    summary = concentration_summary(full_metrics, variants)
    repeat_metrics, repeat_daily, repeat_actions = run_schedule(
        context, policy, full_schedule
    )
    deterministic = {
        "daily": round1.frame_hash(full_daily) == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(full_actions) == round1.frame_hash(repeat_actions),
        "metrics": all(
            full_metrics[key] == repeat_metrics[key]
            for key in ("cumulative_return", "cagr", "sharpe", "max_drawdown")
        ),
    }
    if not all(deterministic.values()):
        raise RuntimeError("leave-one-trigger-out replay failed")

    supported = bool(
        summary["largest_positive_effect_share"] <= 0.35
        and len({date[:4] for date in trigger_dates}) >= 3
        and summary["trigger_count"] >= 20
    )
    result = {
        "status": "pressure_leave_one_trigger_out_complete_2026_not_opened",
        "source_strategy": context.rules["strategy_id"],
        "development_boundary": [research_base.FIRST_BUY, round1.DEVELOPMENT_END],
        "rule_under_test": {
            "score_sell_pressure_trigger": policy["score_sell_pressure_trigger"],
            "normal_daily_score_sell_limit": policy["max_daily_score_sells"],
            "pressure_daily_score_sell_limit": policy["score_sell_pressure_limit"],
        },
        "method": (
            "Suppress the second score exit on exactly one observed pressure trigger "
            "date at a time, while leaving every other rule and date unchanged."
        ),
        "full_metrics_0_30pct": full_metrics,
        "leave_one_out_metrics_0_30pct": variants,
        "concentration": summary,
        "supported": supported,
        "checkpoint_equivalence": checkpoint_equivalence,
        "deterministic_replay": deterministic,
        "data_access": context.access,
        "source_manifests": context.manifests,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "sensitivity.json", result)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
