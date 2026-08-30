from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
sys.path.insert(0, str(MAIN_ROOT))

import research_v260_lowrisk_score_tuning_20260821 as research_base


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/strategy_agent_v260_fixed10_hold_optimization_20260822"
)

DEVELOPMENT_END = "20251231"
VALIDATION_START = "20260105"
VALIDATION_END = "20260820"
BASELINE_COST = 0.003
STRESS_COST = 0.0065


HOLD_POLICIES = {
    "production_exit_fixed10": {
        "min_hold_mode": "production_dynamic",
        "max_hold_days": 20,
        "sell_score_below": 0.85,
        "replacement_advantage": 0.05,
        "sell_confirmation_days": 1,
        "renewal_policy": "none",
        "age_bands": None,
        "max_daily_score_sells": None,
    },
    "renew_strong_at_20": {
        "min_hold_mode": "production_dynamic",
        "max_hold_days": 20,
        "sell_score_below": 0.85,
        "replacement_advantage": 0.05,
        "sell_confirmation_days": 1,
        "renewal_policy": "no_score_exit",
        "age_bands": None,
        "max_daily_score_sells": None,
    },
    "renew_strong_at_30": {
        "min_hold_mode": "production_dynamic",
        "max_hold_days": 30,
        "sell_score_below": 0.85,
        "replacement_advantage": 0.05,
        "sell_confirmation_days": 1,
        "renewal_policy": "no_score_exit",
        "age_bands": None,
        "max_daily_score_sells": None,
    },
    "renew_strong_confirm_2d": {
        "min_hold_mode": "production_dynamic",
        "max_hold_days": 30,
        "sell_score_below": 0.85,
        "replacement_advantage": 0.05,
        "sell_confirmation_days": 2,
        "renewal_policy": "no_score_exit",
        "age_bands": None,
        "max_daily_score_sells": None,
    },
    "age_relaxed_replacement": {
        "min_hold_mode": "production_dynamic",
        "max_hold_days": 30,
        "sell_score_below": 0.85,
        "replacement_advantage": 0.10,
        "sell_confirmation_days": 1,
        "renewal_policy": "no_score_exit",
        "age_bands": [[0, 0.10], [10, 0.05], [20, 0.00]],
        "max_daily_score_sells": None,
    },
    "one_weak_exit_per_day": {
        "min_hold_mode": "production_dynamic",
        "max_hold_days": 30,
        "sell_score_below": 0.85,
        "replacement_advantage": 0.05,
        "sell_confirmation_days": 1,
        "renewal_policy": "no_score_exit",
        "age_bands": None,
        "max_daily_score_sells": 1,
    },
}


def frame_hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256(
        frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def load_arrays(end_date: str):
    harness = research_base.load_harness()
    harness.END_DATE = str(end_date)
    protocol, rules, manifests = harness.validate_control_plane()
    arrays, access = harness.build_development_arrays()
    if str(access["logical_max_date"]) != str(end_date):
        raise RuntimeError(f"unexpected data endpoint: {access['logical_max_date']}")
    return harness, protocol, rules, manifests, arrays, access


def normalize_daily_float_override(value, length: int, name: str):
    if value is None:
        return None
    normalized = (
        np.full(int(length), float(value), dtype=np.float64)
        if np.ndim(value) == 0
        else np.asarray(value, dtype=np.float64)
    )
    if normalized.shape != (int(length),):
        raise ValueError(f"{name} must match the trading calendar")
    return normalized


def run_fixed10(
    harness,
    arrays: dict[str, np.ndarray],
    protocol: dict,
    production_definition: dict,
    score: np.ndarray,
    order: np.ndarray,
    policy: dict,
    end_date: str,
    *,
    slip: float,
    record_actions: bool = False,
    simulator=None,
    min_hold_days_override=None,
    sell_score_below_override=None,
    replacement_advantage_override=None,
    sell_score_below_age_bands_override=None,
    max_hold_renewal_score_override=None,
    max_hold_renewal_rank_override=None,
    max_daily_score_sells_override=None,
    score_sell_pressure_trigger_override=None,
    score_sell_pressure_limit_override=None,
    score_sell_pressure_confirmation_days_override=None,
    score_sell_pressure_cooldown_days_override=None,
    score_sell_pressure_min_replacement_score_override=None,
    score_sell_pressure_extra_min_age_override=None,
    score_sell_priority_override=None,
    score_sell_pressure_observer=None,
    pairwise_replacement_guard_override=None,
    portfolio_rebalance_active_override=None,
    portfolio_rebalance_min_weight_deviation_override=None,
    portfolio_rebalance_overweight_deviation_override=None,
    portfolio_rebalance_underweight_deviation_override=None,
    entry_block_mask_override=None,
    maintenance_buy_block_mask_override=None,
    score_exit_confirmation_mask_override=None,
    exit_score_override=None,
    selection_mask_override=None,
    max_entry_open_gap_override=None,
    min_entry_open_gap_override=None,
    entry_price_loss_exit_override=None,
    defer_sell_on_limit_up_override=None,
    absolute_max_hold_days_override=None,
    position_observer=None,
    target_positions_override: int = 10,
    target_gross_override: float = 1.0,
):
    case_protocol = copy.deepcopy(protocol)
    case_protocol["execution"]["fixed_slippage_ratio"] = float(slip)
    definition = {
        **production_definition,
        "max_hold_days": int(policy["max_hold_days"]),
        "sell_score_below": float(policy["sell_score_below"]),
        "replacement_advantage": float(policy["replacement_advantage"]),
    }
    length = len(arrays["dates"])
    target_positions = int(target_positions_override)
    if target_positions <= 0:
        raise ValueError("target_positions_override must be positive")
    target_gross = float(target_gross_override)
    if not 0.0 < target_gross <= 1.0:
        raise ValueError("target_gross_override must be in (0, 1]")
    target_weight = target_gross / float(target_positions)
    sell_score_schedule = normalize_daily_float_override(
        sell_score_below_override, length, "sell_score_below_override"
    )
    min_hold = harness.v162.min_hold_schedule(arrays, definition["min_hold_policy"])
    if policy["min_hold_mode"] != "production_dynamic":
        min_hold[:] = int(policy["min_hold_mode"])
    if min_hold_days_override is not None:
        min_hold = (
            np.full(length, int(min_hold_days_override), dtype=np.int16)
            if np.ndim(min_hold_days_override) == 0
            else np.asarray(min_hold_days_override, dtype=np.int16)
        )
        if min_hold.shape != (length,):
            raise ValueError("min_hold_days_override must match the trading calendar")
    age_bands = policy.get("age_bands")
    simulation_function = harness.v109.simulate if simulator is None else simulator
    simulator_kwargs = {}
    if policy.get("score_peak_drop_exit") is not None:
        simulator_kwargs["score_peak_drop_exit_override"] = float(
            policy["score_peak_drop_exit"]
        )
    if sell_score_below_age_bands_override is not None:
        simulator_kwargs["sell_score_below_age_bands_override"] = [
            (int(age), float(value))
            for age, value in sell_score_below_age_bands_override
        ]
    if replacement_advantage_override is not None:
        simulator_kwargs["replacement_advantage_override"] = (
            float(replacement_advantage_override)
            if np.ndim(replacement_advantage_override) == 0
            else np.asarray(replacement_advantage_override, dtype=np.float64)
        )
    if policy.get("price_peak_drawdown_exit") is not None:
        simulator_kwargs["price_peak_drawdown_exit_override"] = float(
            policy["price_peak_drawdown_exit"]
        )
    if max_hold_renewal_score_override is not None:
        simulator_kwargs["max_hold_renewal_score_override"] = (
            max_hold_renewal_score_override
        )
    elif policy.get("max_hold_renewal_score") is not None:
        simulator_kwargs["max_hold_renewal_score_override"] = float(
            policy["max_hold_renewal_score"]
        )
    if max_hold_renewal_rank_override is not None:
        simulator_kwargs["max_hold_renewal_rank_override"] = (
            max_hold_renewal_rank_override
        )
    elif policy.get("max_hold_renewal_rank") is not None:
        simulator_kwargs["max_hold_renewal_rank_override"] = int(
            policy["max_hold_renewal_rank"]
        )
    if portfolio_rebalance_active_override is not None:
        simulator_kwargs["portfolio_rebalance_active_override"] = (
            portfolio_rebalance_active_override
        )
    if portfolio_rebalance_min_weight_deviation_override is not None:
        simulator_kwargs["portfolio_rebalance_min_weight_deviation_override"] = float(
            portfolio_rebalance_min_weight_deviation_override
        )
    if portfolio_rebalance_overweight_deviation_override is not None:
        simulator_kwargs["portfolio_rebalance_overweight_deviation_override"] = float(
            portfolio_rebalance_overweight_deviation_override
        )
    if portfolio_rebalance_underweight_deviation_override is not None:
        simulator_kwargs["portfolio_rebalance_underweight_deviation_override"] = float(
            portfolio_rebalance_underweight_deviation_override
        )
    if entry_block_mask_override is not None:
        simulator_kwargs["entry_block_mask_override"] = entry_block_mask_override
    if maintenance_buy_block_mask_override is not None:
        simulator_kwargs["maintenance_buy_block_mask_override"] = (
            maintenance_buy_block_mask_override
        )
    if score_exit_confirmation_mask_override is not None:
        simulator_kwargs["score_exit_confirmation_mask_override"] = (
            score_exit_confirmation_mask_override
        )
    if exit_score_override is not None:
        simulator_kwargs["exit_score_override"] = exit_score_override
    if max_entry_open_gap_override is not None:
        simulator_kwargs["max_entry_open_gap_override"] = float(
            max_entry_open_gap_override
        )
    if min_entry_open_gap_override is not None:
        simulator_kwargs["min_entry_open_gap_override"] = float(
            min_entry_open_gap_override
        )
    if entry_price_loss_exit_override is not None:
        simulator_kwargs["entry_price_loss_exit_override"] = float(
            entry_price_loss_exit_override
        )
    if defer_sell_on_limit_up_override is not None:
        simulator_kwargs["defer_sell_on_limit_up_override"] = bool(
            defer_sell_on_limit_up_override
        )
    if absolute_max_hold_days_override is not None:
        simulator_kwargs["absolute_max_hold_days_override"] = int(
            absolute_max_hold_days_override
        )
    if position_observer is not None:
        simulator_kwargs["position_observer"] = position_observer
    if score_sell_pressure_trigger_override is not None:
        simulator_kwargs["score_sell_pressure_trigger_override"] = (
            int(score_sell_pressure_trigger_override)
            if np.ndim(score_sell_pressure_trigger_override) == 0
            else np.asarray(score_sell_pressure_trigger_override, dtype=np.int16)
        )
    if score_sell_pressure_limit_override is not None:
        simulator_kwargs["score_sell_pressure_limit_override"] = (
            int(score_sell_pressure_limit_override)
            if np.ndim(score_sell_pressure_limit_override) == 0
            else np.asarray(score_sell_pressure_limit_override, dtype=np.int16)
        )
    if score_sell_pressure_confirmation_days_override is not None:
        simulator_kwargs["score_sell_pressure_confirmation_days_override"] = (
            int(score_sell_pressure_confirmation_days_override)
            if np.ndim(score_sell_pressure_confirmation_days_override) == 0
            else np.asarray(
                score_sell_pressure_confirmation_days_override, dtype=np.int16
            )
        )
    if score_sell_pressure_cooldown_days_override is not None:
        simulator_kwargs["score_sell_pressure_cooldown_days_override"] = int(
            score_sell_pressure_cooldown_days_override
        )
    if score_sell_pressure_min_replacement_score_override is not None:
        simulator_kwargs[
            "score_sell_pressure_min_replacement_score_override"
        ] = float(score_sell_pressure_min_replacement_score_override)
    if score_sell_pressure_extra_min_age_override is not None:
        simulator_kwargs["score_sell_pressure_extra_min_age_override"] = (
            int(score_sell_pressure_extra_min_age_override)
            if np.ndim(score_sell_pressure_extra_min_age_override) == 0
            else np.asarray(
                score_sell_pressure_extra_min_age_override, dtype=np.int16
            )
        )
    if score_sell_priority_override is not None:
        simulator_kwargs["score_sell_priority_override"] = np.asarray(
            score_sell_priority_override, dtype=np.float64
        )
    if score_sell_pressure_observer is not None:
        simulator_kwargs["score_sell_pressure_observer"] = (
            score_sell_pressure_observer
        )
    if pairwise_replacement_guard_override is not None:
        simulator_kwargs["pairwise_replacement_guard_override"] = bool(
            pairwise_replacement_guard_override
        )
    return simulation_function(
        arrays,
        score,
        order,
        definition,
        harness.v162.case_protocol(case_protocol, definition),
        str(end_date),
        research_base.FIRST_BUY,
        record_actions=record_actions,
        gross_target_override=np.full(length, target_gross, dtype=np.float64),
        target_cohorts_override=target_positions,
        target_pct_override=np.full(length, target_weight, dtype=np.float64),
        min_hold_days_override=min_hold,
        selection_mask_override=(
            harness.v174.selection_mask(
                arrays, definition["max_rank_deterioration"]
            )
            if selection_mask_override is None
            else np.asarray(selection_mask_override, dtype=np.bool_)
        ),
        sell_score_below_override=sell_score_schedule,
        sell_confirmation_days_override=int(policy["sell_confirmation_days"]),
        replacement_advantage_age_bands_override=(
            [(int(age), float(value)) for age, value in age_bands]
            if age_bands
            else None
        ),
        max_positions_override=target_positions,
        max_positions_schedule_override=np.full(
            length, target_positions, dtype=np.int16
        ),
        entry_slots_schedule_override=np.full(
            length, target_positions, dtype=np.int16
        ),
        max_daily_score_sells_override=(
            policy.get("max_daily_score_sells")
            if max_daily_score_sells_override is None
            else max_daily_score_sells_override
        ),
        refill_after_sells_override=True,
        max_refill_buys_override=target_positions,
        reentry_cooldown_days_override=int(policy.get("reentry_cooldown_days", 0)),
        max_hold_renewal_policy_override=str(policy["renewal_policy"]),
        **simulator_kwargs,
    )


def interval_metrics(daily: pd.DataFrame, start: str, end: str) -> dict:
    return research_base.metrics(research_base.interval(daily, start, end))


def action_metrics(actions: pd.DataFrame, start: str, end: str) -> dict:
    if actions.empty:
        return {"buy_count": 0, "sell_count": 0, "round_trips": 0}
    date_values = actions["buy_date"].astype(str)
    selected = actions[(date_values >= start) & (date_values <= end)]
    buys = int((selected["action"] == "BUY").sum())
    sells = int((selected["action"] == "SELL").sum())
    return {"buy_count": buys, "sell_count": sells, "round_trips": sells}


def evaluate_run(
    daily: pd.DataFrame,
    actions: pd.DataFrame,
    start: str,
    end: str,
    target_positions: int = 10,
) -> dict:
    metrics = interval_metrics(daily, start, end)
    positions = research_base.interval(daily, start, end)["positions"].astype(int)
    return {
        **metrics,
        **action_metrics(actions, start, end),
        "target_positions": int(target_positions),
        "days_at_target_positions": int((positions == target_positions).sum()),
        "days_below_target_positions": int((positions < target_positions).sum()),
        "full_target_position_ratio": float(
            (positions == target_positions).mean()
        ),
        # Compatibility keys retained for frozen fixed-10 reports.
        "days_at_10_positions": int((positions == 10).sum()),
        "days_below_10_positions": int((positions < 10).sum()),
        "full_10_position_ratio": float((positions == 10).mean()),
    }


def selection_utility(baseline: dict, stress: dict) -> float:
    annual = np.asarray(list(baseline["annual_returns"].values()), dtype=float)
    dispersion = float(np.std(annual, ddof=0)) if len(annual) else 0.0
    return float(
        0.30 * baseline["cagr"]
        + 0.20 * baseline["sharpe"]
        + 0.20 * stress["cagr"]
        + 0.10 * stress["sharpe"]
        - 0.15 * baseline["max_drawdown"]
        - 0.05 * dispersion
        - 0.0005 * baseline["turnover_annualized"]
    )


def numeric_delta(candidate: dict, baseline: dict) -> dict:
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


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    dev = load_arrays(DEVELOPMENT_END)
    harness, protocol, rules, manifests, arrays, development_access = dev
    production_definition = harness.production_definition(protocol)
    score, order = harness.v95.score_pair(arrays, 0.0, 7, 0.1)

    development_results = {}
    for policy_id, policy in HOLD_POLICIES.items():
        daily, actions = run_fixed10(
            harness,
            arrays,
            protocol,
            production_definition,
            score,
            order,
            policy,
            DEVELOPMENT_END,
            slip=BASELINE_COST,
            record_actions=True,
        )
        stress_daily, stress_actions = run_fixed10(
            harness,
            arrays,
            protocol,
            production_definition,
            score,
            order,
            policy,
            DEVELOPMENT_END,
            slip=STRESS_COST,
            record_actions=True,
        )
        baseline_metrics = evaluate_run(
            daily, actions, research_base.FIRST_BUY, DEVELOPMENT_END
        )
        stress_metrics = evaluate_run(
            stress_daily,
            stress_actions,
            research_base.FIRST_BUY,
            DEVELOPMENT_END,
        )
        development_results[policy_id] = {
            "policy": policy,
            "metrics_0_30pct": baseline_metrics,
            "metrics_0_65pct": stress_metrics,
            "selection_utility": selection_utility(baseline_metrics, stress_metrics),
        }

    naive = development_results["production_exit_fixed10"]
    eligible = {
        key: value
        for key, value in development_results.items()
        if value["metrics_0_30pct"]["cagr"] > naive["metrics_0_30pct"]["cagr"]
        and value["metrics_0_30pct"]["sharpe"] >= naive["metrics_0_30pct"]["sharpe"]
        and value["metrics_0_30pct"]["max_drawdown"]
        <= naive["metrics_0_30pct"]["max_drawdown"] * 1.10
        and value["metrics_0_65pct"]["cumulative_return"] > 0.0
    }
    selection_pool = eligible or development_results
    selected_id = max(
        selection_pool,
        key=lambda key: (
            selection_pool[key]["selection_utility"],
            selection_pool[key]["metrics_0_30pct"]["sharpe"],
            key,
        ),
    )
    frozen_policy = copy.deepcopy(HOLD_POLICIES[selected_id])
    freeze = {
        "status": "frozen_before_2026_validation",
        "source_strategy": rules["strategy_id"],
        "score_rule": {
            "weight_5d": 0.0,
            "weight_10d": 1.0,
            "smooth_window": 7,
            "current_day_weight": 0.1,
        },
        "portfolio": {"positions": 10, "target_weight_each": 0.10, "gross_target": 1.0},
        "development_boundary": [research_base.FIRST_BUY, DEVELOPMENT_END],
        "candidate_budget": list(HOLD_POLICIES),
        "selected_policy_id": selected_id,
        "selected_policy": frozen_policy,
        "development_results": development_results,
        "development_access": development_access,
        "validation_opened": False,
    }
    atomic_json(OUTPUT_ROOT / "candidate_frozen_before_2026.json", freeze)

    validation = load_arrays(VALIDATION_END)
    vh, vp, _, _, va, validation_access = validation
    vdef = vh.production_definition(vp)
    vscore, vorder = vh.v95.score_pair(va, 0.0, 7, 0.1)

    production_daily, production_actions = research_base.run_shell(
        vh,
        va,
        vp,
        vscore,
        vorder,
        vdef,
        "production_shell",
        VALIDATION_END,
        actions=True,
    )
    naive_daily, naive_actions = run_fixed10(
        vh,
        va,
        vp,
        vdef,
        vscore,
        vorder,
        HOLD_POLICIES["production_exit_fixed10"],
        VALIDATION_END,
        slip=BASELINE_COST,
        record_actions=True,
    )
    candidate_daily, candidate_actions = run_fixed10(
        vh,
        va,
        vp,
        vdef,
        vscore,
        vorder,
        frozen_policy,
        VALIDATION_END,
        slip=BASELINE_COST,
        record_actions=True,
    )
    repeat_daily, repeat_actions = run_fixed10(
        vh,
        va,
        vp,
        vdef,
        vscore,
        vorder,
        frozen_policy,
        VALIDATION_END,
        slip=BASELINE_COST,
        record_actions=True,
    )
    deterministic = {
        "daily": frame_hash(candidate_daily) == frame_hash(repeat_daily),
        "actions": frame_hash(candidate_actions) == frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("fixed10 candidate deterministic replay failed")

    stress_daily, stress_actions = run_fixed10(
        vh,
        va,
        vp,
        vdef,
        vscore,
        vorder,
        frozen_policy,
        VALIDATION_END,
        slip=STRESS_COST,
        record_actions=True,
    )
    production_metrics = evaluate_run(
        production_daily, production_actions, VALIDATION_START, VALIDATION_END
    )
    naive_metrics = evaluate_run(
        naive_daily, naive_actions, VALIDATION_START, VALIDATION_END
    )
    candidate_metrics = evaluate_run(
        candidate_daily, candidate_actions, VALIDATION_START, VALIDATION_END
    )
    candidate_stress_metrics = evaluate_run(
        stress_daily, stress_actions, VALIDATION_START, VALIDATION_END
    )
    validation_passed = bool(
        candidate_metrics["cumulative_return"] > naive_metrics["cumulative_return"]
        and candidate_metrics["sharpe"] >= naive_metrics["sharpe"]
        and candidate_metrics["max_drawdown"] <= naive_metrics["max_drawdown"] * 1.10
        and candidate_stress_metrics["cumulative_return"] > 0.0
    )

    production_daily.to_csv(
        OUTPUT_ROOT / "production_daily.csv", index=False, encoding="utf-8-sig"
    )
    naive_daily.to_csv(
        OUTPUT_ROOT / "naive_fixed10_daily.csv", index=False, encoding="utf-8-sig"
    )
    candidate_daily.to_csv(
        OUTPUT_ROOT / "candidate_fixed10_daily.csv", index=False, encoding="utf-8-sig"
    )
    candidate_actions.to_csv(
        OUTPUT_ROOT / "candidate_fixed10_actions.csv", index=False, encoding="utf-8-sig"
    )
    result = {
        "schema_version": 1,
        "status": (
            "passed_2026_vs_naive_fixed10_research_only"
            if validation_passed
            else "rejected_on_2026_validation"
        ),
        "source_strategy": rules["strategy_id"],
        "selected_policy_id": selected_id,
        "selected_policy": frozen_policy,
        "development_boundary": [research_base.FIRST_BUY, DEVELOPMENT_END],
        "validation_boundary": [VALIDATION_START, VALIDATION_END],
        "validation_used_for_selection": False,
        "development": development_results,
        "validation_2026": {
            "production": production_metrics,
            "naive_fixed10": naive_metrics,
            "candidate_fixed10": candidate_metrics,
            "candidate_fixed10_0_65pct": candidate_stress_metrics,
            "candidate_minus_naive": numeric_delta(candidate_metrics, naive_metrics),
            "candidate_minus_production": numeric_delta(candidate_metrics, production_metrics),
        },
        "validation_passed_vs_naive_fixed10": validation_passed,
        "deterministic_replay": deterministic,
        "validation_access": validation_access,
        "production_modified": False,
        "trading_triggered": False,
    }
    atomic_json(OUTPUT_ROOT / "result.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected": selected_id,
                "development": development_results[selected_id],
                "validation": result["validation_2026"],
            },
            ensure_ascii=False,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
