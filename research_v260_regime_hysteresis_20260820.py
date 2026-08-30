# -*- coding: utf-8 -*-
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

import research_v260_entry_1d70_confirmation_20260820 as base


OUT = base.ROOT / "quant/data_file/reports/strategy_agent_v260_regime_hysteresis_20260820"
WEAK_ENTER_NORMAL = 0.005
NORMAL_EXIT_WEAK = -0.005
NORMAL_ENTER_STRONG = 0.040
STRONG_EXIT_NORMAL = 0.030


def state_schedule(arrays: dict[str, np.ndarray], definition: dict) -> np.ndarray:
    momentum = base.v260.v258.v134.market_momentum(
        arrays, int(definition["market_lookback"])
    )
    state = np.zeros(len(momentum), dtype=np.int8)
    current = 0
    for index, value in enumerate(momentum):
        if not np.isfinite(value):
            current = 0
        elif current == 0:
            if value >= NORMAL_ENTER_STRONG:
                current = 2
            elif value >= WEAK_ENTER_NORMAL:
                current = 1
        elif current == 1:
            if value >= NORMAL_ENTER_STRONG:
                current = 2
            elif value < NORMAL_EXIT_WEAK:
                current = 0
        else:
            if value < NORMAL_EXIT_WEAK:
                current = 0
            elif value < STRONG_EXIT_NORMAL:
                current = 1
        state[index] = current
    return state


def candidate_run(
    arrays: dict[str, np.ndarray],
    score: np.ndarray,
    order: np.ndarray,
    definition: dict,
    protocol: dict,
    end: str,
    start: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    state = state_schedule(arrays, definition)
    gross = np.where(state == 0, float(definition["low_gross"]), 1.0)
    target = np.full(len(state), float(definition["weak_target_pct"]), dtype=float)
    target[state == 1] = float(definition["normal_target_pct"])
    target[state == 2] = float(definition["strong_target_pct"])
    min_hold = np.full(len(state), 4, dtype=np.int16)
    min_hold[state == 2] = 3
    max_positions = np.full(len(state), 14, dtype=np.int16)
    max_positions[state == 1] = 15
    max_positions[state == 2] = 16
    dates = arrays["dates"].astype(str)
    start_index = int(np.searchsorted(dates, start))
    max_positions[start_index : start_index + int(definition["position_warmup_days"])] = 16

    return base.v260.v258.v252.v162.v109.simulate(
        arrays,
        score,
        order,
        definition,
        base.v260.v258.v252.v162.case_protocol(protocol, definition),
        end,
        start,
        gross_target_override=gross,
        target_cohorts_override=4,
        target_pct_override=target,
        min_hold_days_override=min_hold,
        record_actions=True,
        selection_mask_override=base.v260.v258.v252.v174.selection_mask(
            arrays, definition["max_rank_deterioration"]
        ),
        candidate_target_multiplier_override=base.v260.v258.v252.warmup_multiplier(
            arrays, score, definition, protocol, start
        ),
        max_positions_override=int(definition["max_positions"]),
        max_positions_schedule_override=max_positions,
    )


def evaluate(
    arrays: dict[str, np.ndarray],
    score: np.ndarray,
    order: np.ndarray,
    definition: dict,
    protocol: dict,
    start: str,
    end: str,
    cost: float,
    scope: str,
) -> dict:
    case_protocol = copy.deepcopy(protocol)
    case_protocol["execution"]["fixed_slippage_ratio"] = cost
    output = {}
    for case in ("production", "regime_hysteresis"):
        if case == "production":
            daily, actions = base.v260.run_case(
                arrays, score, order, definition, case_protocol, end, start,
                record_actions=True,
            )
        else:
            daily, actions = candidate_run(
                arrays, score, order, definition, case_protocol, end, start
            )
        output[case] = {
            "metrics": base.metrics_for(daily, start, end),
            "annual": base.annual_metrics(daily, start, end),
            "daily_digest": base.frame_digest(daily),
            "actions_digest": base.frame_digest(actions),
        }
        daily.to_csv(
            OUT / f"{scope}_{case}_cost_{cost:.4f}_daily.csv",
            index=False,
            encoding="utf-8-sig",
        )
        actions.to_csv(
            OUT / f"{scope}_{case}_cost_{cost:.4f}_actions.csv",
            index=False,
            encoding="utf-8-sig",
        )
    baseline = output["production"]["metrics"]
    candidate = output["regime_hysteresis"]["metrics"]
    output["candidate_minus_production"] = {
        key: candidate[key] - baseline[key]
        for key in baseline
        if key in candidate and isinstance(candidate[key], (int, float))
    }
    return output


def passes_gate(results: dict) -> bool:
    baseline = results[f"{base.BASELINE_COST:.4f}"]["candidate_minus_production"]
    stress = results[f"{base.STRESS_COST:.4f}"]["candidate_minus_production"]
    return bool(
        baseline["cumulative_return"] > 0
        and baseline["sharpe"] > 0
        and baseline["max_drawdown"] <= 0.02
        and stress["cumulative_return"] > 0
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(base.PROTOCOL_PATH.read_text(encoding="utf-8"))
    arrays = base.v260.v258.v252.fresh_arrays()
    score, order = base.v95.score_pair(arrays, 0.0, 7, 0.1)
    definition = base.v260.definition_for(protocol, 50)
    latest_signal_date = str(arrays["dates"].astype(str)[-2])
    contract = {
        "strategy_id": base.STRATEGY_ID,
        "candidate": "production_regime_hysteresis_v1",
        "single_change": "All regime-dependent exposure, target size, minimum hold and position-width controls consume one hysteretic market state.",
        "thresholds": {
            "weak_enter_normal": WEAK_ENTER_NORMAL,
            "normal_exit_weak": NORMAL_EXIT_WEAK,
            "normal_enter_strong": NORMAL_ENTER_STRONG,
            "strong_exit_normal": STRONG_EXIT_NORMAL,
        },
        "development": [base.DEVELOPMENT_START, base.DEVELOPMENT_END],
        "validation": [base.VALIDATION_START, latest_signal_date],
        "production_modified": False,
    }
    (OUT / "research_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    development = {
        f"{cost:.4f}": evaluate(
            arrays, score, order, definition, protocol,
            base.DEVELOPMENT_START, base.DEVELOPMENT_END, cost, "development"
        )
        for cost in (base.BASELINE_COST, base.STRESS_COST)
    }
    replay = evaluate(
        arrays, score, order, definition, protocol,
        base.DEVELOPMENT_START, base.DEVELOPMENT_END,
        base.BASELINE_COST, "development_replay"
    )
    deterministic = all(
        replay[case][key] == development[f"{base.BASELINE_COST:.4f}"][case][key]
        for case in ("production", "regime_hysteresis")
        for key in ("daily_digest", "actions_digest")
    )
    development_passed = passes_gate(development) and deterministic

    validation = None
    validation_passed = False
    if development_passed:
        validation = {
            f"{cost:.4f}": evaluate(
                arrays, score, order, definition, protocol,
                base.VALIDATION_START, latest_signal_date, cost, "validation"
            )
            for cost in (base.BASELINE_COST, base.STRESS_COST)
        }
        validation_passed = passes_gate(validation)

    status = (
        "validated_candidate"
        if validation_passed
        else "rejected_in_validation"
        if development_passed
        else "rejected_in_development"
    )
    result = {
        "status": status,
        "development_passed": development_passed,
        "validation_passed": validation_passed,
        "deterministic_replay": deterministic,
        "development": development,
        "validation": validation,
        "production_modified": False,
    }
    (OUT / "ab_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
