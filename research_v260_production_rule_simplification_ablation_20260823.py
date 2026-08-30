from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import research_v260_lowrisk_score_tuning_20260821 as research_base


REPO = Path(r"D:/work/quant/quant_mcp")
OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports"
    / "strategy_agent_v260_production_rule_simplification_ablation_20260823"
)
RESULT_PATH = OUTPUT_ROOT / "rule_simplification_ablation.json"
DEVELOPMENT_START = "20220607"
DEVELOPMENT_END = "20251231"
BASELINE_COST = 0.003
STRESS_COST = 0.0065


CASES = {
    "production": {},
    "remove_position_width_warmup": {"position_warmup": False},
    "remove_weak_gross_gate": {"weak_gross": 1.00},
    "remove_score_warmup": {"score_warmup": False},
    "remove_score_target_multiplier": {"score_multiplier": False},
    "fixed_width_15": {"fixed_width": 15},
    "fixed_min_hold_4": {"fixed_min_hold": 4},
    "remove_rank_deterioration_filter": {"rank_filter": False},
    "remove_strong_concentration": {"strong_target": 0.25},
    "increase_strong_concentration_40pct": {"strong_target": 0.40},
    "simplified_robust_candidate": {
        "position_warmup": False,
        "weak_gross": 0.40,
        "score_multiplier": False,
        "strong_target": 0.25,
    },
}


def frame_hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256(
        frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def classify_rules(results: dict) -> dict:
    baseline = results["production"]

    def unchanged(case: str) -> bool:
        return all(
            results[case][cost]["daily_sha256"]
            == baseline[cost]["daily_sha256"]
            and results[case][cost]["actions_sha256"]
            == baseline[cost]["actions_sha256"]
            for cost in ("0.003", "0.0065")
        )

    return {
        "development_no_op_cleanup_candidates": {
            "position_width_warmup": {
                "evidence": "byte-identical daily and actions at both costs",
                "verified": unchanged("remove_position_width_warmup"),
                "reason": "The 50-day width override is a dead branch under the current schedule.",
            },
            "weak_gross_gate": {
                "evidence": "byte-identical daily and actions at both costs",
                "verified": unchanged("remove_weak_gross_gate"),
                "reason": "The weak-market gross setting does not reach executed targets.",
            },
        },
        "retain_profit_logic": {
            "dynamic_14_15_16_width": "Fixed width 15 reduces baseline return materially.",
            "strong_market_min_hold_3": "Forcing four days reduces return and worsens drawdown.",
            "rank_deterioration_filter": "Removing it reduces baseline return and worsens drawdown.",
        },
        "replace_architecturally_not_by_parameter_patch": {
            "score_warmup": (
                "A simulation-start-relative 50-day scoring mode is a cold-start artifact. "
                "Replace it with persisted historical state before changing live behavior."
            )
        },
        "fragile_profit_rules_keep_frozen_pending_better_replacement": {
            "score_target_multiplier_0_9_to_1_1": (
                "Raises baseline aggregate return but loses badly under 0.65% cost; "
                "removal is more robust but less profitable."
            ),
            "strong_market_37_5pct_target": (
                "Raises baseline return but worsens stress cost and concentration risk."
            ),
        },
        "reject_as_overfit": {
            "strong_market_40pct_target": (
                "Aggregate gain is concentrated in 2025 while 2023/2024 and 0.65% stress worsen."
            ),
            "fixed10_layered_exit_refill_stack": (
                "Development uplift failed the independent 2026 comparison; composition and "
                "holding-state drift dominated the result."
            ),
        },
    }


def run_case(harness, arrays, protocol, definition, score, order, changes, cost):
    case_protocol = copy.deepcopy(protocol)
    case_protocol["execution"]["fixed_slippage_ratio"] = float(cost)
    case_definition = copy.deepcopy(definition)
    if "weak_gross" in changes:
        case_definition["low_gross"] = float(changes["weak_gross"])
    if "strong_target" in changes:
        case_definition["strong_target_pct"] = float(changes["strong_target"])

    selection = (
        harness.v174.selection_mask(
            arrays, case_definition["max_rank_deterioration"]
        )
        if changes.get("rank_filter", True)
        else None
    )
    if not changes.get("score_multiplier", True):
        multiplier = None
    else:
        multiplier_definition = case_definition
        if not changes.get("score_warmup", True):
            multiplier_definition = {**case_definition, "warmup_days": 0}
        multiplier = harness.v260.v258.v252.warmup_multiplier(
            arrays,
            score,
            multiplier_definition,
            case_protocol,
            DEVELOPMENT_START,
        )

    if "fixed_width" in changes:
        positions = np.full(
            len(arrays["dates"]), int(changes["fixed_width"]), dtype=np.int16
        )
    elif not changes.get("position_warmup", True):
        positions = harness.v260.v258.position_schedule(arrays, case_definition)
    else:
        positions = harness.v260.position_schedule(
            arrays, case_definition, DEVELOPMENT_START
        )
    min_hold = (
        np.full(
            len(arrays["dates"]),
            int(changes["fixed_min_hold"]),
            dtype=np.int16,
        )
        if "fixed_min_hold" in changes
        else None
    )
    return harness.v162.run_case(
        arrays,
        score,
        order,
        case_definition,
        case_protocol,
        DEVELOPMENT_END,
        DEVELOPMENT_START,
        record_actions=True,
        selection_mask_override=selection,
        candidate_target_multiplier_override=multiplier,
        min_hold_days_schedule_override=min_hold,
        max_positions_schedule_override=positions,
    )


def summarize(daily: pd.DataFrame, actions: pd.DataFrame) -> dict:
    selected = research_base.interval(daily, DEVELOPMENT_START, DEVELOPMENT_END)
    return {
        **research_base.metrics(selected),
        "daily_sha256": frame_hash(daily),
        "actions_sha256": frame_hash(actions),
        "buy_count": int((actions["action"] == "BUY").sum()),
        "sell_count": int((actions["action"] == "SELL").sum()),
    }


def build_report() -> dict:
    harness = research_base.load_harness()
    harness.END_DATE = DEVELOPMENT_END
    protocol, _, manifests = harness.validate_control_plane()
    arrays, access = harness.build_development_arrays()
    if str(access["logical_max_date"]) != DEVELOPMENT_END:
        raise RuntimeError("development boundary drifted")
    if any(str(date) >= "20260101" for date in arrays["dates"]):
        raise PermissionError("2026 entered rule selection")

    definition = harness.production_definition(protocol)
    score, order = harness.v95.score_pair(arrays, 0.0, 7, 0.1)
    results = {}
    for name, changes in CASES.items():
        results[name] = {}
        for cost in (BASELINE_COST, STRESS_COST):
            daily, actions = run_case(
                harness,
                arrays,
                protocol,
                definition,
                score,
                order,
                changes,
                cost,
            )
            results[name][str(cost)] = summarize(daily, actions)

    classifications = classify_rules(results)
    if not all(
        item["verified"]
        for item in classifications["development_no_op_cleanup_candidates"].values()
    ):
        raise RuntimeError("a proposed dead-rule removal changed behavior")

    return {
        "status": "production_rule_simplification_analysis_complete",
        "selection_boundary": [DEVELOPMENT_START, DEVELOPMENT_END],
        "2026_used_for_candidate_selection": False,
        "costs": [BASELINE_COST, STRESS_COST],
        "source_manifest_count": len(manifests),
        "results": results,
        "rule_classification": classifications,
        "decision": {
            "historical_no_op_cleanup_candidates": [
                "remove position_width_warmup",
                "remove weak_gross_gate",
            ],
            "observed_development_business_delta": 0.0,
            "future_behavior_guaranteed_unchanged": False,
            "new_profit_candidate_selected": False,
            "reason": (
                "No economically meaningful deletion improved baseline profit without "
                "worsening robustness. The 40% strong-market concentration result is rejected "
                "as year-concentrated and cost-fragile."
            ),
            "next_research_direction": (
                "Improve score quality or regime-independent calibration; do not add more "
                "holding, refill, or cash-sweep branches."
            ),
        },
        "production_modified": False,
        "registry_modified": False,
        "trading_triggered": False,
    }


def main() -> None:
    report = build_report()
    atomic_json(RESULT_PATH, report)
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
