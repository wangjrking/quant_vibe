from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HARNESS = (
    ROOT
    / "quant/data_file/runtime/agent_workspaces/strategy-agent/work"
    / "prod_v260_fixed10_equalweight_development_comparison_20260812_r2"
    / "observation_attempt_v1/run_comparison.py"
)
OUT = ROOT / "quant/data_file/reports/strategy_agent_v260_full_investment_ab_20260821"


def load_harness():
    spec = importlib.util.spec_from_file_location("v260_development_harness", HARNESS)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load development harness: {HARNESS}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_full_investment(harness, arrays, score, order, definition, protocol, slip):
    case_protocol = copy.deepcopy(protocol)
    case_protocol["execution"]["fixed_slippage_ratio"] = float(slip)

    schedule = harness.v260.position_schedule(
        arrays, definition, harness.FIRST_BUY_DATE
    )
    production_gross, production_target = harness.v162.v153.schedules(
        arrays, definition
    )
    weak = production_gross < 1.0

    target = production_target.copy()
    target[weak] = 1.0 / schedule[weak].astype(float)

    multiplier = harness.v260.v258.v252.warmup_multiplier(
        arrays,
        score,
        definition,
        case_protocol,
        harness.FIRST_BUY_DATE,
    )
    return harness.v109.simulate(
        arrays,
        score,
        order,
        definition,
        harness.v162.case_protocol(case_protocol, definition),
        harness.END_DATE,
        harness.FIRST_BUY_DATE,
        record_actions=True,
        gross_target_override=np.ones(len(arrays["dates"]), dtype=float),
        target_cohorts_override=4,
        target_pct_override=target,
        min_hold_days_override=harness.v162.min_hold_schedule(
            arrays, definition["min_hold_policy"]
        ),
        selection_mask_override=harness.v174.selection_mask(
            arrays, definition["max_rank_deterioration"]
        ),
        candidate_target_multiplier_override=multiplier,
        max_positions_schedule_override=schedule,
    )


def delta(base: dict, candidate: dict) -> dict:
    fields = (
        "cumulative_return",
        "cagr",
        "sharpe",
        "max_drawdown",
        "turnover_annualized",
        "round_trips",
        "average_invested_ratio",
        "high_invested_day_ratio_gte_95pct",
        "average_positions",
    )
    return {key: float(candidate[key]) - float(base[key]) for key in fields}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    harness = load_harness()
    protocol, rules, manifests = harness.validate_control_plane()
    arrays, access = harness.build_development_arrays()
    score, order = harness.v95.score_pair(arrays, 0.0, 7, 0.1)
    definition = harness.production_definition(protocol)

    runs = {}
    for slip in (0.003, 0.004, 0.0065):
        baseline_daily, baseline_actions = harness.run_baseline(
            arrays, score, order, definition, protocol, slip
        )
        candidate_daily, candidate_actions = run_full_investment(
            harness, arrays, score, order, definition, protocol, slip
        )
        label = f"slippage_{slip:.4f}"
        baseline = harness.metric_summary(baseline_daily, baseline_actions)
        candidate = harness.metric_summary(candidate_daily, candidate_actions)
        runs[label] = {
            "baseline": baseline,
            "full_investment": candidate,
            "delta": delta(baseline, candidate),
        }
        if slip == 0.003:
            baseline_daily.to_csv(
                OUT / "baseline_daily.csv", index=False, encoding="utf-8-sig"
            )
            baseline_actions.to_csv(
                OUT / "baseline_actions.csv", index=False, encoding="utf-8-sig"
            )
            candidate_daily.to_csv(
                OUT / "full_investment_daily.csv", index=False, encoding="utf-8-sig"
            )
            candidate_actions.to_csv(
                OUT / "full_investment_actions.csv", index=False, encoding="utf-8-sig"
            )

    repeat_daily, repeat_actions = run_full_investment(
        harness, arrays, score, order, definition, protocol, 0.003
    )
    deterministic = {
        "daily": harness.frame_hash(repeat_daily)
        == runs["slippage_0.0030"]["full_investment"]["daily_hash"],
        "actions": harness.frame_hash(repeat_actions)
        == runs["slippage_0.0030"]["full_investment"]["actions_hash"],
    }
    if not all(deterministic.values()):
        raise RuntimeError("deterministic replay failed")

    payload = {
        "schema_version": 1,
        "status": "research_ab_completed",
        "candidate": "v260_full_investment_weak_cash_gate_removed_v1",
        "coverage": {
            "start": harness.FIRST_BUY_DATE,
            "end": harness.END_DATE,
            "2025_read": False,
            "2026_read": False,
        },
        "single_change": {
            "unchanged": [
                "production 10D score and 7-day smoothing",
                "entry ranking and one-new-name-per-day",
                "exit rules",
                "14/15/16 market-state position-width schedule",
                "raw T+1 open, eligibility, costs and round lots",
                "score-trend sizing multiplier",
            ],
            "changed": [
                "weak-market gross target is forced from defensive cash to 1.00",
                "weak-market per-new-name target is 1/current position limit",
            ],
        },
        "source_strategy": rules["strategy_id"],
        "source_manifests": manifests,
        "access": access,
        "results": runs,
        "deterministic_replay": deterministic,
        "production_modified": False,
    }
    (OUT / "comparison_result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=harness.json_default)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["results"]["slippage_0.0030"], ensure_ascii=False))


if __name__ == "__main__":
    main()
