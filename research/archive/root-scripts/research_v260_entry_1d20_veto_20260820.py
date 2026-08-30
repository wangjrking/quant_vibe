# -*- coding: utf-8 -*-
from __future__ import annotations

import json

import research_v260_entry_1d70_confirmation_20260820 as base


OUT = base.ROOT / "quant/data_file/reports/strategy_agent_v260_entry_1d20_veto_20260820"
CANDIDATE_ID = "entry_1d20_veto"
ENTRY_FLOOR = 0.20


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base.OUT = OUT
    protocol = json.loads(base.PROTOCOL_PATH.read_text(encoding="utf-8"))
    arrays = base.v260.v258.v252.fresh_arrays()
    score, order = base.v95.score_pair(arrays, 0.0, 7, 0.1)
    definition = base.v260.definition_for(protocol, 50)
    latest_signal_date = str(arrays["dates"].astype(str)[-2])
    contract = {
        "strategy_id": base.STRATEGY_ID,
        "candidate": "production_entry_1d_bottom20_veto_v1",
        "single_change": "New entries with non-finite rank_1d or rank_1d below 0.20 are rejected; all production scoring, exits, sizing and execution remain unchanged.",
        "development": [base.DEVELOPMENT_START, base.DEVELOPMENT_END],
        "validation": [base.VALIDATION_START, latest_signal_date],
        "production_modified": False,
    }
    (OUT / "research_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    development = {}
    for cost in (base.BASELINE_COST, base.STRESS_COST):
        result, _ = base.evaluate_pair(
            arrays, score, order, definition, protocol,
            base.DEVELOPMENT_START, base.DEVELOPMENT_END, cost, "development",
            candidate_id=CANDIDATE_ID, entry_floor=ENTRY_FLOOR,
        )
        development[f"{cost:.4f}"] = result

    replay, _ = base.evaluate_pair(
        arrays, score, order, definition, protocol,
        base.DEVELOPMENT_START, base.DEVELOPMENT_END,
        base.BASELINE_COST, "development_replay",
        candidate_id=CANDIDATE_ID, entry_floor=ENTRY_FLOOR,
    )
    deterministic = all(
        replay[case][key] == development[f"{base.BASELINE_COST:.4f}"][case][key]
        for case in ("production", CANDIDATE_ID)
        for key in ("daily_digest", "actions_digest")
    )
    delta_base = development[f"{base.BASELINE_COST:.4f}"]["candidate_minus_production"]
    delta_stress = development[f"{base.STRESS_COST:.4f}"]["candidate_minus_production"]
    annual_base = development[f"{base.BASELINE_COST:.4f}"]["production"]["annual"]
    annual_candidate = development[f"{base.BASELINE_COST:.4f}"][CANDIDATE_ID]["annual"]
    improved_years = sum(
        annual_candidate[year]["cumulative_return"] > annual_base[year]["cumulative_return"]
        for year in sorted(annual_base)
    )
    development_passed = bool(
        delta_base["cumulative_return"] > 0
        and delta_base["sharpe"] > 0
        and delta_base["max_drawdown"] <= 0.02
        and delta_stress["cumulative_return"] > 0
        and improved_years >= 3
        and deterministic
    )

    validation = None
    validation_passed = False
    if development_passed:
        validation = {}
        for cost in (base.BASELINE_COST, base.STRESS_COST):
            result, _ = base.evaluate_pair(
                arrays, score, order, definition, protocol,
                base.VALIDATION_START, latest_signal_date, cost, "validation",
                candidate_id=CANDIDATE_ID, entry_floor=ENTRY_FLOOR,
            )
            validation[f"{cost:.4f}"] = result
        val_base = validation[f"{base.BASELINE_COST:.4f}"]["candidate_minus_production"]
        val_stress = validation[f"{base.STRESS_COST:.4f}"]["candidate_minus_production"]
        validation_passed = bool(
            val_base["cumulative_return"] > 0
            and val_base["sharpe"] > 0
            and val_base["max_drawdown"] <= 0.02
            and val_stress["cumulative_return"] > 0
        )

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
        "development_improved_years": improved_years,
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
