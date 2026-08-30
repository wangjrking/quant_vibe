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
CHECKPOINT = (
    REPORTS
    / "strategy_agent_v260_fixed10_residual_cash_sweep_checkpoint_20260823/"
    "pre2026_checkpoint.json"
)
PROTOCOL = (
    REPORTS
    / "strategy_agent_v260_fixed10_one_shot_2026_validation_protocol_20260822/"
    "one_shot_validation_protocol.json"
)
SATURATION = (
    REPORTS
    / "strategy_agent_v260_fixed10_pre2026_search_saturation_20260822/"
    "search_saturation.json"
)
INTEGRITY = (
    REPORTS
    / "strategy_agent_v260_fixed10_pre2026_freeze_integrity_20260822/"
    "freeze_integrity.json"
)
RESULT_AUDIT = (
    REPORTS
    / "strategy_agent_v260_fixed10_one_shot_2026_result_audit_20260822/"
    "readonly_result_audit.json"
)
OUTPUT_ROOT = (
    REPORTS / "strategy_agent_v260_fixed10_goal_completion_readiness_20260822"
)
PRODUCTION_STRATEGY = "prod_v260_10d_regime_warmup_all4key_v20260724"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_requirements(
    checkpoint: dict,
    protocol: dict,
    saturation: dict,
    integrity: dict,
    result_audit: dict | None,
) -> dict[str, bool]:
    metrics = checkpoint["metrics_0_30pct"]
    return {
        "production_strategy_lineage_exact": (
            checkpoint["source_strategy"] == PRODUCTION_STRATEGY
        ),
        "equalweight_full_investment_direction_documented": (
            int(metrics["target_positions"]) == 10
            and "average_positions" in metrics
            and "full_10_position_ratio" in metrics
            and "average_invested_ratio" in metrics
        ),
        "all_tuning_ends_before_20260101": (
            str(protocol["development_boundary"]["end"]) < "20260101"
        ),
        "validation_starts_on_or_after_20260101": (
            str(protocol["validation_boundary"]["start"]) >= "20260101"
            and protocol["validation_boundary"]["read_once"] is True
        ),
        "pre2026_rule_search_saturated": (
            saturation["status"]
            == "pre2026_broad_rule_search_saturated_2026_not_opened"
            and saturation["remaining_pre2026_selection_candidates"] == []
        ),
        "candidate_and_protocol_freeze_integrity": (
            integrity["status"]
            == "pre2026_freeze_integrity_passed_2026_not_opened"
            and all(integrity["gates"].values())
        ),
        "one_shot_2026_result_readonly_audited": (
            result_audit is not None
            and result_audit.get("status") == "readonly_result_audit_passed"
            and result_audit.get("validation_2026_opened") is True
        ),
        "production_unmodified": (
            checkpoint["production_modified"] is False
            and protocol["production_modified"] is False
            and integrity["production_modified"] is False
            and (
                result_audit is None
                or result_audit.get("production_modified") is False
            )
        ),
    }


def main() -> None:
    checkpoint = read(CHECKPOINT)
    protocol = read(PROTOCOL)
    saturation = read(SATURATION)
    integrity = read(INTEGRITY)
    result_audit = read(RESULT_AUDIT) if RESULT_AUDIT.is_file() else None
    requirements = build_requirements(
        checkpoint,
        protocol,
        saturation,
        integrity,
        result_audit,
    )
    incomplete = [name for name, passed in requirements.items() if not passed]
    result = {
        "status": (
            "goal_completion_evidence_ready"
            if not incomplete
            else "goal_active_requirements_remaining"
        ),
        "candidate_id": checkpoint["selected_candidate"],
        "requirements": requirements,
        "incomplete_requirements": incomplete,
        "next_action": (
            "completion audit and close goal"
            if not incomplete
            else "complete the frozen one-shot 2026 validation and readonly audit"
        ),
        "validation_2026_opened": result_audit is not None,
        "production_modified": False,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    round1.atomic_json(OUTPUT_ROOT / "goal_completion_readiness.json", result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
