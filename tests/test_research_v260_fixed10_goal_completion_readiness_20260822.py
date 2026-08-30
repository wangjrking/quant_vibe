from __future__ import annotations

import sys
from pathlib import Path


MAIN_ROOT = Path(r"D:/work/quant/quant_mcp/quant/main")
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_goal_completion_readiness_20260822 as subject


def test_completion_readiness_binds_current_cash_sweep_candidate() -> None:
    assert (
        subject.CHECKPOINT.parent.name
        == "strategy_agent_v260_fixed10_residual_cash_sweep_checkpoint_20260823"
    )


def fixtures():
    metrics = {
        "target_positions": 10,
        "average_positions": 10.0,
        "full_10_position_ratio": 1.0,
        "average_invested_ratio": 0.98,
    }
    checkpoint = {
        "source_strategy": subject.PRODUCTION_STRATEGY,
        "metrics_0_30pct": metrics,
        "production_modified": False,
    }
    protocol = {
        "development_boundary": {"end": "20251231"},
        "validation_boundary": {"start": "20260101", "read_once": True},
        "production_modified": False,
    }
    saturation = {
        "status": "pre2026_broad_rule_search_saturated_2026_not_opened",
        "remaining_pre2026_selection_candidates": [],
    }
    integrity = {
        "status": "pre2026_freeze_integrity_passed_2026_not_opened",
        "gates": {"all": True},
        "production_modified": False,
    }
    audit = {
        "status": "readonly_result_audit_passed",
        "validation_2026_opened": True,
        "production_modified": False,
    }
    return checkpoint, protocol, saturation, integrity, audit


def test_completion_requires_audited_one_shot_without_elapsed_time_gate() -> None:
    checkpoint, protocol, saturation, integrity, audit = fixtures()
    before = subject.build_requirements(
        checkpoint, protocol, saturation, integrity, None
    )
    assert "optimization_elapsed_at_least_24h" not in before
    assert before["one_shot_2026_result_readonly_audited"] is False
    after = subject.build_requirements(
        checkpoint, protocol, saturation, integrity, audit
    )
    assert all(after.values())


def test_equalweight_direction_is_soft_but_early_validation_is_rejected() -> None:
    checkpoint, protocol, saturation, integrity, audit = fixtures()
    checkpoint["metrics_0_30pct"]["average_positions"] = 9.9
    checkpoint["metrics_0_30pct"]["average_invested_ratio"] = 0.80
    protocol["validation_boundary"]["start"] = "20251231"
    requirements = subject.build_requirements(
        checkpoint, protocol, saturation, integrity, audit
    )
    assert requirements["equalweight_full_investment_direction_documented"] is True
    assert requirements["validation_starts_on_or_after_20260101"] is False
