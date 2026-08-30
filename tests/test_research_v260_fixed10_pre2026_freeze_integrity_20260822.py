from __future__ import annotations

import sys
from pathlib import Path


MAIN_ROOT = Path(r"D:/work/quant/quant_mcp/quant/main")
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

from research_v260_fixed10_pre2026_freeze_integrity_20260822 import (
    validate_cross_artifact_consistency,
)


def fixtures() -> tuple[dict, dict, dict]:
    policy = {"sell_threshold": 0.85, "target_positions": 10}
    metrics = {"cumulative_return": 4.0}
    checkpoint = {
        "selected_candidate": "candidate",
        "selected_policy": policy,
        "metrics_0_30pct": metrics,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    protocol = {
        "development_boundary": {
            "frozen_candidate": "candidate",
            "candidate_policy": policy,
            "end": "20251231",
        },
        "validation_boundary": {"start": "20260101", "read_once": True},
        "event_overlay_role": "diagnostic_only_not_selection_candidate",
        "hard_boundaries": {
            "event_overlay_cannot_change_base_decision": True,
            "no_parameter_tuning": True,
            "no_extra_candidate": True,
        },
        "validation_2026_opened": False,
        "production_modified": False,
    }
    ledger = {
        "current_candidate": {
            "candidate_id": "candidate",
            "policy": policy,
            "metrics_0_30pct": metrics,
        },
        "validation_2026_opened": False,
        "production_modified": False,
    }
    return checkpoint, protocol, ledger


def test_cross_artifact_consistency_accepts_identical_freeze() -> None:
    assert all(validate_cross_artifact_consistency(*fixtures()).values())


def test_cross_artifact_consistency_rejects_candidate_or_boundary_drift() -> None:
    checkpoint, protocol, ledger = fixtures()
    ledger["current_candidate"]["candidate_id"] = "drifted"
    protocol["validation_boundary"]["start"] = "20251231"
    checks = validate_cross_artifact_consistency(checkpoint, protocol, ledger)
    assert checks["candidate_id_identical"] is False
    assert checks["development_ends_before_validation"] is False
