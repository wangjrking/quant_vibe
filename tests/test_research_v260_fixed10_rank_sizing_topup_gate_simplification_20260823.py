from __future__ import annotations

import sys
from pathlib import Path


MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_rank_sizing_topup_gate_simplification_20260823 as mod


def test_topup_gate_removal_changes_only_one_key() -> None:
    policy = {"maintenance_topup_requires_score": 0.8, "sell_score_below": 0.85}
    result = mod.without_maintenance_topup_score_gate(policy)
    assert result["maintenance_topup_requires_score"] is None
    assert result["sell_score_below"] == 0.85
    assert mod.changed_policy_keys(policy, result) == [
        "maintenance_topup_requires_score"
    ]


def test_topup_gate_removal_does_not_mutate_input() -> None:
    policy = {"maintenance_topup_requires_score": 0.8}
    mod.without_maintenance_topup_score_gate(policy)
    assert policy["maintenance_topup_requires_score"] == 0.8


def test_absent_topup_gate_is_rejected() -> None:
    try:
        mod.without_maintenance_topup_score_gate(
            {"maintenance_topup_requires_score": None}
        )
    except ValueError:
        pass
    else:
        raise AssertionError("already absent gate was accepted as a new change")
