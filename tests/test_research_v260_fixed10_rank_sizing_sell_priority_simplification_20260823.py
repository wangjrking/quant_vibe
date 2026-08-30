from __future__ import annotations

import sys
from pathlib import Path


MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_rank_sizing_sell_priority_simplification_20260823 as mod


def test_sell_priority_removal_changes_only_priority_metadata() -> None:
    policy = {
        "confirmed_weak_sell_priority": {"formula": "score-vol"},
        "score_sell_pressure_limit": 2,
    }
    result = mod.without_confirmed_weak_sell_priority(policy)
    assert result["confirmed_weak_sell_priority"] is None
    assert result["score_sell_pressure_limit"] == 2


def test_sell_priority_removal_does_not_mutate_input() -> None:
    policy = {"confirmed_weak_sell_priority": {"formula": "score-vol"}}
    mod.without_confirmed_weak_sell_priority(policy)
    assert policy["confirmed_weak_sell_priority"] == {"formula": "score-vol"}


def test_absent_sell_priority_is_rejected() -> None:
    try:
        mod.without_confirmed_weak_sell_priority(
            {"confirmed_weak_sell_priority": None}
        )
    except ValueError:
        pass
    else:
        raise AssertionError("already absent sell priority was accepted")
