from __future__ import annotations

import sys
from pathlib import Path


MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_rank_sizing_pressure_trigger_simplification_20260823 as mod


def test_uniform4_policy_changes_only_rule_label() -> None:
    policy = {
        "score_sell_pressure_trigger": 4,
        "score_sell_pressure_trigger_rule": "4_strong_market_5_weak_market",
        "score_sell_pressure_limit": 2,
    }
    result = mod.uniform4_policy(policy)
    assert result["score_sell_pressure_trigger"] == 4
    assert result["score_sell_pressure_trigger_rule"] == "uniform4"
    assert result["score_sell_pressure_limit"] == 2


def test_uniform4_policy_does_not_mutate_input() -> None:
    policy = {
        "score_sell_pressure_trigger": 4,
        "score_sell_pressure_trigger_rule": "4_strong_market_5_weak_market",
    }
    mod.uniform4_policy(policy)
    assert policy["score_sell_pressure_trigger_rule"] == "4_strong_market_5_weak_market"


def test_nonfour_current_trigger_is_rejected() -> None:
    try:
        mod.uniform4_policy({"score_sell_pressure_trigger": 5})
    except ValueError:
        pass
    else:
        raise AssertionError("unexpected current trigger was accepted")
