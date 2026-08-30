from __future__ import annotations

import sys
from pathlib import Path


MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_rank_sizing_pressure_age_simplification_20260823 as mod


def test_uniform4_age_policy_changes_age_fields_only() -> None:
    policy = {
        "score_sell_pressure_extra_min_age": 8,
        "score_sell_pressure_extra_min_age_rule": "4_strong_market_10_weak_market",
        "score_sell_pressure_limit": 2,
    }
    result = mod.uniform4_age_policy(policy)
    assert result["score_sell_pressure_extra_min_age"] == 4
    assert result["score_sell_pressure_extra_min_age_rule"] == "uniform4"
    assert result["score_sell_pressure_limit"] == 2


def test_uniform4_age_policy_does_not_mutate_input() -> None:
    policy = {
        "score_sell_pressure_extra_min_age": 8,
        "score_sell_pressure_extra_min_age_rule": "4_strong_market_10_weak_market",
    }
    mod.uniform4_age_policy(policy)
    assert policy["score_sell_pressure_extra_min_age"] == 8


def test_uniform4_age_policy_preserves_unrelated_rules() -> None:
    policy = {
        "score_sell_pressure_extra_min_age": 8,
        "score_sell_pressure_extra_min_age_rule": "4_strong_market_10_weak_market",
        "sell_score_below": 0.85,
    }
    result = mod.uniform4_age_policy(policy)
    assert result["sell_score_below"] == 0.85
