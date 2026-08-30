from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import research_v260_fixed10_pairwise_rule_interaction_audit_20260822 as audit


def test_pair_ids_cover_all_six_rule_pairs_once() -> None:
    pairs = audit.pair_ids()
    assert len(pairs) == 15
    assert len(set(pairs)) == 15
    assert all(left != right for left, right in pairs)


def test_apply_removals_combines_rules_without_mutating_source_policy() -> None:
    policy = {
        "score_sell_pressure_limit": 2,
        "renewal_policy": "no_score_exit",
        "max_hold_renewal_score": 0.8,
        "portfolio_rebalance_interval_days": 20,
    }
    context = SimpleNamespace(score=np.array([[0.3, 0.8]], dtype=np.float64))
    components = {
        "extra_age": np.array([4, 10]),
        "priority_matrix": np.array([[0.2, 0.7]], dtype=np.float64),
    }
    case, age, trigger, sell_priority = audit.apply_removals(
        policy,
        context,
        components,
        ("pressure_second_exit", "weak_volatility_sell_priority"),
    )
    assert case["score_sell_pressure_limit"] == 1
    assert np.array_equal(age, components["extra_age"])
    assert trigger is None
    assert sell_priority is context.score
    assert policy["score_sell_pressure_limit"] == 2


def test_apply_removals_rejects_unknown_rule() -> None:
    with pytest.raises(ValueError, match="unknown rule removals"):
        audit.apply_removals(
            {},
            SimpleNamespace(score=np.array([])),
            {"extra_age": 4, "priority_matrix": np.array([])},
            ("unknown",),
        )
