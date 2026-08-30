from __future__ import annotations

import research_v260_fixed10_joint_parameter_perturbation_20260822 as target


def test_fractional_factorial_is_balanced_and_unique():
    cases = target.fractional_factorial_cases()
    assert len(cases) == 8
    assert len({case["case_id"] for case in cases}) == 8
    for key in (
        "min_hold_days",
        "sell_score_below",
        "replacement_advantage",
        "volatility_penalty",
    ):
        values = [case[key] for case in cases]
        unique = sorted(set(values))
        assert len(unique) == 2
        assert values.count(unique[0]) == values.count(unique[1]) == 4


def test_fractional_factorial_uses_d_equals_abc():
    for case in target.fractional_factorial_cases():
        a = case["min_hold_days"] - 4
        b = round((case["sell_score_below"] - 0.85) / 0.01)
        c = round((case["replacement_advantage"] - 0.05) / 0.01)
        d = round((case["volatility_penalty"] - 0.05) / 0.01)
        assert d == a * b * c
