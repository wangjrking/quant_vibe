from __future__ import annotations

import research_v260_fixed10_rule_ablation_support_matrix_20260822 as target


def test_simplification_requires_return_margin_and_weak_incremental_support():
    metrics = {
        "metrics_0_30pct": {
            "cumulative_return": 2.0,
            "cagr": 0.4,
            "annual_returns": {"2022": 0.1, "2023": 0.2},
            "full_10_position_ratio": 1.0,
        },
        "metrics_0_65pct": {"cumulative_return": 1.0},
    }
    production = {"cumulative_return": 1.5, "cagr": 0.3}
    evidence = {"broad_support_for_retaining_rule": False}
    assert target.simplification_eligible(metrics, production, evidence) is True
    evidence["broad_support_for_retaining_rule"] = True
    assert target.simplification_eligible(metrics, production, evidence) is False
