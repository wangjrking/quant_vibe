from __future__ import annotations

from quant.main.research_v260_fixed10_hold_robustness_20260822 import (
    POLICIES,
    policy,
    robust_key,
    walk_forward_selections,
)


def test_policy_budget_is_bounded() -> None:
    assert len(POLICIES) == 6
    assert {value[0] for value in POLICIES.values()} == {20, 25, 30}
    assert {value[1] for value in POLICIES.values()} == {1, 2}


def test_policy_keeps_production_threshold_and_limits_daily_exit() -> None:
    actual = policy(25, 2)
    assert actual["sell_score_below"] == 0.85
    assert actual["replacement_advantage"] == 0.05
    assert actual["max_daily_score_sells"] == 1
    assert actual["renewal_policy"] == "no_score_exit"


def test_robust_key_prefers_higher_worst_year() -> None:
    common = {
        "min_window_cagr": 0.10,
        "median_annual_return": 0.20,
        "metrics_0_30pct": {"sharpe": 1.0, "max_drawdown": 0.20},
    }
    assert robust_key({**common, "min_annual_return": 0.02}) > robust_key(
        {**common, "min_annual_return": 0.01}
    )


def test_walk_forward_selection_uses_only_prior_years() -> None:
    def result(values: dict[str, float]) -> dict:
        return {"metrics_0_30pct": {"annual_returns": values}}

    results = {
        "stable": result({"2022": 0.1, "2023": 0.1, "2024": 0.1, "2025": 0.1}),
        "late": result({"2022": -0.1, "2023": 0.5, "2024": 0.8, "2025": 1.0}),
    }
    evidence = walk_forward_selections(results)
    assert [item["selected_policy_id"] for item in evidence] == ["stable", "stable"]
    assert [item["forward_test_year"] for item in evidence] == ["2024", "2025"]
