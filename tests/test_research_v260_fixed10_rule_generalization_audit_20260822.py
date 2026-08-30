from __future__ import annotations

from research_v260_fixed10_rule_generalization_audit_20260822 import support_summary


def metrics(annual, cumulative):
    return {"annual_returns": annual, "cumulative_return": cumulative}


def fixture(full_years, removed_years, gains):
    full = {
        "metrics_0_30pct": metrics(full_years, gains[0]),
        "metrics_0_65pct": {"cumulative_return": gains[1]},
        "train_2022_2024": {"cumulative_return": gains[2]},
        "holdout_2025": {"cumulative_return": gains[3]},
    }
    removed = {
        "metrics": {
            "metrics_0_30pct": metrics(removed_years, 0.0),
            "metrics_0_65pct": {"cumulative_return": 0.0},
            "train_2022_2024": {"cumulative_return": 0.0},
            "holdout_2025": {"cumulative_return": 0.0},
        }
    }
    return full, removed


def test_three_positive_years_is_robust_core() -> None:
    full, removed = fixture(
        {"2022": 0.1, "2023": 0.1, "2024": 0.1, "2025": 0.0},
        {"2022": 0.0, "2023": 0.0, "2024": 0.0, "2025": 0.0},
        (0.3, 0.2, 0.2, 0.1),
    )
    assert support_summary(full, removed)["classification"] == "robust_core_rule"


def test_two_positive_two_equal_is_secondary_fixed_rule() -> None:
    full, removed = fixture(
        {"2022": 0.0, "2023": 0.0, "2024": 0.1, "2025": 0.1},
        {"2022": 0.0, "2023": 0.0, "2024": 0.0, "2025": 0.0},
        (0.2, 0.2, 0.1, 0.1),
    )
    assert support_summary(full, removed)["classification"] == (
        "secondary_fixed_rule_not_independent_tuning_knob"
    )


def test_negative_holdout_gain_is_not_supported() -> None:
    full, removed = fixture(
        {"2022": 0.1, "2023": 0.1, "2024": 0.1, "2025": -0.1},
        {"2022": 0.0, "2023": 0.0, "2024": 0.0, "2025": 0.0},
        (0.2, 0.2, 0.2, -0.1),
    )
    assert support_summary(full, removed)["classification"] == (
        "insufficient_generalization_support"
    )
