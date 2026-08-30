from __future__ import annotations

import research_v260_fixed10_weak_confirmation_support_diagnostic_20260822 as target


def test_broad_support_rejects_single_year_gain():
    summary = {
        "total_log_excess": 0.1,
        "yearly_log_excess": {"2022": 0.0, "2023": 0.0, "2024": 0.0, "2025": 0.1},
        "log_excess_after_removing_top5_positive_days": 0.01,
    }
    paired = {
        "5": {
            "probability_annualized_log_return_higher": 0.95,
            "probability_sharpe_higher": 0.95,
            "annualized_log_return_delta": {"p025": 0.0},
        }
    }
    result = target.broad_support(summary, paired)
    assert result["positive_year_fraction"] == 0.25
    assert result["broad_support_for_two_days"] is False
