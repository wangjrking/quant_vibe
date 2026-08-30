from __future__ import annotations

import research_v260_fixed10_min_hold4_fragility_diagnostic_20260822 as target


def bootstrap(probability_return, probability_sharpe, lower_bound):
    return {
        "probability_annualized_log_return_higher": probability_return,
        "probability_sharpe_higher": probability_sharpe,
        "annualized_log_return_delta": {"p025": lower_bound},
    }


def test_support_summary_requires_broad_positive_evidence():
    concentration = {
        "yearly_log_excess": {"2022": 0.1, "2023": 0.2, "2024": 0.3, "2025": -0.01},
        "log_excess_after_removing_top5_positive_days": 0.05,
    }
    bootstraps = {
        "5": bootstrap(0.90, 0.85, 0.01),
        "20": bootstrap(0.88, 0.82, 0.00),
        "60": bootstrap(0.81, 0.80, 0.02),
    }
    result = target.support_summary(concentration, bootstraps)
    assert result["positive_year_fraction"] == 0.75
    assert result["positive_after_removing_top5_days"] is True
    assert result["return_probability_above_80pct_all_blocks"] is True
    assert result["sharpe_probability_above_80pct_all_blocks"] is True
    assert result["return_ci_lower_bound_nonnegative_all_blocks"] is True


def test_support_summary_surfaces_concentration_and_uncertainty():
    concentration = {
        "yearly_log_excess": {"2022": 0.1, "2023": -0.2},
        "log_excess_after_removing_top5_positive_days": -0.01,
    }
    bootstraps = {"5": bootstrap(0.79, 0.95, -0.01)}
    result = target.support_summary(concentration, bootstraps)
    assert result["positive_year_fraction"] == 0.5
    assert result["positive_after_removing_top5_days"] is False
    assert result["return_probability_above_80pct_all_blocks"] is False
    assert result["return_ci_lower_bound_nonnegative_all_blocks"] is False
