from __future__ import annotations

import numpy as np

import research_v260_fixed10_market_state_branch_support_20260822 as target


def test_uniform_schedule_is_constant() -> None:
    schedule = target.uniform_schedule((2, 3), 5)

    assert schedule.dtype == np.int16
    assert schedule.shape == (2, 3)
    assert np.all(schedule == 5)


def test_broad_support_rejects_top_day_concentration() -> None:
    summary = {
        "total_log_excess": 0.1,
        "yearly_log_excess": {"2022": 0.01, "2023": 0.02, "2024": 0.03, "2025": 0.04},
        "log_excess_after_removing_top5_positive_days": -0.01,
    }
    paired = {
        "5": {
            "probability_annualized_log_return_higher": 0.95,
            "probability_sharpe_higher": 0.95,
            "annualized_log_return_delta": {"p025": 0.01},
        }
    }

    result = target.broad_support(summary, paired)

    assert result["positive_year_fraction"] == 1.0
    assert result["broad_support_for_current_branch"] is False
