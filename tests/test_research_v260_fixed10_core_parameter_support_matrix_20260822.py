from __future__ import annotations

import research_v260_fixed10_core_parameter_support_matrix_20260822 as target


def test_concise_concentration_drops_daily_and_monthly_series():
    source = {
        "total_log_excess": 1.0,
        "equivalent_relative_wealth_gain": 2.0,
        "active_day_count": 3,
        "positive_day_count": 2,
        "negative_day_count": 1,
        "top1_positive_share": 0.2,
        "top5_positive_share": 0.5,
        "top10_absolute_share": 0.6,
        "log_excess_after_removing_top5_positive_days": 0.1,
        "positive_month_fraction": 0.7,
        "yearly_log_excess": {"2022": 1.0},
        "monthly_log_excess": {"202201": 1.0},
    }
    result = target.concise_concentration(source)
    assert "monthly_log_excess" not in result
    assert result["yearly_log_excess"] == {"2022": 1.0}


def test_value_id_preserves_integer_and_float_semantics():
    assert target.value_id(20) == "20"
    assert target.value_id(0.05) == "0.0500"
