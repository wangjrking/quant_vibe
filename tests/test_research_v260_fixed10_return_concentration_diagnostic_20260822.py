from __future__ import annotations

import pandas as pd

import research_v260_fixed10_return_concentration_diagnostic_20260822 as diagnostic


def test_monthly_concentration_compounds_days_and_removes_top_month() -> None:
    daily = pd.DataFrame(
        {
            "date": ["20220607", "20220608", "20220701", "20220704"],
            "return": [0.10, -0.05, 0.02, 0.03],
        }
    )
    result = diagnostic.monthly_concentration(daily)
    assert result["month_count"] == 2
    assert result["positive_month_count"] == 2
    top = result["without_top_positive_months"]["1"]
    assert top["months"] == ["202207"]
    assert abs(top["cumulative_return_without_top_months"] - 0.045) < 1e-12
