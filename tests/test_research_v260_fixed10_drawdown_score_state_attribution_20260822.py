import pytest

from research_v260_fixed10_drawdown_score_state_attribution_20260822 import (
    annual_direction,
    classify_row,
    normalized_rows,
    summarize_flags,
    summarize_subset,
)


def row(year, pnl, exposure, flag):
    return {
        "year": str(year),
        "mark_pnl": float(pnl),
        "prior_value": float(exposure),
        "mark_return": float(pnl / exposure),
        "risk": bool(flag),
    }


def test_classification_uses_existing_thresholds_only() -> None:
    actual = classify_row(0.84, -0.01, 0.75, True)
    assert all(actual.values())
    boundary = classify_row(0.85, 0.0, 0.7499, False)
    assert not boundary["score_below_existing_exit_threshold"]
    assert not boundary["score_declining_3d"]
    assert not boundary["high_volatility"]


def test_subset_summary_is_exposure_and_pnl_exact() -> None:
    actual = summarize_subset([row(2022, -10, 100, True), row(2022, 10, 200, False)])
    assert actual["position_days"] == 2
    assert actual["gross_exposure"] == 300.0
    assert actual["net_mark_pnl"] == 0.0
    assert actual["gross_negative_mark_pnl"] == -10.0
    assert actual["mean_mark_return"] == pytest.approx(-0.025)
    assert actual["negative_position_day_ratio"] == 0.5


def test_flag_shares_use_exposure_and_gross_negative_denominators() -> None:
    rows = [row(2022, -20, 100, True), row(2022, -10, 300, False)]
    actual = summarize_flags(rows, ("risk",))["flags"]["risk"]
    assert actual["share_of_rows"] == 0.5
    assert actual["share_of_gross_exposure"] == 0.25
    assert actual["share_of_gross_negative"] == pytest.approx(2.0 / 3.0)


def test_annual_direction_compares_flagged_with_unflagged() -> None:
    rows = [
        row(2022, -20, 100, True),
        row(2022, 10, 100, False),
        row(2023, -5, 100, True),
        row(2023, 0, 100, False),
    ]
    actual = annual_direction(rows, "risk")
    assert actual["2022"]["flagged_minus_unflagged"] == pytest.approx(-0.30)
    assert actual["2023"]["flagged_minus_unflagged"] == pytest.approx(-0.05)


def test_normalized_rows_makes_missing_float_values_deterministic() -> None:
    assert normalized_rows([{"value": float("nan")}]) == [{"value": None}]
