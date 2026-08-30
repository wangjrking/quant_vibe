from __future__ import annotations

import pytest

from research_v260_fixed10_rank_sizing_walkforward_amplitude_20260823 import (
    compound,
    walkforward_split,
)


def test_compound_uses_multiplicative_returns() -> None:
    assert compound({"2022": 0.10, "2023": -0.10}, ("2022", "2023")) == pytest.approx(-0.01)


def test_walkforward_selection_is_made_only_on_train_years() -> None:
    annual = {
        "early": {"2022": 0.20, "2023": 0.10, "2024": -0.10},
        "late": {"2022": 0.10, "2023": 0.10, "2024": 0.30},
    }
    actual = walkforward_split(annual, ("2022", "2023"), ("2024",))
    assert actual["train_selected"] == "early"
    assert actual["test_winner"] == "late"
    assert actual["train_selection_is_test_winner"] is False
    assert actual["selected_test_regret_vs_test_winner"] == pytest.approx(0.40)


def test_missing_year_is_rejected() -> None:
    with pytest.raises(ValueError, match="missing annual"):
        compound({"2022": 0.1}, ("2022", "2023"))
