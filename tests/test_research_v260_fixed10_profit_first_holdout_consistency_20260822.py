from __future__ import annotations

import sys
from pathlib import Path


MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_profit_first_holdout_consistency_20260822 as mod


def candidate(annual_returns: dict[str, float]) -> dict:
    return {"metrics_0_30pct": {"annual_returns": annual_returns}}


def test_compound_annual_returns() -> None:
    result = mod.compound_annual_returns(
        {"2022": 0.10, "2023": 0.20, "2024": -0.10},
        ("2022", "2023", "2024"),
    )
    assert abs(result - 0.188) < 1e-12


def test_period_winner_is_profit_driven() -> None:
    winner, values = mod.period_winner(
        {
            "9": candidate({"2022": 0.10, "2023": 0.10, "2024": 0.10}),
            "10": candidate({"2022": 0.05, "2023": 0.05, "2024": 0.05}),
            "11": candidate({"2022": 0.20, "2023": 0.20, "2024": 0.20}),
        },
        ("2022", "2023", "2024"),
    )
    assert winner == "11"
    assert values["11"] > values["10"]


def test_missing_year_fails_closed() -> None:
    try:
        mod.compound_annual_returns({"2022": 0.1}, ("2022", "2023"))
    except ValueError as exc:
        assert "2023" in str(exc)
    else:
        raise AssertionError("missing year was silently accepted")


def test_numeric_selected_value_matches_serialized_candidate_key() -> None:
    assert mod.normalize_selected(1.0, {"0.90", "0.95", "1.00"}) == "1.00"
    assert mod.normalize_selected(0.0, {"0.0", "0.1"}) == "0.0"
