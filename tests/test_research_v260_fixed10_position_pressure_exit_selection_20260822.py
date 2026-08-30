import sys
from pathlib import Path


MAIN = Path(__file__).resolve().parents[1]
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))

from research_v260_fixed10_position_pressure_exit_selection_20260822 import (
    coarse_mechanism_gates,
    positive_annual_delta_count,
)


def _case(cagr, sharpe, drawdown, holdout, stress, annual, triggers, years):
    return {
        "train_2022_2024": {
            "cagr": cagr,
            "sharpe": sharpe,
            "max_drawdown": drawdown,
        },
        "holdout_2025": {"cumulative_return": holdout},
        "metrics_0_65pct": {"cumulative_return": stress},
        "metrics_0_30pct": {"annual_returns": annual},
        "pressure_diagnostics": {
            "triggered_days": triggers,
            "triggered_years": years,
        },
    }


def test_positive_annual_delta_count_is_calendar_based():
    baseline = _case(0, 0, 1, 0, 0, {"2022": 0.1, "2023": 0.2}, 0, [])
    candidate = _case(0, 0, 1, 0, 0, {"2022": 0.2, "2023": 0.1}, 0, [])
    assert positive_annual_delta_count(candidate, baseline) == 1


def test_coarse_gate_accepts_supported_region_without_boundary_veto():
    baseline = _case(
        0.30,
        0.90,
        0.40,
        0.90,
        1.90,
        {"2022": 0.10, "2023": 0.20, "2024": 0.30, "2025": 0.40},
        0,
        [],
    )
    trigger4 = _case(
        0.33,
        0.95,
        0.37,
        1.00,
        2.10,
        {"2022": 0.09, "2023": 0.25, "2024": 0.35, "2025": 0.45},
        20,
        ["2022", "2023", "2024", "2025"],
    )
    trigger5 = _case(
        0.34,
        0.98,
        0.35,
        0.98,
        2.08,
        {"2022": 0.09, "2023": 0.24, "2024": 0.36, "2025": 0.44},
        8,
        ["2022", "2023", "2024"],
    )
    trigger6 = _case(
        0.28,
        0.84,
        0.36,
        0.95,
        1.85,
        {"2022": 0.10, "2023": 0.18, "2024": 0.28, "2025": 0.42},
        3,
        ["2023", "2024"],
    )
    results = {
        "baseline_one_exit": baseline,
        "control_trigger4_limit2": trigger4,
        "candidate_trigger5_limit2": trigger5,
        "control_trigger6_limit2": trigger6,
    }
    assert all(coarse_mechanism_gates(results).values())
