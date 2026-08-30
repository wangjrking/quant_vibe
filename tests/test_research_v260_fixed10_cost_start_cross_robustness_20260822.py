from __future__ import annotations

import research_v260_fixed10_cost_start_cross_robustness_20260822 as audit


def test_matrix_summary_covers_all_cells_and_extremes() -> None:
    matrix = {
        "0.0030": {
            "0": {"cumulative_return": 1.0, "cagr": 0.2, "max_drawdown": 0.1},
            "5": {"cumulative_return": 0.8, "cagr": 0.18, "max_drawdown": 0.2},
        },
        "0.0065": {
            "0": {"cumulative_return": 0.5, "cagr": 0.1, "max_drawdown": 0.3},
            "5": {"cumulative_return": 0.4, "cagr": 0.08, "max_drawdown": 0.25},
        },
    }
    result = audit.matrix_summary(matrix)
    assert result == {
        "cell_count": 4,
        "all_cumulative_returns_positive": True,
        "all_cagrs_positive": True,
        "minimum_cumulative_return": 0.4,
        "minimum_cagr": 0.08,
        "maximum_drawdown": 0.3,
    }
