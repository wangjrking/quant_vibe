from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


MAIN_ROOT = Path(r"D:/work/quant/quant_mcp/quant/main")
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_entry_beta_tail_guard_20260822 as module


def test_percent_rank_is_stable_and_cross_sectional() -> None:
    values = np.array([[2.0, np.nan, 1.0, 3.0]])
    result = module.cross_sectional_percent_rank(values)
    assert np.allclose(result[0, [0, 2, 3]], [0.5, 0.0, 1.0])
    assert np.isnan(result[0, 1])


def test_tail_gate_allows_unknown_and_rejects_only_tail() -> None:
    production = np.array([[True, True, True, False]])
    percentile = np.array([[np.nan, 0.99, 0.9901, 0.0]])
    result = module.entry_selection_mask(
        "exclude_highest_beta_1pct", production, percentile
    )
    assert result.tolist() == [[True, True, False, False]]


def test_candidate_and_control_budgets_are_separate() -> None:
    assert set(module.SELECTABLE_PERCENTILES) == {
        "production_entry",
        "exclude_highest_beta_1pct",
        "exclude_highest_beta_2_5pct",
    }
    assert not (set(module.SELECTABLE_PERCENTILES) & set(module.ROBUSTNESS_PERCENTILES))


def test_unknown_candidate_fails_closed() -> None:
    try:
        module.entry_selection_mask("unknown", np.ones((1, 1)), np.zeros((1, 1)))
    except ValueError as exc:
        assert "unknown candidate" in str(exc)
    else:
        raise AssertionError("unknown candidate must fail closed")
