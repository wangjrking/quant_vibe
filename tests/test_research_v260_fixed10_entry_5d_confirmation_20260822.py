from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np


MAIN_ROOT = Path(r"D:/work/quant/quant_mcp/quant/main")
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_entry_5d_confirmation_20260822 as module
import research_v260_fixed10_hold_optimization_20260822 as round1


def test_candidate_budget_is_coarse() -> None:
    assert module.CANDIDATE_IDS == (
        "production_entry",
        "entry_rank5d_at_least_050",
        "entry_rank5d_at_least_070",
    )


def test_entry_floor_composes_with_production_mask() -> None:
    production = np.array([[True, True, False, True]])
    rank_5d = np.array([[0.49, 0.50, 0.90, np.nan]])
    result = module.entry_selection_mask(
        "entry_rank5d_at_least_050", production, rank_5d
    )
    assert result.tolist() == [[False, True, False, False]]


def test_production_candidate_is_exact_mask_copy() -> None:
    production = np.array([[True, False]])
    result = module.entry_selection_mask(
        "production_entry", production, np.array([[np.nan, np.nan]])
    )
    assert np.array_equal(result, production)
    assert result is not production


def test_wrapper_defaults_to_production_selection_mask() -> None:
    assert inspect.signature(round1.run_fixed10).parameters[
        "selection_mask_override"
    ].default is None
