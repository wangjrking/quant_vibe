from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


MAIN_ROOT = Path(r"D:/work/quant/quant_mcp/quant/main")
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_entry_beta_tail_confirmation_20260822 as module


def test_candidate_is_single_and_controls_are_neighbors() -> None:
    assert module.SELECTABLE_PERCENTILES == {
        "production_entry": None,
        "exclude_highest_beta_0_5pct": 0.995,
    }
    assert module.ROBUSTNESS_PERCENTILES == {
        "control_exclude_highest_beta_0_25pct": 0.9975,
        "control_exclude_highest_beta_0_75pct": 0.9925,
    }


def test_confirmation_gate_is_exact_tail_only() -> None:
    production = np.array([[True, True, True]])
    percentile = np.array([[np.nan, 0.995, 0.9951]])
    result = module.entry_selection_mask(
        "exclude_highest_beta_0_5pct", production, percentile
    )
    assert result.tolist() == [[True, True, False]]


def test_unknown_candidate_fails_closed() -> None:
    try:
        module.entry_selection_mask("unknown", np.ones((1, 1)), np.zeros((1, 1)))
    except ValueError as exc:
        assert "unknown candidate" in str(exc)
    else:
        raise AssertionError("unknown candidate must fail closed")
