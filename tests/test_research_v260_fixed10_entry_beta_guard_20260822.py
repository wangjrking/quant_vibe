from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


MAIN_ROOT = Path(r"D:/work/quant/quant_mcp/quant/main")
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_entry_beta_guard_20260822 as module


def test_beta_guard_is_coarse_and_entry_only() -> None:
    assert module.CANDIDATE_BETA_CAPS == {
        "production_entry": None,
        "entry_beta_at_most_200": 2.0,
        "entry_beta_at_most_150": 1.5,
    }
    assert module.ROBUSTNESS_BETA_CAPS == {
        "control_entry_beta_at_most_175": 1.75,
        "control_entry_beta_at_most_225": 2.25,
    }


def test_beta_cap_allows_unknown_history_and_rejects_high_beta() -> None:
    production = np.array([[True, True, True, False]])
    beta = np.array([[np.nan, 1.5, 1.5001, 0.0]])
    result = module.entry_selection_mask(
        "entry_beta_at_most_150", production, beta
    )
    assert result.tolist() == [[True, True, False, False]]


def test_rolling_beta_is_past_only() -> None:
    market = np.array([100.0, 101.0, 99.0, 102.0, 101.0, 104.0])
    close = np.column_stack([market, market**2 / 100.0])
    first = module.trailing_market_beta(close, lookback=4, min_observations=3)
    changed = close.copy()
    changed[-1] *= 10.0
    second = module.trailing_market_beta(changed, lookback=4, min_observations=3)
    assert np.allclose(first[:-1], second[:-1], equal_nan=True)


def test_unknown_candidate_fails_closed() -> None:
    try:
        module.entry_selection_mask("unknown", np.ones((1, 1)), np.zeros((1, 1)))
    except ValueError as exc:
        assert "unknown candidate" in str(exc)
    else:
        raise AssertionError("unknown candidate must fail closed")
