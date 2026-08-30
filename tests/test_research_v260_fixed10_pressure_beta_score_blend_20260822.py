from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


MAIN = Path(__file__).resolve().parents[1]
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))

from research_v260_fixed10_pressure_beta_score_blend_20260822 import (
    CANDIDATES,
    blended_score,
    stable_descending_order,
)


def test_candidate_budget_is_one_fixed_blend() -> None:
    assert CANDIDATES == ("production_score", "production95_low_beta05")


def test_blend_rewards_lower_beta_for_equal_production_scores() -> None:
    actual = blended_score(
        np.array([[0.8, 0.8]], dtype=float),
        np.array([[0.0, 1.0]], dtype=float),
    )

    assert actual[0, 0] > actual[0, 1]


def test_missing_beta_keeps_original_score() -> None:
    actual = blended_score(
        np.array([[0.7, 0.8]], dtype=float),
        np.array([[np.nan, 0.5]], dtype=float),
    )

    assert actual[0, 0] == np.float32(0.7)


def test_order_is_descending_and_stable() -> None:
    actual = stable_descending_order(np.array([[0.5, 0.7, 0.7, np.nan]]))

    np.testing.assert_array_equal(actual, np.array([[1, 2, 0, 3]]))
