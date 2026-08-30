from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


MAIN = Path(__file__).resolve().parents[1]
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))

from research_v260_runtime.fixed10_risk_event_v110 import (
    pairwise_supported_score_sells,
)


def test_each_score_sell_requires_its_corresponding_replacement() -> None:
    selected = pairwise_supported_score_sells(
        score_sell_candidates=[0, 1],
        replacement_candidates=[2, 3],
        exit_scores=np.array([0.70, 0.75, 0.90, 0.78]),
        replacement_scores=np.array([0.70, 0.75, 0.90, 0.78]),
        required_advantages={0: 0.05, 1: 0.05},
        limit=2,
    )

    assert selected == [0]


def test_pairwise_guard_accepts_two_independently_supported_sells() -> None:
    selected = pairwise_supported_score_sells(
        score_sell_candidates=[1, 0],
        replacement_candidates=[2, 3],
        exit_scores=np.array([0.70, 0.75, 0.90, 0.82]),
        replacement_scores=np.array([0.70, 0.75, 0.90, 0.82]),
        required_advantages={0: 0.05, 1: 0.05},
        limit=2,
    )

    assert selected == [0, 1]


def test_mandatory_sell_reserves_the_best_replacement() -> None:
    selected = pairwise_supported_score_sells(
        score_sell_candidates=[0, 1],
        replacement_candidates=[2, 3, 4],
        exit_scores=np.array([0.70, 0.75, 0.95, 0.82, 0.79]),
        replacement_scores=np.array([0.70, 0.75, 0.95, 0.82, 0.79]),
        required_advantages={0: 0.05, 1: 0.05},
        limit=2,
        reserved_replacements=1,
    )

    assert selected == [0]


def test_nonfinite_replacement_fails_closed() -> None:
    selected = pairwise_supported_score_sells(
        score_sell_candidates=[0],
        replacement_candidates=[1],
        exit_scores=np.array([0.70, np.nan]),
        replacement_scores=np.array([0.70, np.nan]),
        required_advantages={0: 0.05},
        limit=1,
    )

    assert selected == []
