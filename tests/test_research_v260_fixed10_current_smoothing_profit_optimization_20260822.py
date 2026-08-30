from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_current_smoothing_profit_optimization_20260822 as mod


def test_smoothing_candidates_are_broad_neighbors():
    assert mod.SMOOTHING_WINDOWS == (5, 7, 10)


def test_context_with_window_replaces_only_score_and_order():
    class ScoreBuilder:
        @staticmethod
        def score_pair(arrays, weight, window, alpha):
            return np.full((1, 2), window, dtype=float), np.array([[1, 0]])

    context = mod.harness.PressureContext(
        harness=type("Harness", (), {"v95": ScoreBuilder()})(),
        protocol={},
        definition={},
        rules={},
        manifests={},
        arrays={"dates": np.array(["20220101"])},
        access={},
        score=np.zeros((1, 2)),
        order=np.array([[0, 1]]),
        active=np.array([True]),
        maintenance_block=np.zeros((1, 2), dtype=bool),
        empty_block=np.zeros((1, 2), dtype=bool),
    )
    changed = mod.context_with_window(context, 5)
    assert np.array_equal(changed.score, np.full((1, 2), 5.0))
    assert np.array_equal(changed.order, np.array([[1, 0]]))
    assert changed.arrays is context.arrays
