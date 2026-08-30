from __future__ import annotations

import sys
from pathlib import Path


MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_current_exit_smoothing_profit_optimization_20260822 as mod


def test_exit_smoothing_candidates_are_coarse_and_include_current():
    assert mod.EXIT_SMOOTHING_WINDOWS == (3, 5, 7, 10)


def test_exit_score_requests_only_declared_window():
    calls = []

    class ScoreHarness:
        class v95:
            @staticmethod
            def score_pair(arrays, weight_1d, window, alpha):
                calls.append((arrays, weight_1d, window, alpha))
                return "exit-score", "ignored-order"

    class Context:
        harness = ScoreHarness()
        arrays = {"sentinel": True}

    assert mod.exit_score(Context(), 5) == "exit-score"
    assert calls == [({"sentinel": True}, 0.0, 5, 0.1)]
