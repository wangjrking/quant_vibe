from __future__ import annotations

import sys
from pathlib import Path


MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_current_exit_raw_alpha_profit_optimization_20260822 as mod


def test_exit_alpha_candidates_are_coarse_and_include_current():
    assert mod.EXIT_RAW_ALPHAS == (0.0, 0.1, 0.2)


def test_exit_score_keeps_window_and_changes_only_alpha():
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

    assert mod.exit_score(Context(), 0.2) == "exit-score"
    assert calls == [({"sentinel": True}, 0.0, 7, 0.2)]
