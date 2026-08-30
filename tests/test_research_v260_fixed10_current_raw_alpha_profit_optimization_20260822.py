from __future__ import annotations

import sys
from pathlib import Path


MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_current_raw_alpha_profit_optimization_20260822 as mod


def test_raw_alpha_candidates_are_simple_neighbors():
    assert mod.RAW_ALPHAS == (0.0, 0.1, 0.2)
    assert mod.SMOOTHING_WINDOW == 7
