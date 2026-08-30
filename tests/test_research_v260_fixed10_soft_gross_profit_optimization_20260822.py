from __future__ import annotations

import sys
from pathlib import Path


MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_soft_gross_profit_optimization_20260822 as mod


def test_gross_candidates_are_simple_and_bounded():
    assert mod.GROSS_TARGETS == (0.90, 0.95, 1.00)
    assert all(0.0 < gross <= 1.0 for gross in mod.GROSS_TARGETS)


def test_target_weight_is_equal_weight_direction():
    assert 0.95 / mod.TARGET_POSITIONS == 0.095
