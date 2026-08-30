from __future__ import annotations

import sys
from pathlib import Path


MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_current_entry_5d_profit_optimization_20260822 as mod


def test_entry_5d_candidates_are_broad_and_include_current():
    assert mod.ENTRY_5D_WEIGHTS == (0.0, 0.1, 0.3, 0.5)
