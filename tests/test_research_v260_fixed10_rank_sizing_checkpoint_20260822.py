from __future__ import annotations

import sys
from pathlib import Path


MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_rank_sizing_checkpoint_20260822 as mod


def test_checkpoint_costs_are_fixed_and_include_stress() -> None:
    assert mod.COST_LEVELS == (0.0030, 0.0040, 0.0065)


def test_checkpoint_start_offsets_are_predeclared() -> None:
    assert mod.START_OFFSETS == (0, 5, 20, 60)
