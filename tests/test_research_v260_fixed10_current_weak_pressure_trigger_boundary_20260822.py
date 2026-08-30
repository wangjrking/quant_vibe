from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


MAIN_ROOT = Path(r"D:/work/quant/quant_mcp/quant/main")
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_current_weak_pressure_trigger_boundary_20260822 as subject


def test_trigger_schedule_changes_only_weak_market() -> None:
    strong = np.array([True, False, False, True], dtype=np.bool_)
    np.testing.assert_array_equal(
        subject.trigger_schedule(strong, 6),
        np.array([4, 6, 6, 4]),
    )


def test_trigger_schedule_is_deterministic() -> None:
    strong = np.array([False, True], dtype=np.bool_)
    np.testing.assert_array_equal(
        subject.trigger_schedule(strong, 5),
        subject.trigger_schedule(strong, 5),
    )
