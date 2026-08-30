from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


MAIN_ROOT = Path(r"D:/work/quant/quant_mcp/quant/main")
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_confirmed_severe_weak_defensive_sleeve_20260822 as subject


def test_confirmed_weak_requires_consecutive_days_and_resets() -> None:
    strong = np.array([True, False, False, False, False, True, False, False, False])
    np.testing.assert_array_equal(
        subject.confirmed_weak_mask(strong, 3),
        np.array([False, False, False, True, True, False, False, False, True]),
    )


def test_confirmation_one_matches_all_weak_days() -> None:
    strong = np.array([True, False, True, False])
    np.testing.assert_array_equal(subject.confirmed_weak_mask(strong, 1), ~strong)
