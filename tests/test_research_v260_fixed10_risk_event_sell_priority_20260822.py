from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


MAIN_ROOT = Path(r"D:/work/quant/quant_mcp/quant/main")
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_risk_event_sell_priority_20260822 as subject


def test_event_names_are_ordered_before_non_event_names() -> None:
    base = np.array([[0.10, 0.20, 0.30, 0.40]], dtype=float)
    event = np.array([[False, True, False, True]])

    result = subject.event_first_priority(base, event)

    assert result[0, 1] < result[0, 0]
    assert result[0, 3] < result[0, 2]


def test_existing_priority_is_preserved_within_each_tier() -> None:
    base = np.array([[0.40, 0.10, 0.30, 0.20]], dtype=float)
    event = np.array([[True, False, True, False]])

    result = subject.event_first_priority(base, event)

    assert result[0, 2] < result[0, 0]
    assert result[0, 1] < result[0, 3]


def test_event_priority_fails_closed_on_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="do not align"):
        subject.event_first_priority(
            np.ones((2, 2)), np.ones((1, 2), dtype=bool)
        )
