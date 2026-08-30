import sys
from pathlib import Path

import numpy as np
import pytest


MAIN = Path(__file__).resolve().parents[1]
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))

from research_v260_fixed10_regime_pressure_trigger_boundary_20260822 import (
    pressure_trigger_schedule,
)


def test_pressure_trigger_boundary_preserves_strong_state():
    assert pressure_trigger_schedule([True, False, True], 5).tolist() == [4, 5, 4]
    assert pressure_trigger_schedule([True, False, True], 6).tolist() == [4, 6, 4]


def test_pressure_trigger_boundary_rejects_unapproved_search():
    with pytest.raises(ValueError):
        pressure_trigger_schedule(np.asarray([True, False]), 7)
