import sys
from pathlib import Path

import numpy as np


MAIN = Path(__file__).resolve().parents[1]
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))

from research_v260_fixed10_pressure_regime_replacement_margin_20260822 import (
    replacement_margin_schedule,
)


def test_replacement_margin_is_005_strong_007_weak():
    actual = replacement_margin_schedule([True, False, True])
    assert actual.dtype == np.float64
    assert np.allclose(actual, [0.05, 0.07, 0.05])


def test_replacement_margin_schedule_and_scalar_are_both_numeric_contracts():
    assert np.ndim(0.05) == 0
    assert np.ndim(replacement_margin_schedule([True, False])) == 1
