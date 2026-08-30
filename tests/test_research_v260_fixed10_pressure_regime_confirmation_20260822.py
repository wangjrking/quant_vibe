import sys
from pathlib import Path

import numpy as np


MAIN = Path(__file__).resolve().parents[1]
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))

from research_v260_fixed10_pressure_regime_confirmation_20260822 import (
    pressure_confirmation_schedule,
)
from research_v260_runtime.fixed10_risk_event_v110 import resolve_score_sell_limit


def test_pressure_confirmation_schedule_is_one_strong_two_weak():
    actual = pressure_confirmation_schedule([True, False, True])
    assert actual.dtype == np.int16
    assert actual.tolist() == [1, 2, 1]


def test_second_day_confirmation_remains_fail_closed_on_day_one():
    assert resolve_score_sell_limit(1, 5, 5, 2, 1, 2) == 1
    assert resolve_score_sell_limit(1, 5, 5, 2, 2, 2) == 2
