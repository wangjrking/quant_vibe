import sys
from pathlib import Path

import pytest


MAIN = Path(__file__).resolve().parents[1]
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))

from research_v260_runtime.fixed10_risk_event_v110 import resolve_score_sell_limit


def test_pressure_limit_stays_at_one_below_trigger():
    assert resolve_score_sell_limit(1, 2, 3, 2) == 1


def test_pressure_limit_rises_to_two_at_trigger():
    assert resolve_score_sell_limit(1, 3, 3, 2) == 2


def test_pressure_configuration_is_atomic():
    with pytest.raises(ValueError, match="provided together"):
        resolve_score_sell_limit(1, 3, 3, None)


def test_pressure_rule_never_reduces_existing_limit():
    assert resolve_score_sell_limit(3, 5, 3, 2) == 3


def test_pressure_confirmation_requires_full_streak():
    assert resolve_score_sell_limit(1, 4, 4, 2, pressure_streak=1, confirmation_days=2) == 1
    assert resolve_score_sell_limit(1, 4, 4, 2, pressure_streak=2, confirmation_days=2) == 2
