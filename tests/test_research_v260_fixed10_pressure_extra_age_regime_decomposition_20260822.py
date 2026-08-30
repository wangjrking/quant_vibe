import sys
from pathlib import Path


MAIN = Path(__file__).resolve().parents[1]
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))

from research_v260_fixed10_pressure_extra_age_regime_decomposition_20260822 import age_schedule


def test_age_schedule_maps_strong_and_weak_states():
    assert age_schedule([True, False, True], 8, 4).tolist() == [8, 4, 8]
    assert age_schedule([True, False, True], 4, 8).tolist() == [4, 8, 4]
