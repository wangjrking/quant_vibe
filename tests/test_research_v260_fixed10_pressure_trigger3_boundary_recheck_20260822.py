from __future__ import annotations

import sys
from pathlib import Path


MAIN = Path(__file__).resolve().parents[1]
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))

from research_v260_fixed10_pressure_trigger3_boundary_recheck_20260822 import (
    TRIGGERS,
)


def test_only_unchecked_lower_boundary_is_added() -> None:
    assert TRIGGERS == (3, 4)
