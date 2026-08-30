from __future__ import annotations

import sys
from pathlib import Path


MAIN = Path(__file__).resolve().parents[1]
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))

from research_v260_fixed10_pressure_beta_cap_recheck_20260822 import (
    BETA_CAP,
    CANDIDATES,
)


def test_candidate_budget_is_one_coarse_risk_guard() -> None:
    assert CANDIDATES == ("beta_cap_off", "beta_cap_150")
    assert BETA_CAP == 1.50
