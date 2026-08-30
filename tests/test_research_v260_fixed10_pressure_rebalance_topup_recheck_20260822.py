import sys
from pathlib import Path


MAIN = Path(__file__).resolve().parents[1]
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))

from research_v260_fixed10_pressure_rebalance_topup_recheck_20260822 import (
    REBALANCE_INTERVALS,
    TOPUP_SCORES,
)


def test_rebalance_and_topup_budgets_are_coarse():
    assert REBALANCE_INTERVALS == (10, 20, 40)
    assert TOPUP_SCORES == (0.75, 0.80, 0.85)
