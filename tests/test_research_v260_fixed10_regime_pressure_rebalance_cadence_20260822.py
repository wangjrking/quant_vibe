import sys
from pathlib import Path


MAIN = Path(__file__).resolve().parents[1]
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))

from research_v260_fixed10_regime_pressure_rebalance_cadence_20260822 import (
    CANDIDATES,
)


def test_cadence_budget_is_one_existing_neighbor_only():
    assert CANDIDATES == {"rebalance_20d": 20, "rebalance_10d": 10}
