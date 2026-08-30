from __future__ import annotations

import numpy as np
import pytest

from research_v260_fixed10_entry_rank_sizing_profit_optimization_20260822 import (
    entry_rank_multipliers,
)
from research_v260_fixed10_rank_sizing_midpoint_profit_check_20260823 import (
    MIDPOINT_BOTTOM,
    MIDPOINT_TOP,
)


def test_midpoint_schedule_is_symmetric_and_gross_neutral() -> None:
    order = np.array([[0, 1, 2, 3]], dtype=np.int64)
    actual = entry_rank_multipliers(
        order,
        positions=4,
        top_multiplier=MIDPOINT_TOP,
        bottom_multiplier=MIDPOINT_BOTTOM,
    )
    assert actual[0].tolist() == pytest.approx(
        np.linspace(1.05, 0.95, 4).tolist()
    )
    assert actual[0].sum() == pytest.approx(4.0)


def test_midpoint_is_strictly_closer_to_equalweight_than_current() -> None:
    assert abs(MIDPOINT_TOP - 1.0) < abs(1.10 - 1.0)
    assert abs(MIDPOINT_BOTTOM - 1.0) < abs(0.90 - 1.0)
