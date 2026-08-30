from __future__ import annotations

import numpy as np
import pytest

from research_v260_fixed10_friction_compensated_gross_20260823 import (
    friction_compensated_entry_multipliers,
    friction_compensated_gross,
)


def test_formula_exactly_compensates_declared_buy_friction() -> None:
    gross = friction_compensated_gross(0.003)
    assert gross == pytest.approx(1.003 * 1.0003)
    assert gross / ((1.0 + 0.003) * (1.0 + 0.0003)) == pytest.approx(1.0)


def test_higher_disclosed_slippage_has_higher_compensation() -> None:
    assert friction_compensated_gross(0.0065) > friction_compensated_gross(0.003)


def test_compensation_scales_new_entry_budget_without_changing_rank_order() -> None:
    order = np.array([[0, 1, 2, 3]], dtype=np.int64)
    actual = friction_compensated_entry_multipliers(order, 0.003, positions=4)
    assert actual[0, 0] > actual[0, 1] > actual[0, 2] > actual[0, 3]
    assert actual[0].sum() == pytest.approx(4.0 * friction_compensated_gross(0.003))


def test_invalid_slippage_is_rejected() -> None:
    with pytest.raises(ValueError, match="slippage"):
        friction_compensated_gross(-0.01)
    with pytest.raises(ValueError, match="slippage"):
        friction_compensated_gross(1.0)
