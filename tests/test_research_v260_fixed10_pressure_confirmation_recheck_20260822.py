from quant.main.research_v260_fixed10_pressure_confirmation_recheck_20260822 import (
    CONFIRMATION_DAYS,
)


def test_confirmation_budget_is_short_and_bounded():
    assert CONFIRMATION_DAYS == (1, 2, 3)
