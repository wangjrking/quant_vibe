from quant.main.research_v260_fixed10_pressure_exit_limit_recheck_20260822 import (
    EXIT_LIMITS,
)


def test_exit_limit_budget_is_small_and_ordered():
    assert EXIT_LIMITS == (1, 2, 3)
