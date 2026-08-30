import numpy as np

from quant.main.research_v260_fixed10_event_entry_block_robustness_20260822 import (
    distribution,
)


def test_distribution_is_deterministic_and_order_independent() -> None:
    forward = distribution([1.0, 2.0, 3.0, 4.0, 5.0])
    reverse = distribution([5.0, 4.0, 3.0, 2.0, 1.0])
    assert forward == reverse
    assert forward["min"] == 1.0
    assert forward["median"] == 3.0
    assert forward["max"] == 5.0


def test_distribution_accepts_generator() -> None:
    result = distribution(float(value) for value in range(3))
    assert np.isclose(result["median"], 1.0)
