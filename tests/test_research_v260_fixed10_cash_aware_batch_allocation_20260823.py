from __future__ import annotations

import numpy as np
import pytest

from research_v260_fixed10_cash_aware_batch_allocation_20260823 import (
    allocate_cash_aware_lots,
)


def allocate(targets, opens, cash):
    return allocate_cash_aware_lots(
        candidates=np.arange(len(targets), dtype=np.int64),
        target_pcts=np.asarray(targets, dtype=np.float64),
        raw_opens=np.asarray(opens, dtype=np.float64),
        equity_before=100_000.0,
        cash_available=float(cash),
        slippage_ratio=0.003,
    )


def test_full_cash_returns_independent_target_lots() -> None:
    actual = allocate([0.10, 0.10], [10.0, 20.0], 100_000.0)
    assert actual.tolist() == pytest.approx([900.0, 400.0])


def test_cash_shortfall_is_shared_instead_of_skipping_the_last_name() -> None:
    actual = allocate([0.40, 0.40, 0.40], [10.0, 10.0, 10.0], 50_000.0)
    assert (actual > 0.0).all()
    spend = float(actual.sum() * 10.0 * 1.003 * 1.0003)
    assert spend <= 50_000.0 + 1e-9
    assert actual.max() - actual.min() <= 100.0


def test_allocation_is_deterministic_and_uses_board_lots() -> None:
    first = allocate([0.11, 0.10, 0.09], [9.0, 11.0, 13.0], 25_000.0)
    second = allocate([0.11, 0.10, 0.09], [9.0, 11.0, 13.0], 25_000.0)
    assert np.array_equal(first, second)
    assert np.allclose(np.mod(first, 100.0), 0.0)


def test_invalid_inputs_fail_closed() -> None:
    with pytest.raises(ValueError, match="aligned"):
        allocate_cash_aware_lots(
            candidates=np.arange(2),
            target_pcts=np.array([0.1, 0.1]),
            raw_opens=np.array([10.0]),
            equity_before=100_000.0,
            cash_available=100_000.0,
            slippage_ratio=0.003,
        )
