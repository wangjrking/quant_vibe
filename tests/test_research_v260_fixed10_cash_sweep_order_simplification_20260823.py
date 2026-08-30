from __future__ import annotations

import numpy as np
import pytest

from research_v260_runtime import fixed10_residual_cash_sweep_v112 as runtime


def inputs() -> dict:
    return {
        "stocks": np.asarray(["000001.SZ", "000002.SZ"]),
        "shares": {0: 800.0, 1: 800.0},
        "raw_opens": np.asarray([10.0, 10.0]),
        "mark_prices": np.asarray([10.0, 10.0]),
        "eligible_indices": {0, 1},
        "cash": 2200.0,
        "target_pct": 0.5,
        "slippage_ratio": 0.0,
    }


def test_one_order_limit_only_tops_up_one_holding() -> None:
    _, additions = runtime.residual_cash_sweep_plan(
        **inputs(), max_additions=1
    )
    assert len(additions) == 1
    assert sum(additions.values()) == 100.0


def test_unlimited_sweep_can_top_up_both_holdings() -> None:
    _, additions = runtime.residual_cash_sweep_plan(**inputs())
    assert set(additions) == {0, 1}


def test_nonpositive_order_limit_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        runtime.residual_cash_sweep_plan(**inputs(), max_additions=0)
