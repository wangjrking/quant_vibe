from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


MAIN_ROOT = Path(r"D:/work/quant/quant_mcp/quant/main")
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_position_correlation_diagnostic_20260822 as module


def test_returns_do_not_use_future_rows() -> None:
    close = np.array([[1.0], [2.0], [4.0]])
    first = module.returns_from_close(close)
    changed = close.copy()
    changed[-1] = 100.0
    second = module.returns_from_close(changed)
    assert np.allclose(first[:-1], second[:-1], equal_nan=True)


def test_pairwise_correlation_detects_opposite_series() -> None:
    returns = np.array(
        [
            [0.0, 4.0],
            [1.0, 3.0],
            [2.0, 2.0],
            [3.0, 1.0],
            [4.0, 0.0],
        ]
    )
    result = module.pairwise_correlations(returns, 4, [0, 1], 5, 5)
    assert len(result) == 1
    assert np.isclose(result[0], -1.0)
