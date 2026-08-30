import sys
from pathlib import Path

import numpy as np


MAIN = Path(__file__).resolve().parents[1]
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))

from research_v260_fixed10_entry_score_1d70_10d30_current_rules_20260822 import (
    candidate_passes,
    entry_score,
)


def test_entry_score_uses_fixed_rank_mix():
    arrays = {
        "rank_1d": np.array([[1.0, 0.0]], dtype=np.float64),
        "rank_10d": np.array([[0.0, 1.0]], dtype=np.float64),
    }
    score, order = entry_score(arrays, 0.70)
    assert score.tolist() == [[0.699999988079071, 0.30000001192092896]]
    assert order.tolist() == [[0, 1]]


def test_candidate_requires_train_holdout_and_stress_to_all_pass():
    base = {"cagr": 0.2, "sharpe": 1.0, "max_drawdown": 0.3}
    better = {"cagr": 0.21, "sharpe": 1.1, "max_drawdown": 0.29}
    hold_base = {"cumulative_return": 0.2, "sharpe": 1.0}
    hold_better = {"cumulative_return": 0.21, "sharpe": 1.1}
    stress_base = {"cumulative_return": 0.1}
    stress_better = {"cumulative_return": 0.11}
    assert candidate_passes(
        base, better, hold_base, hold_better, stress_base, stress_better
    )["passed"]
    hold_worse = {"cumulative_return": 0.19, "sharpe": 1.1}
    assert not candidate_passes(
        base, better, hold_base, hold_worse, stress_base, stress_better
    )["passed"]
