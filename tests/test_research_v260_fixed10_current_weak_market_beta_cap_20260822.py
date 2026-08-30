import sys
from pathlib import Path

import numpy as np


MAIN = Path(__file__).resolve().parents[1]
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))

from research_v260_fixed10_current_weak_market_beta_cap_20260822 import candidate_mask


def test_beta_cap_applies_only_on_weak_market_days():
    base = np.ones((2, 3), dtype=np.bool_)
    beta = np.asarray([[1.0, 2.0, np.nan], [1.0, 2.0, np.nan]])
    actual = candidate_mask(base, beta, [False, True])
    assert actual.tolist() == [[True, True, True], [True, False, True]]
