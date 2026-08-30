import sys
from pathlib import Path

import numpy as np


MAIN = Path(__file__).resolve().parents[1]
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))

from research_v260_fixed10_severe_event_exit_deferral_20260822 import confirmation_mask


def test_deferral_disables_only_exact_severe_keys():
    severe = np.array([[True, False], [False, True]])
    assert confirmation_mask(severe, True).tolist() == [[False, True], [True, False]]


def test_disabled_deferral_allows_all_score_exits():
    severe = np.array([[True, False]])
    assert confirmation_mask(severe, False).tolist() == [[True, True]]
