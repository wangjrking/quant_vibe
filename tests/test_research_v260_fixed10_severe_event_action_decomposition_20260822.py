import sys
from pathlib import Path

import numpy as np
import pytest


MAIN = Path(__file__).resolve().parents[1]
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))

from research_v260_fixed10_severe_event_action_decomposition_20260822 import compose_masks


def test_entry_and_topup_scopes_are_independent():
    severe = np.array([[True, False]])
    quality = np.array([[False, True]])
    entry, topup = compose_masks(severe, quality, True, False)
    assert entry.tolist() == [[True, False]]
    assert topup.tolist() == [[False, True]]
    entry, topup = compose_masks(severe, quality, False, True)
    assert entry.tolist() == [[False, False]]
    assert topup.tolist() == [[True, True]]


def test_misaligned_masks_fail_closed():
    with pytest.raises(ValueError, match="align"):
        compose_masks(np.zeros((1, 2)), np.zeros((2, 1)), True, True)
