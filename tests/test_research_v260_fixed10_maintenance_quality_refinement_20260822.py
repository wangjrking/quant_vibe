from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


MAIN_ROOT = Path(r"D:/work/quant/quant_mcp/quant/main")
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_maintenance_quality_refinement_20260822 as module


def test_candidate_thresholds_reuse_existing_values() -> None:
    assert module.CANDIDATE_THRESHOLDS == (0.80, 0.85)
    assert module.candidate_id(0.80) == "maintenance_topup_score_080"
    assert module.candidate_id(0.85) == "maintenance_topup_score_085"


def test_maintenance_block_is_monotone() -> None:
    score = np.asarray([[0.79, 0.80, 0.84, 0.85, np.nan]])
    loose = module.maintenance_block(score, 0.80)
    strict = module.maintenance_block(score, 0.85)
    assert np.all(strict | ~loose)
    np.testing.assert_array_equal(loose, [[True, False, False, False, True]])
    np.testing.assert_array_equal(strict, [[True, True, True, False, True]])
