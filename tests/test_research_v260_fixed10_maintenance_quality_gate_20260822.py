from __future__ import annotations

import numpy as np

from quant.main.research_v260_fixed10_maintenance_quality_gate_20260822 import (
    CANDIDATES,
    MAINTENANCE_SCORE_THRESHOLD,
    maintenance_quality_block,
)


def test_candidate_budget_is_binary() -> None:
    assert CANDIDATES == ("current_topup", "topup_requires_score_080")


def test_threshold_reuses_existing_renewal_score() -> None:
    assert MAINTENANCE_SCORE_THRESHOLD == 0.80


def test_disabled_gate_blocks_nothing() -> None:
    score = np.asarray([[0.10, np.nan, 0.90]])
    assert not maintenance_quality_block(score, False).any()


def test_active_gate_blocks_weak_or_invalid_scores() -> None:
    score = np.asarray([[0.79, 0.80, 0.81, np.nan]])
    block = maintenance_quality_block(score, True)
    assert block.tolist() == [[True, False, False, True]]
