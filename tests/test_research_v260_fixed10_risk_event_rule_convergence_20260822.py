import sys
from pathlib import Path

import numpy as np
import pytest


MAIN = Path(__file__).resolve().parents[1]
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))

from research_v260_fixed10_risk_event_rule_convergence_20260822 import (
    compose_entry_masks,
    selection_gates,
)


def test_event_rules_keep_severe_hard_and_condition_other_events_on_existing_threshold():
    score = np.array([[0.90, 0.84, 0.70]])
    ordinary = np.array([[True, True, False]])
    severe = np.array([[False, False, True]])
    warning = np.array([[True, True, False]])
    masks = compose_entry_masks(score, ordinary, severe, warning, 0.85)
    assert masks["risk_event_off"].tolist() == [[False, False, False]]
    assert masks["severe_pause5"].tolist() == [[False, False, True]]
    assert masks["severe_pause5_warning_quality"].tolist() == [[False, True, True]]
    assert masks["severe_pause5_ordinary_warning_quality"].tolist() == [
        [False, True, True]
    ]


def test_event_mask_shape_mismatch_fails_closed():
    with pytest.raises(ValueError, match="align"):
        compose_entry_masks(
            np.zeros((1, 2)),
            np.zeros((1, 1), dtype=bool),
            np.zeros((1, 2), dtype=bool),
            np.zeros((1, 2), dtype=bool),
            0.85,
        )


def test_selection_requires_multiple_direct_episodes():
    base_metrics = {
        "train_2022_2024": {"sharpe": 1.0, "cagr": 0.4},
        "holdout_2025": {"cumulative_return": 0.5},
        "metrics_0_65pct": {"cumulative_return": 1.0},
        "metrics_0_30pct": {"max_drawdown": 0.3, "full_10_position_ratio": 1.0},
    }
    candidate = {
        "train_2022_2024": {"sharpe": 1.1, "cagr": 0.5},
        "holdout_2025": {"cumulative_return": 0.6},
        "metrics_0_65pct": {"cumulative_return": 1.1},
        "metrics_0_30pct": {"max_drawdown": 0.2, "full_10_position_ratio": 1.0},
    }
    thin = {
        "distinct_signal_date_count": 2,
        "affected_years": ["2024", "2025"],
    }
    broad = {
        "distinct_signal_date_count": 3,
        "affected_years": ["2024", "2025"],
    }
    assert not selection_gates(candidate, base_metrics, thin)[
        "direct_evidence_not_single_episode"
    ]
    assert selection_gates(candidate, base_metrics, broad)[
        "direct_evidence_not_single_episode"
    ]
