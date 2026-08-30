from __future__ import annotations

import numpy as np
import pytest

from research_v260_fixed10_risk_event_minimal_policy_20260822 import (
    compose_maintenance_block,
    compose_minimal_masks,
)


def test_minimal_policy_ignores_ordinary_in_primary_overlay() -> None:
    ordinary = np.array([[True, False, False]], dtype=bool)
    severe = np.array([[False, True, False]], dtype=bool)
    alert = np.array([[False, False, True]], dtype=bool)

    masks = compose_minimal_masks(ordinary, severe, alert)

    assert masks["severe_and_alert_next_session_only"].tolist() == [
        [False, True, True]
    ]
    assert masks["all_events_next_session_control"].tolist() == [
        [True, True, True]
    ]


def test_minimal_policy_off_is_empty_and_independent() -> None:
    ordinary = np.ones((2, 2), dtype=bool)
    severe = np.zeros((2, 2), dtype=bool)
    alert = np.zeros((2, 2), dtype=bool)

    masks = compose_minimal_masks(ordinary, severe, alert)
    masks["risk_event_off"][0, 0] = True

    assert not masks["severe_next_session_only"].any()
    assert not masks["severe_and_alert_next_session_only"].any()


def test_minimal_policy_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="do not align"):
        compose_minimal_masks(
            np.zeros((1, 2), dtype=bool),
            np.zeros((2, 1), dtype=bool),
            np.zeros((1, 2), dtype=bool),
        )


def test_maintenance_block_combines_quality_and_event_once() -> None:
    score = np.array([[0.90, 0.70, np.nan]], dtype=float)
    event = np.array([[True, False, False]], dtype=bool)

    actual = compose_maintenance_block(score, 0.80, event)

    assert actual.tolist() == [[True, True, True]]
