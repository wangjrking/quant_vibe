from __future__ import annotations

import numpy as np

import research_v260_fixed10_tushare_risk_event_entry_gate_20260822 as module


def test_overlay_masks_keep_ordinary_event_entry_only() -> None:
    ordinary = np.array([[True, False, False]], dtype=np.bool_)
    severe = np.array([[False, True, False]], dtype=np.bool_)
    alert = np.array([[False, False, True]], dtype=np.bool_)

    entry, maintenance = module.compose_overlay_masks({
        "ordinary_1d": ordinary,
        "severe_5d": severe,
        "alert_active": alert,
    })

    assert entry.tolist() == [[True, True, True]]
    assert maintenance.tolist() == [[False, True, True]]


def test_overlay_mask_composition_does_not_mutate_components() -> None:
    ordinary = np.array([[True, False]], dtype=np.bool_)
    severe = np.array([[False, True]], dtype=np.bool_)
    alert = np.zeros((1, 2), dtype=np.bool_)
    original = [array.copy() for array in (ordinary, severe, alert)]

    module.compose_overlay_masks({
        "ordinary_1d": ordinary,
        "severe_5d": severe,
        "alert_active": alert,
    })

    for actual, expected in zip((ordinary, severe, alert), original):
        assert np.array_equal(actual, expected)
