from __future__ import annotations

import numpy as np
import pytest

import research_v260_fixed10_rank_sizing_neutral_placebo_20260823 as module


def test_neutral_schedule_preserves_dispersion_and_gross() -> None:
    schedule = module.neutral_schedule()
    assert schedule.sum() == pytest.approx(10.0)
    assert schedule.min() == pytest.approx(0.9)
    assert schedule.max() == pytest.approx(1.1)


def test_neutral_schedule_has_near_zero_rank_correlation() -> None:
    correlation = np.corrcoef(np.arange(10), module.neutral_schedule())[0, 1]
    assert abs(correlation) < 0.007


def test_schedule_multipliers_follow_global_rank_order() -> None:
    order = np.array([[2, 0, 1, 3, 4, 5, 6, 7, 8, 9]], dtype=np.int64)
    schedule = module.neutral_schedule()
    result = module.schedule_multipliers(order, schedule)
    assert result[0, 2] == pytest.approx(schedule[0])
    assert result[0, 0] == pytest.approx(schedule[1])


def test_direction_bootstrap_uses_broad_fixed_blocks() -> None:
    assert module.BLOCK_LENGTHS == (5, 20, 60)


def test_temporal_direction_attribution_keeps_annual_disagreement_visible() -> None:
    result = module.temporal_direction_attribution(
        np.array(["20220104", "20220105", "20230103", "20230104"]),
        np.array([0.10, 0.0, -0.10, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0]),
    )
    assert result["annual"]["2022"]["forward_minus_neutral"] > 0.0
    assert result["annual"]["2023"]["forward_minus_neutral"] < 0.0
    assert result["positive_annual_count"] == 1
