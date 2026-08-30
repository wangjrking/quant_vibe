from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


MAIN_ROOT = Path(r"D:/work/quant/quant_mcp/quant/main")
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_current_risk_event_rules_20260822 as subject


def test_compose_entry_blocks_keeps_events_entry_only() -> None:
    score = np.array([[0.90, 0.80, 0.70]], dtype=float)
    ordinary = np.array([[True, False, False]])
    severe = np.array([[False, True, False]])
    alert = np.array([[False, False, True]])

    result = subject.compose_entry_blocks(score, ordinary, severe, alert)

    assert not result["risk_event_off"].any()
    assert result["severe_pause5"].tolist() == [[False, True, False]]
    assert result["severe_pause5_alert_quality"].tolist() == [[False, True, True]]
    assert result["severe_pause5_ordinary_alert_quality"].tolist() == [
        [False, True, True]
    ]


def test_ordinary_event_does_not_block_high_quality_entry() -> None:
    score = np.array([[0.85, 0.849999]], dtype=float)
    ordinary = np.ones(score.shape, dtype=bool)
    empty = np.zeros(score.shape, dtype=bool)

    result = subject.compose_entry_blocks(score, ordinary, empty, empty)

    assert result["severe_pause5_ordinary_alert_quality"].tolist() == [
        [False, True]
    ]


def test_compose_entry_blocks_fails_closed_on_shape_mismatch() -> None:
    score = np.ones((2, 2), dtype=float)
    bad = np.ones((1, 2), dtype=bool)
    good = np.zeros((2, 2), dtype=bool)

    with pytest.raises(ValueError, match="do not align"):
        subject.compose_entry_blocks(score, bad, good, good)


def test_selection_gate_requires_broad_direct_evidence() -> None:
    metrics = {
        "train_2022_2024": {"cagr": 0.2, "sharpe": 1.0},
        "holdout_2025": {"cumulative_return": 0.3},
        "metrics_0_65pct": {"cumulative_return": 0.4},
        "metrics_0_30pct": {
            "max_drawdown": 0.2,
            "full_10_position_ratio": 1.0,
        },
    }
    candidate = {
        **metrics,
        "train_2022_2024": {"cagr": 0.21, "sharpe": 1.01},
    }
    direct = {"distinct_signal_date_count": 2, "affected_years": ["2024"]}

    gates = subject.selection_gates(candidate, metrics, direct)

    assert gates["evidence_spans_three_dates"] is False
    assert gates["evidence_spans_two_years"] is False
