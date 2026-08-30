from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


MAIN_ROOT = Path(r"D:/work/quant/quant_mcp/quant/main")
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_score_5d10d_current_rules_20260822 as module


def test_candidate_budget_is_binary() -> None:
    assert module.CANDIDATE_IDS == (
        "score_10d",
        "score_5d50_10d50",
        "score_5d50_10d50_entry_only",
    )


def test_smoothing_is_past_only_and_deterministic() -> None:
    raw = np.array([[0.0], [1.0], [0.0]], dtype=np.float64)
    score, order = module.smooth_score(raw, window=2, alpha=0.0)
    assert np.allclose(score[:, 0], [0.0, 0.5, 0.5])
    assert order.tolist() == [[0], [0], [0]]


def test_score_formula_is_exact_equal_weight() -> None:
    arrays = {
        "rank_5d": np.array([[0.2, 0.8]], dtype=np.float64),
        "rank_10d": np.array([[0.6, 0.4]], dtype=np.float64),
    }
    score, order = module.candidate_score(
        "score_5d50_10d50",
        arrays,
        np.zeros((1, 2), dtype=np.float32),
        np.zeros((1, 2), dtype=np.int64),
    )
    assert np.allclose(score[0], [0.4, 0.6])
    assert order.tolist() == [[1, 0]]


def test_unknown_candidate_fails_closed() -> None:
    try:
        module.candidate_score(
            "unknown",
            {"rank_5d": np.zeros((1, 1)), "rank_10d": np.zeros((1, 1))},
            np.zeros((1, 1)),
            np.zeros((1, 1), dtype=np.int64),
        )
    except ValueError as exc:
        assert "unknown candidate" in str(exc)
    else:
        raise AssertionError("unknown candidate must fail closed")


def test_entry_only_candidate_uses_same_entry_score() -> None:
    arrays = {
        "rank_5d": np.array([[0.2, 0.8]], dtype=np.float64),
        "rank_10d": np.array([[0.6, 0.4]], dtype=np.float64),
    }
    full_score, full_order = module.candidate_score(
        "score_5d50_10d50", arrays, np.zeros((1, 2)), np.zeros((1, 2))
    )
    entry_score, entry_order = module.candidate_score(
        "score_5d50_10d50_entry_only",
        arrays,
        np.zeros((1, 2)),
        np.zeros((1, 2)),
    )
    assert np.array_equal(full_score, entry_score)
    assert np.array_equal(full_order, entry_order)
