from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


MAIN_ROOT = Path(r"D:/work/quant/quant_mcp/quant/main")
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_score_5d_minor_blend_20260822 as module


def test_candidate_weights_are_coarse_and_bounded() -> None:
    assert module.SELECTABLE_WEIGHTS_5D == {
        "score_10d": 0.0,
        "score_5d10_10d90": 0.10,
        "score_5d20_10d80": 0.20,
    }
    assert module.ROBUSTNESS_WEIGHTS_5D == {
        "control_score_5d30_10d70": 0.30
    }


def test_minor_blend_formula_is_exact() -> None:
    arrays = {
        "rank_5d": np.array([[0.0, 1.0]]),
        "rank_10d": np.array([[1.0, 0.0]]),
    }
    score, order = module.candidate_score(
        "score_5d20_10d80",
        arrays,
        np.zeros((1, 2)),
        np.zeros((1, 2), dtype=np.int64),
    )
    assert np.allclose(score[0], [0.8, 0.2])
    assert order.tolist() == [[0, 1]]


def test_unknown_candidate_fails_closed() -> None:
    try:
        module.candidate_score("unknown", {}, np.zeros((1, 1)), np.zeros((1, 1)))
    except ValueError as exc:
        assert "unknown candidate" in str(exc)
    else:
        raise AssertionError("unknown candidate must fail closed")
