from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np


MAIN_ROOT = Path(r"D:/work/quant/quant_mcp/quant/main")
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_exit_short_term_confirmation_20260822 as module
import research_v260_fixed10_hold_optimization_20260822 as round1
from research_v260_runtime import fixed10_risk_event_v110 as runtime


def test_candidate_budget_is_binary() -> None:
    assert module.CANDIDATE_IDS == (
        "production_score_exit",
        "score_exit_requires_rank1d_below_050",
    )


def test_candidate_confirmation_uses_strict_below_median() -> None:
    rank_1d = np.array([[np.nan, 0.49, 0.50, 0.90]], dtype=np.float64)
    baseline = module.score_exit_confirmation_mask(
        "production_score_exit", rank_1d
    )
    candidate = module.score_exit_confirmation_mask(
        "score_exit_requires_rank1d_below_050", rank_1d
    )
    assert baseline.tolist() == [[True, True, True, True]]
    assert candidate.tolist() == [[False, True, False, False]]


def test_unknown_candidate_fails_closed() -> None:
    try:
        module.score_exit_confirmation_mask("unknown", np.zeros((1, 1)))
    except ValueError as exc:
        assert "unknown candidate" in str(exc)
    else:
        raise AssertionError("unknown candidate must fail closed")


def test_runtime_and_wrapper_default_to_existing_behavior() -> None:
    runtime_parameter = inspect.signature(runtime.simulate).parameters[
        "score_exit_confirmation_mask_override"
    ]
    wrapper_parameter = inspect.signature(round1.run_fixed10).parameters[
        "score_exit_confirmation_mask_override"
    ]
    assert runtime_parameter.default is None
    assert wrapper_parameter.default is None
    assert inspect.signature(runtime.simulate).parameters["exit_score_override"].default is None
    assert inspect.signature(round1.run_fixed10).parameters["exit_score_override"].default is None
