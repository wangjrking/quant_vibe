from __future__ import annotations

import sys
from pathlib import Path


MAIN_ROOT = Path(r"D:/work/quant/quant_mcp/quant/main")
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_score_smoothing_current_rules_20260822 as module


def test_candidate_budget_is_local_and_coarse() -> None:
    assert module.CANDIDATE_WINDOWS == {
        "smooth_5d": 5,
        "smooth_7d": 7,
        "smooth_10d": 10,
    }


def test_unknown_candidate_fails_closed() -> None:
    class Harness:
        class v95:
            @staticmethod
            def score_pair(*args):
                raise AssertionError("must not run")

    try:
        module.candidate_score(Harness(), {}, "unknown")
    except ValueError as exc:
        assert "unknown candidate" in str(exc)
    else:
        raise AssertionError("unknown candidate must fail closed")
