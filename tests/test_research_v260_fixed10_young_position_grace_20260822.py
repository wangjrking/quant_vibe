from __future__ import annotations

import inspect
import sys
from pathlib import Path


MAIN_ROOT = Path(r"D:/work/quant/quant_mcp/quant/main")
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_young_position_grace_20260822 as module
from research_v260_runtime import fixed10_risk_event_v110 as runtime


def test_candidate_budget_is_binary() -> None:
    assert module.CANDIDATES == (
        "constant_exit_085",
        "young_exit_080_then_085",
    )


def test_age_bands_reuse_existing_thresholds() -> None:
    assert module.threshold_age_bands("constant_exit_085") is None
    assert module.threshold_age_bands("young_exit_080_then_085") == [
        (0, 0.80),
        (10, 0.85),
    ]


def test_runtime_and_wrapper_default_to_no_age_override() -> None:
    runtime_parameter = inspect.signature(runtime.simulate).parameters[
        "sell_score_below_age_bands_override"
    ]
    wrapper_parameter = inspect.signature(round1.run_fixed10).parameters[
        "sell_score_below_age_bands_override"
    ]
    assert runtime_parameter.default is None
    assert wrapper_parameter.default is None
