from __future__ import annotations

import inspect
import sys
from pathlib import Path


MAIN_ROOT = Path(r"D:/work/quant/quant_mcp/quant/main")
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_absolute_hold_cap_20260822 as module
import research_v260_fixed10_hold_optimization_20260822 as round1
from research_v260_runtime import fixed10_risk_event_v110 as runtime


def test_candidate_budget_is_binary_and_cap_is_two_cycles() -> None:
    assert module.CANDIDATES == (
        "unlimited_renewals",
        "absolute_50_session_cap",
    )
    assert module.ABSOLUTE_CAP_DAYS == 50


def test_absolute_cap_mapping_is_explicit() -> None:
    assert module.absolute_cap("unlimited_renewals") is None
    assert module.absolute_cap("absolute_50_session_cap") == 50


def test_runtime_default_keeps_existing_behavior() -> None:
    parameter = inspect.signature(runtime.simulate).parameters[
        "absolute_max_hold_days_override"
    ]
    assert parameter.default is None
    wrapper_parameter = inspect.signature(round1.run_fixed10).parameters[
        "absolute_max_hold_days_override"
    ]
    assert wrapper_parameter.default is None
