from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np


MAIN_ROOT = Path(r"D:/work/quant/quant_mcp/quant/main")
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_entry_gap_down_guard_20260822 as module
from research_v260_runtime import fixed10_risk_event_v110 as runtime


def test_only_minus_five_percent_case_is_selectable() -> None:
    selectable = [key for key, item in module.CASES.items() if item["selectable"]]
    assert selectable == ["candidate_min_entry_gap_m005"]


def test_runtime_minimum_gap_override_is_optional() -> None:
    parameter = inspect.signature(runtime.simulate).parameters[
        "min_entry_open_gap_override"
    ]
    assert parameter.default is None


def test_entry_gap_down_boundary_and_missing_values() -> None:
    result = runtime.entry_open_gap_allowed(
        np.array([95.0, 94.99, 105.0, np.nan]),
        np.array([100.0, 100.0, 100.0, 100.0]),
        minimum_gap=-0.05,
    )
    assert np.array_equal(result, [True, False, True, False])
