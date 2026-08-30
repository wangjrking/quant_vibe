from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np


MAIN_ROOT = Path(r"D:/work/quant/quant_mcp/quant/main")
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_entry_open_gap_guard_20260822 as module
from research_v260_runtime import fixed10_risk_event_v110 as runtime


def test_only_five_percent_gap_case_is_selectable() -> None:
    selectable = [key for key, item in module.CASES.items() if item["selectable"]]
    assert selectable == ["candidate_max_entry_gap_005"]


def test_runtime_gap_override_is_optional() -> None:
    parameter = inspect.signature(runtime.simulate).parameters[
        "max_entry_open_gap_override"
    ]
    assert parameter.default is None


def test_entry_open_gap_boundary_and_missing_values() -> None:
    result = runtime.entry_open_gap_allowed(
        np.array([105.0, 105.01, 95.0, np.nan]),
        np.array([100.0, 100.0, 100.0, 100.0]),
        0.05,
    )
    assert np.array_equal(result, [True, False, True, False])
