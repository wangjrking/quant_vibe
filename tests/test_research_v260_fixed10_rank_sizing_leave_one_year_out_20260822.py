from __future__ import annotations

import sys
from pathlib import Path


MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_rank_sizing_leave_one_year_out_20260822 as mod


def test_compounded_excluding_removes_only_named_year() -> None:
    annual = {"2022": 0.1, "2023": 0.2, "2024": 0.3, "2025": 0.4}
    actual = mod.compounded_excluding(annual, "2024")
    assert abs(actual - ((1.1 * 1.2 * 1.4) - 1.0)) < 1e-12


def test_compounded_excluding_rejects_missing_retained_year() -> None:
    annual = {"2022": 0.1, "2023": 0.2, "2024": 0.3}
    try:
        mod.compounded_excluding(annual, "2024")
    except KeyError:
        pass
    else:
        raise AssertionError("missing retained year was accepted")
