from __future__ import annotations

import sys
from pathlib import Path


MAIN_ROOT = Path(r"D:/work/quant/quant_mcp/quant/main")
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_weak_market_size_guard_confirmation_20260822 as module


def test_only_ten_percent_case_is_selectable() -> None:
    selectable = [key for key, item in module.CASES.items() if item["selectable"]]
    assert selectable == ["candidate_bottom_10pct"]


def test_controls_bracket_candidate() -> None:
    assert module.CASES["control_bottom_05pct"]["percentile"] < 0.10
    assert module.CASES["control_bottom_15pct"]["percentile"] > 0.10
