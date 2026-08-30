from __future__ import annotations

import sys
from pathlib import Path


MAIN_ROOT = Path(r"D:/work/quant/quant_mcp/quant/main")
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_confirmed_weak_vol_priority_boundary_20260822 as subject


def test_candidate_ids_are_stable() -> None:
    assert subject.candidate_id(0.0) == "confirmed_weak_vol_penalty_0.00"
    assert subject.candidate_id(0.1) == "confirmed_weak_vol_penalty_0.10"
