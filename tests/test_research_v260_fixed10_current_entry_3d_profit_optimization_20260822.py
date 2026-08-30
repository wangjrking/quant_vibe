from __future__ import annotations

import sys
from pathlib import Path


MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_current_entry_3d_profit_optimization_20260822 as mod


def test_entry_3d_candidates_are_broad_and_include_current():
    assert mod.ENTRY_3D_WEIGHTS == (0.0, 0.1, 0.3, 0.5)


def test_entry_order_endpoints_use_declared_horizons(monkeypatch):
    arrays = {
        "rank_3d": [[0.1, 0.9]],
        "rank_10d": [[0.8, 0.2]],
    }

    monkeypatch.setattr(
        mod.score_tools,
        "smooth_score",
        lambda raw, window, alpha: (raw, raw),
    )

    assert mod.entry_order(arrays, 0.0).tolist() == [[0.8, 0.2]]
    assert mod.entry_order(arrays, 1.0).tolist() == [[0.1, 0.9]]
