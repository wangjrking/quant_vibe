from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


MAIN_ROOT = Path(r"D:/work/quant/quant_mcp/quant/main")
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_weak_market_exit_capacity_current_20260822 as module


def test_candidate_budget_is_binary_and_predeclared() -> None:
    assert module.CANDIDATES == (
        "one_exit_all_markets",
        "two_exits_only_in_weak_market",
    )


def test_exit_limit_schedule_changes_only_weak_days(monkeypatch) -> None:
    score = np.zeros((4, 3), dtype=float)
    monkeypatch.setattr(
        module.regime,
        "strong_market_mask",
        lambda _score, _protocol: np.asarray([True, False, True, False]),
    )
    actual = module.exit_limit_schedule(score, {}, 2)
    np.testing.assert_array_equal(actual, np.asarray([1, 2, 1, 2], dtype=np.int16))


def test_baseline_schedule_is_one_for_every_day(monkeypatch) -> None:
    score = np.zeros((3, 2), dtype=float)
    monkeypatch.setattr(
        module.regime,
        "strong_market_mask",
        lambda _score, _protocol: np.asarray([True, False, False]),
    )
    actual = module.exit_limit_schedule(score, {}, 1)
    np.testing.assert_array_equal(actual, np.ones(3, dtype=np.int16))
