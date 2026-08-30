from __future__ import annotations

import copy
import sys
from pathlib import Path


MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_current_reentry_profit_optimization_20260822 as mod


def test_cooldown_candidates_are_small_and_interpretable():
    assert mod.COOLDOWN_DAYS == (0, 1, 3)


def test_run_case_does_not_mutate_base_policy(monkeypatch):
    captured = {}
    monkeypatch.setattr(mod.width, "build_overrides", lambda context, policy: (4, None))

    def fake_run(context, policy, extra_age, cost, **kwargs):
        captured.update(policy)
        return "daily", "actions"

    monkeypatch.setattr(mod.age_guard, "run_policy_at_cost", fake_run)
    policy = {"reentry_cooldown_days": 0, "other": "unchanged"}
    original = copy.deepcopy(policy)
    assert mod.run_case(object(), policy, 3, 0.003) == ("daily", "actions")
    assert policy == original
    assert captured["reentry_cooldown_days"] == 3
    assert captured["other"] == "unchanged"
