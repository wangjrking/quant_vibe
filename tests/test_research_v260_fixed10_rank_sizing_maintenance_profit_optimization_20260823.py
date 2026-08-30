from __future__ import annotations

import inspect

import pytest

import research_v260_fixed10_rank_sizing_maintenance_profit_optimization_20260823 as module


def test_runtime_hook_defaults_to_disabled() -> None:
    parameter = inspect.signature(module.sizing.runtime.simulate).parameters[
        "maintenance_target_multiplier_override"
    ]
    assert parameter.default is None


def test_selection_uses_net_cumulative_return_only() -> None:
    metrics = {
        "new_entry_only": {"cumulative_return": 1.0},
        "new_entry_and_maintenance": {"cumulative_return": 1.1},
    }
    assert module.select_by_pre2026_net_return(metrics) == "new_entry_and_maintenance"


def test_selection_keeps_current_when_it_earns_more() -> None:
    metrics = {
        "new_entry_only": {"cumulative_return": 1.2},
        "new_entry_and_maintenance": {"cumulative_return": 1.1},
    }
    assert module.select_by_pre2026_net_return(metrics) == "new_entry_only"


def test_selection_rejects_extra_arms() -> None:
    with pytest.raises(ValueError):
        module.select_by_pre2026_net_return(
            {
                "new_entry_only": {"cumulative_return": 1.0},
                "new_entry_and_maintenance": {"cumulative_return": 1.1},
                "third": {"cumulative_return": 2.0},
            }
        )
