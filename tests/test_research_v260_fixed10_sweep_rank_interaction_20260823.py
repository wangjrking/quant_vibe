from __future__ import annotations

import inspect

import research_v260_fixed10_rank_sizing_residual_cash_sweep_20260823 as sweep


def test_rank_sizing_is_enabled_by_default() -> None:
    assert inspect.signature(sweep.run_with_runtime).parameters[
        "rank_sizing"
    ].default is True


def test_interaction_has_only_two_declared_costs() -> None:
    import research_v260_fixed10_sweep_rank_interaction_20260823 as module

    assert module.sizing.BASELINE_COST == 0.003
    assert module.sizing.STRESS_COST == 0.0065
