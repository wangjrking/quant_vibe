from __future__ import annotations

from inspect import signature

from quant.main.research_v260_fixed10_limitup_exit_deferral_20260822 import CANDIDATES
from quant.main.research_v260_runtime import fixed10_risk_event_v110


def test_candidate_budget_is_binary() -> None:
    assert CANDIDATES == ("sell_as_planned", "defer_sell_when_open_limit_up")


def test_limit_up_exit_deferral_defaults_to_disabled() -> None:
    parameter = signature(fixed10_risk_event_v110.simulate).parameters[
        "defer_sell_on_limit_up_override"
    ]
    assert parameter.default is False
