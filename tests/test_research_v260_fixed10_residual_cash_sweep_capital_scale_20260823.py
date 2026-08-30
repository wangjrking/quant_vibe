from __future__ import annotations

from dataclasses import dataclass

import pytest

import research_v260_fixed10_residual_cash_sweep_capital_scale_20260823 as module


@dataclass(frozen=True)
class Context:
    protocol: dict


def test_context_with_initial_cash_does_not_mutate_source() -> None:
    source = Context(protocol={"execution": {"initial_cash": 700000.0}})
    result = module.context_with_initial_cash(source, 350000.0)
    assert result.protocol["execution"]["initial_cash"] == 350000.0
    assert source.protocol["execution"]["initial_cash"] == 700000.0


def test_context_with_initial_cash_rejects_nonpositive_value() -> None:
    source = Context(protocol={"execution": {"initial_cash": 700000.0}})
    with pytest.raises(ValueError, match="positive"):
        module.context_with_initial_cash(source, 0.0)
