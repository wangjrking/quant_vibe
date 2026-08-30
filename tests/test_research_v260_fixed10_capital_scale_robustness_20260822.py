from __future__ import annotations

import pytest

import research_v260_fixed10_capital_scale_robustness_20260822 as target


def test_context_with_initial_cash_is_non_mutating():
    source = target.harness.PressureContext(
        harness=None,
        protocol={"execution": {"initial_cash": 700_000.0}},
        definition={},
        rules={},
        manifests={},
        arrays={},
        access={},
        score=None,
        order=None,
        active=None,
        maintenance_block=None,
        empty_block=None,
    )
    changed = target.context_with_initial_cash(source, 350_000.0)
    assert source.protocol["execution"]["initial_cash"] == 700_000.0
    assert changed.protocol["execution"]["initial_cash"] == 350_000.0


def test_context_with_initial_cash_rejects_nonpositive_value():
    context = target.harness.PressureContext(
        harness=None,
        protocol={"execution": {"initial_cash": 700_000.0}},
        definition={},
        rules={},
        manifests={},
        arrays={},
        access={},
        score=None,
        order=None,
        active=None,
        maintenance_block=None,
        empty_block=None,
    )
    with pytest.raises(ValueError, match="finite and positive"):
        target.context_with_initial_cash(context, 0.0)
