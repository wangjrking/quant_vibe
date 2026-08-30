from __future__ import annotations

import inspect

import numpy as np

from research_v260_runtime import fixed10_residual_cash_sweep_v112 as runtime


def test_cash_sweep_is_disabled_by_default() -> None:
    assert inspect.signature(runtime.simulate).parameters[
        "residual_cash_sweep_override"
    ].default is False


def test_cash_sweep_trigger_can_require_a_buy_trade() -> None:
    assert runtime.residual_cash_sweep_triggered("any_trade", 1, set()) is True
    assert runtime.residual_cash_sweep_triggered("buy_trade", 1, set()) is False
    assert runtime.residual_cash_sweep_triggered("buy_trade", 1, {2}) is True


def test_cash_sweep_only_tops_up_existing_eligible_name() -> None:
    cash, additions = runtime.residual_cash_sweep_plan(
        stocks=np.asarray(["000001.SZ", "000002.SZ"]),
        shares={0: 900.0, 1: 1000.0},
        raw_opens=np.asarray([10.0, 10.0]),
        mark_prices=np.asarray([10.0, 10.0]),
        eligible_indices={0},
        cash=1100.0,
        target_pct=0.5,
        slippage_ratio=0.0,
    )
    assert additions == {0: 100.0}
    assert 0.0 <= cash < 100.0


def test_cash_sweep_does_not_buy_ineligible_holding() -> None:
    cash, additions = runtime.residual_cash_sweep_plan(
        stocks=np.asarray(["000001.SZ", "000002.SZ"]),
        shares={0: 900.0, 1: 1000.0},
        raw_opens=np.asarray([10.0, 10.0]),
        mark_prices=np.asarray([10.0, 10.0]),
        eligible_indices={1},
        cash=1100.0,
        target_pct=0.5,
        slippage_ratio=0.0,
    )
    assert additions == {}
    assert cash == 1100.0


def test_cash_sweep_rejects_new_name() -> None:
    try:
        runtime.residual_cash_sweep_plan(
            stocks=np.asarray(["000001.SZ", "000002.SZ"]),
            shares={0: 900.0},
            raw_opens=np.asarray([10.0, 10.0]),
            mark_prices=np.asarray([10.0, 10.0]),
            eligible_indices={1},
            cash=1100.0,
            target_pct=0.5,
            slippage_ratio=0.0,
        )
    except ValueError as exc:
        assert "existing holdings" in str(exc)
    else:
        raise AssertionError("cash sweep accepted a new stock")


def test_cash_sweep_can_prioritize_highest_score_existing_holding() -> None:
    cash, additions = runtime.residual_cash_sweep_plan(
        stocks=np.asarray(["000001.SZ", "000002.SZ"]),
        shares={0: 900.0, 1: 900.0},
        raw_opens=np.asarray([10.0, 10.0]),
        mark_prices=np.asarray([10.0, 10.0]),
        eligible_indices={0, 1},
        cash=1100.0,
        target_pct=0.55,
        slippage_ratio=0.0,
        max_additions=1,
        recipient_mode="highest_score",
        priority_scores=np.asarray([0.2, 0.8]),
    )
    assert additions == {1: 100.0}
    assert 0.0 <= cash < 100.0
