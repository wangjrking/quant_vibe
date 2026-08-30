from __future__ import annotations

import sys
from pathlib import Path


MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

from research_v260_fixed10_one_shot_2026_validation_protocol_freeze_20260822 import (
    CONTROL_PLANE_FILES,
    FROZEN_CANDIDATE,
    FROZEN_PRODUCTION_HARNESS,
    VALIDATION_RUNNER,
    build_control_plane_binding,
    build_executable_code_binding,
    build_protocol,
    build_runtime_environment_binding,
    build_runtime_contract,
)


def fixtures():
    checkpoint = {
        "validation_2026_opened": False,
        "selected_candidate": FROZEN_CANDIDATE,
        "selected_policy": {
            "score_sell_pressure_trigger_rule": "4_strong_market_5_weak_market",
            "score_sell_pressure_extra_min_age_rule": "4_strong_market_10_weak_market",
            "confirmed_weak_sell_priority": {
                "confirmation_days": 2,
                "formula": "exit_score - 0.05 * trailing_20d_volatility_percentile",
                "changes_eligibility": False,
                "changes_exit_count": False,
            },
            "entry_rank_sizing": {
                "application": "new_entries_only",
                "ranking": "global_existing_frozen_score_order_before_execution_refill",
                "positions": 10,
                "top_multiplier": 1.1,
                "bottom_multiplier": 0.9,
                "schedule": "linear_descending",
                "reference_top10_multiplier_sum": 10.0,
                "refill_outside_global_top10_multiplier": 1.0,
                "actual_entry_batch_gross_neutral": False,
                "equalweight_is_soft_reference": True,
            },
            "residual_cash_sweep": {
                "enabled": True,
                "trigger": "after_buy_trade",
                "recipient": "most_underweight_existing_buyable_holding",
                "target_pct": "gross_target_divided_by_max_positions",
                "board_lot_shares": 100,
                "max_orders_per_trade_day": None,
                "new_names_allowed": False,
                "forced_sales_allowed": False,
                "equalweight_and_full_investment_are_soft_directions": True,
            },
        },
    }
    event = {
        "validation_2026_opened": False,
        "contract_id": "fixed10_tushare_severe_score_tiebreak_diagnostic_v2",
        "diagnostic_coverage": {
            "source_event_min_dates": {
                "stk_shock": "20260303",
                "stk_high_shock": "20260209",
                "stk_alert": "20260210",
            },
            "ordinary_first_visible_session": "20260304",
            "severe_first_visible_session": "20260210",
            "exchange_alert_first_visible_session": "20260211",
            "event_metric_start": "20260210",
            "precoverage_state": "unknown_not_zero",
        },
        "one_shot_validation_arms": ["production", "candidate", "overlay"],
        "decision_order": {"event_overlay": "diagnostic only"},
        "event_overlay_role": "diagnostic_only_not_selection_candidate",
    }
    return checkpoint, event


def test_protocol_has_one_read_and_no_post_validation_tuning() -> None:
    checkpoint, event = fixtures()
    protocol = build_protocol(checkpoint, event)
    assert protocol["validation_boundary"]["read_once"] is True
    assert protocol["development_boundary"][
        "no_further_parameter_selection_after_validation"
    ] is True
    assert protocol["hard_boundaries"]["no_extra_candidate"] is True
    assert protocol["hard_boundaries"][
        "event_overlay_cannot_change_base_decision"
    ] is True
    assert protocol["event_overlay_role"] == "diagnostic_only_not_selection_candidate"
    assert protocol["event_overlay_runtime"]["diagnostic_coverage"][
        "precoverage_state"
    ] == "unknown_not_zero"
    assert protocol["runtime_contract"]["score_construction"]["smooth_window"] == 7
    assert protocol["runtime_contract"]["pressure_regime"]["weak_extra_min_age"] == 10
    assert protocol["runtime_contract"]["entry_rank_sizing"]["top_multiplier"] == 1.1
    assert protocol["runtime_contract"]["residual_cash_sweep"]["enabled"] is True
    assert protocol["base_candidate_decision"]["machine_thresholds"] == {
        "deterministic_replay_required": True,
    }
    assert protocol["base_candidate_decision"]["required"] == {
        "cumulative_return_above_production": True,
        "deterministic_replay": True,
    }
    disclosed_only = set(
        protocol["base_candidate_decision"]["reported_without_extra_gates"]
    )
    assert {
        "full_10_position_ratio",
        "average_invested_ratio",
        "sharpe_delta_vs_production",
        "max_drawdown_delta_vs_production",
        "turnover_delta_vs_production",
    }.issubset(disclosed_only)
    assert protocol["validation_2026_opened"] is False


def test_protocol_rejects_an_already_opened_validation() -> None:
    checkpoint, event = fixtures()
    checkpoint["validation_2026_opened"] = True
    try:
        build_protocol(checkpoint, event)
    except PermissionError:
        return
    raise AssertionError("an already-opened validation must be rejected")


def test_runtime_contract_rejects_candidate_rule_drift() -> None:
    checkpoint, _ = fixtures()
    checkpoint["selected_policy"]["confirmed_weak_sell_priority"][
        "confirmation_days"
    ] = 3
    try:
        build_runtime_contract(checkpoint)
    except RuntimeError as exc:
        assert "sell-priority" in str(exc)
        return
    raise AssertionError("candidate runtime drift must be rejected")


def test_executable_code_binding_covers_runner_runtime_and_frozen_harness() -> None:
    binding = build_executable_code_binding()
    paths = {item["path"] for item in binding["files"]}
    assert str(VALIDATION_RUNNER.resolve()).replace("\\", "/") in paths
    assert str(FROZEN_PRODUCTION_HARNESS.resolve()).replace("\\", "/") in paths
    assert any(
        path.endswith("research_v260_runtime/fixed10_residual_cash_sweep_v112.py")
        for path in paths
    )
    assert any("production_code/v260_all4key_runtime/" in path for path in paths)
    assert any("adaptive_exit_v162" in path for path in paths)
    assert any("breadth_exit_v109" in path for path in paths)
    assert any("score_deterioration_v174" in path for path in paths)
    assert any("position_warmup_v260" in path for path in paths)
    assert binding["file_count"] == len(paths)
    assert len(binding["bundle_sha256"]) == 64


def test_control_plane_binding_covers_exactly_the_frozen_production_inputs() -> None:
    binding = build_control_plane_binding()
    paths = {item["path"] for item in binding["files"]}
    expected = {
        str(path.resolve()).replace("\\", "/") for path in CONTROL_PLANE_FILES
    }
    assert paths == expected
    assert binding["file_count"] == 7
    assert len(binding["bundle_sha256"]) == 64

    checkpoint, event = fixtures()
    protocol = build_protocol(checkpoint, event)
    assert protocol["source_bindings"]["control_plane"] == binding


def test_runtime_environment_binding_is_minimal_and_deterministic() -> None:
    binding = build_runtime_environment_binding()
    assert binding["python"]["implementation"] == "cpython"
    assert binding["python"]["version"].count(".") == 2
    assert binding["python"]["executable"].endswith("/python.exe")
    assert set(binding["packages"]) == {"duckdb", "numpy", "pandas"}
    assert all(binding["packages"].values())

    checkpoint, event = fixtures()
    protocol = build_protocol(checkpoint, event)
    assert protocol["source_bindings"]["runtime_environment"] == binding
