from __future__ import annotations

import ast
import inspect
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest


MAIN_ROOT = Path(r"D:/work/quant/quant_mcp/quant/main")
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_one_shot_2026_validation_runner_20260822 as runner
from research_v260_fixed10_one_shot_2026_validation_runner_20260822 import (
    EXECUTE_CONFIRMATION,
    build_failure_record,
    evaluate_base_candidate_gates,
    expected_validation_dates,
    expected_t1_signal_map,
    load_and_validate_protocol,
    output_artifact_sha256s,
    preflight,
    require_columns,
    validate_and_select_output_arm,
    validate_execute_confirmation,
    validate_protocol_derivation,
)


def test_execute_requires_exact_confirmation() -> None:
    validate_execute_confirmation(False, None)
    with pytest.raises(PermissionError, match="does not match"):
        validate_execute_confirmation(True, "wrong")
    validate_execute_confirmation(True, EXECUTE_CONFIRMATION)


def test_default_preflight_does_not_open_2026() -> None:
    protocol = load_and_validate_protocol()
    result = preflight(protocol)
    assert result["status"] == "build_preflight_passed_2026_not_opened"
    assert result["validation_2026_opened"] is False
    assert result["production_modified"] is False
    assert result["validation_asset_metadata"]["business_row_query_count"] == 0
    assert (
        result["validation_asset_metadata"]["status"]
        == "validation_asset_metadata_ready_without_business_rows"
    )
    assert len(result["event_source_sha256"]) == 64
    assert result["runtime_contract"]["score_construction"] == {
        "weight_5d": 0.0,
        "smooth_window": 7,
        "current_weight": 0.1,
    }
    assert protocol["event_overlay_runtime"]["diagnostic_coverage"] == {
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
    }


def test_protocol_derivation_rejects_any_runtime_drift() -> None:
    protocol = load_and_validate_protocol()
    bindings = protocol["source_bindings"]
    import json
    from pathlib import Path

    checkpoint = json.loads(
        Path(bindings["candidate_checkpoint"]).read_text(encoding="utf-8")
    )
    event_contract = json.loads(
        Path(bindings["event_overlay_contract"]).read_text(encoding="utf-8")
    )
    drifted = json.loads(json.dumps(protocol))
    drifted["runtime_contract"]["score_construction"]["smooth_window"] = 8
    with pytest.raises(RuntimeError, match="drifted from frozen source derivation"):
        validate_protocol_derivation(drifted, checkpoint, event_contract)


def test_base_decision_is_profit_first_with_determinism_only() -> None:
    decision = {
        "primary": "cumulative_return_above_production",
        "machine_thresholds": {
            "deterministic_replay_required": True,
        },
    }
    base = {
        "cumulative_return": 0.2,
        "sharpe": 1.0,
        "max_drawdown": 0.35,
        "full_10_position_ratio": 1.0,
        "average_invested_ratio": 0.95,
    }
    gates = evaluate_base_candidate_gates(
        base,
        {"cumulative_return": 0.1},
        {"cumulative_return": 0.01},
        {"daily": True, "actions": True},
        decision,
    )
    assert set(gates) == {
        "cumulative_return_above_production",
        "deterministic_replay",
    }
    assert all(gates.values())
    base["cumulative_return"] = 0.05
    assert not evaluate_base_candidate_gates(
        base,
        {"cumulative_return": 0.1},
        {"cumulative_return": 0.01},
        {"daily": True, "actions": True},
        decision,
    )["cumulative_return_above_production"]


def test_failure_record_is_terminal_and_does_not_claim_production_change() -> None:
    protocol = load_and_validate_protocol()
    record = build_failure_record(protocol, RuntimeError("synthetic validation failure"))
    assert record["status"] == "failed_after_one_shot_validation_started"
    assert record["exception_type"] == "RuntimeError"
    assert record["retry_allowed"] is False
    assert record["production_modified"] is False
    assert record["trading_triggered"] is False
    assert record["protocol_sha256"] == runner.sha256(runner.PROTOCOL)


def test_failure_record_binds_the_authoritative_start_marker() -> None:
    protocol = load_and_validate_protocol()
    original_marker = runner.START_MARKER
    with tempfile.TemporaryDirectory() as directory:
        marker = Path(directory) / "validation_started.json"
        marker.write_text('{"status":"started"}\n', encoding="utf-8")
        expected_marker_sha256 = runner.sha256(marker)
        runner.START_MARKER = marker
        try:
            record = build_failure_record(protocol, RuntimeError("synthetic"))
        finally:
            runner.START_MARKER = original_marker
    assert record["start_marker_sha256"] == expected_marker_sha256


def test_validation_builds_market_context_and_reads_event_asset_once() -> None:
    tree = ast.parse(inspect.getsource(runner.execute_validation))
    calls = [
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert calls.count("make_context") == 1
    assert calls.count("load_event_features") == 1


def test_validation_checks_asset_metadata_before_consuming_one_shot() -> None:
    source = inspect.getsource(runner.execute_validation)
    assert source.index("validate_validation_asset_metadata") < source.index(
        "atomic_start_marker"
    )


def test_required_schema_columns_fail_closed() -> None:
    require_columns(["trade_date", "stock_code"], {"trade_date"}, "fixture")
    with pytest.raises(RuntimeError, match="schema incomplete"):
        require_columns(["trade_date"], {"trade_date", "stock_code"}, "fixture")


def test_validation_persists_stress_arm_for_independent_audit() -> None:
    source = inspect.getsource(runner.execute_validation)
    assert '"fixed10_stress_0_65pct", stress_daily, stress_actions' in source


def test_partial_output_counts_as_a_consumed_one_shot() -> None:
    original_root = runner.OUTPUT_ROOT
    original_claim = runner.START_CLAIM
    original_marker = runner.START_MARKER
    original_result = runner.RESULT_PATH
    original_failure = runner.FAILURE_PATH
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        runner.OUTPUT_ROOT = root
        runner.START_CLAIM = root / "validation_claim.lock"
        runner.START_MARKER = root / "validation_started.json"
        runner.RESULT_PATH = root / "result.json"
        runner.FAILURE_PATH = root / "failure.json"
        try:
            assert runner.one_shot_already_started() is False
            (root / "fixed10_daily.csv").write_text("partial", encoding="utf-8")
            assert runner.one_shot_already_started() is True
        finally:
            runner.OUTPUT_ROOT = original_root
            runner.START_CLAIM = original_claim
            runner.START_MARKER = original_marker
            runner.RESULT_PATH = original_result
            runner.FAILURE_PATH = original_failure


def test_atomic_csv_publishes_complete_file_without_temporary_residue() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "arm_daily.csv"
        runner.atomic_csv(pd.DataFrame({"date": ["20260105"]}), path)
        assert path.read_text(encoding="utf-8").splitlines() == [
            "date",
            "20260105",
        ]
        assert not path.with_name(f".{path.name}.tmp").exists()


def test_output_artifact_hashes_require_all_eight_files() -> None:
    original_root = runner.OUTPUT_ROOT
    with tempfile.TemporaryDirectory() as directory:
        runner.OUTPUT_ROOT = Path(directory)
        try:
            with pytest.raises(FileNotFoundError, match="missing"):
                output_artifact_sha256s()
            for name in runner.OUTPUT_ARTIFACT_NAMES:
                (runner.OUTPUT_ROOT / name).write_text(name, encoding="utf-8")
            digests = output_artifact_sha256s()
        finally:
            runner.OUTPUT_ROOT = original_root
    assert set(digests) == set(runner.OUTPUT_ARTIFACT_NAMES)
    assert all(len(value) == 64 for value in digests.values())


def test_validation_start_claim_is_exclusive() -> None:
    protocol = load_and_validate_protocol()
    original_root = runner.OUTPUT_ROOT
    original_claim = runner.START_CLAIM
    original_marker = runner.START_MARKER
    original_result = runner.RESULT_PATH
    original_failure = runner.FAILURE_PATH
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        runner.OUTPUT_ROOT = root
        runner.START_CLAIM = root / "validation_claim.lock"
        runner.START_MARKER = root / "validation_started.json"
        runner.RESULT_PATH = root / "result.json"
        runner.FAILURE_PATH = root / "failure.json"
        try:
            runner.atomic_start_marker(protocol)
            assert runner.START_CLAIM.is_file()
            assert runner.START_MARKER.is_file()
            with pytest.raises(FileExistsError, match="already started"):
                runner.atomic_start_marker(protocol)
        finally:
            runner.OUTPUT_ROOT = original_root
            runner.START_CLAIM = original_claim
            runner.START_MARKER = original_marker
            runner.RESULT_PATH = original_result
            runner.FAILURE_PATH = original_failure


def output_frames():
    daily = pd.DataFrame({
        "date": ["20260105", "20260820"],
        "return": [0.01, -0.01],
        "equity": [101.0, 100.0],
        "turnover": [0.2, 0.0],
        "invested_ratio": [0.98, 0.99],
        "positions": [10, 10],
        "trades": [2, 0],
    })
    actions = pd.DataFrame({
        "signal_date": ["20260102"],
        "buy_date": ["20260105"],
        "action": ["BUY"],
        "stock_code": ["600000.SH"],
        "target_pct": [0.1],
        "execution_open_raw": [10.0],
    })
    return daily, actions


def test_output_quality_gate_accepts_a_valid_fixed10_arm() -> None:
    daily, actions = output_frames()
    selected_daily, selected_actions = validate_and_select_output_arm(
        "fixed10",
        daily,
        actions,
        "20260105",
        "20260820",
        ("20260105", "20260820"),
        {"20260105": "20260102", "20260820": "20260819"},
    )
    assert selected_daily["positions"].tolist() == [10, 10]
    assert selected_actions["stock_code"].tolist() == ["600000.SH"]


def test_output_quality_gate_rejects_bad_dates_values_positions_and_actions() -> None:
    daily, actions = output_frames()
    duplicate_daily = pd.concat([daily, daily.iloc[[0]]], ignore_index=True)
    with pytest.raises(RuntimeError, match="date contract"):
        validate_and_select_output_arm(
            "fixed10", duplicate_daily, actions, "20260105", "20260820"
        )
    nonfinite_daily = daily.copy()
    nonfinite_daily.loc[0, "return"] = float("nan")
    with pytest.raises(RuntimeError, match="nonfinite return"):
        validate_and_select_output_arm(
            "fixed10", nonfinite_daily, actions, "20260105", "20260820"
        )
    nine_positions = daily.copy()
    nine_positions.loc[0, "positions"] = 9
    selected_daily, _ = validate_and_select_output_arm(
        "fixed10", nine_positions, actions, "20260105", "20260820"
    )
    assert selected_daily["positions"].tolist() == [9, 10]
    bj_actions = actions.copy()
    bj_actions.loc[0, "stock_code"] = "430001.BJ"
    with pytest.raises(RuntimeError, match="Beijing"):
        validate_and_select_output_arm(
            "fixed10", daily, bj_actions, "20260105", "20260820"
        )


def test_output_quality_gate_requires_the_exact_source_calendar() -> None:
    dates = expected_validation_dates(
        ["20260105", "20260106", "20260820"], "20260105", "20260820"
    )
    assert dates == ("20260105", "20260106", "20260820")
    daily, actions = output_frames()
    with pytest.raises(RuntimeError, match="calendar completeness"):
        validate_and_select_output_arm(
            "fixed10", daily, actions, "20260105", "20260820", dates
        )
    bad_actions = actions.copy()
    bad_actions.loc[0, "buy_date"] = "20260106"
    with pytest.raises(RuntimeError, match="action date"):
        validate_and_select_output_arm(
            "fixed10",
            daily,
            bad_actions,
            "20260105",
            "20260820",
            ("20260105", "20260820"),
        )


def test_equalweight_is_soft_but_t1_remains_execution_semantics() -> None:
    daily, actions = output_frames()
    mapping = expected_t1_signal_map(
        ["20260102", "20260105", "20260820"],
        ("20260105", "20260820"),
    )
    assert mapping == {"20260105": "20260102", "20260820": "20260105"}
    validate_and_select_output_arm(
        "fixed10",
        daily,
        actions,
        "20260105",
        "20260820",
        ("20260105", "20260820"),
        mapping,
    )
    bad_weight = actions.copy()
    bad_weight.loc[0, "target_pct"] = 0.09
    _, selected_actions = validate_and_select_output_arm(
        "fixed10",
        daily,
        bad_weight,
        "20260105",
        "20260820",
        ("20260105", "20260820"),
        mapping,
    )
    assert selected_actions["target_pct"].tolist() == [0.09]
    same_day = actions.copy()
    same_day.loc[0, "signal_date"] = "20260105"
    with pytest.raises(RuntimeError, match=r"T\+1"):
        validate_and_select_output_arm(
            "fixed10",
            daily,
            same_day,
            "20260105",
            "20260820",
            ("20260105", "20260820"),
            mapping,
        )
