import copy
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd


MAIN_ROOT = Path(r"D:/work/quant/quant_mcp/quant/main")
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_one_shot_2026_validation_runner_20260822 as runner
from research_v260_fixed10_one_shot_2026_result_audit_20260822 import (
    audit_failure,
    audit_result,
    classify_business_result,
    metrics_equivalent,
    output_artifact_digests_match,
    recompute_output_metrics,
    terminal_binding_checks,
)


def metric(
    cumulative,
    sharpe=1.0,
    drawdown=0.2,
    positions=1.0,
    invested=0.98,
    start="20260105",
    end="20260820",
):
    return {
        "start": start,
        "end": end,
        "cumulative_return": cumulative,
        "sharpe": sharpe,
        "max_drawdown": drawdown,
        "full_10_position_ratio": positions,
        "average_invested_ratio": invested,
    }


def protocol():
    return {
        "protocol_id": "p1",
        "validation_boundary": {"start": "20260105", "end": "20260820"},
        "base_candidate_decision": {
            "primary": "cumulative_return_above_production",
            "machine_thresholds": {
                "deterministic_replay_required": True,
            },
        },
        "event_overlay_runtime": {
            "diagnostic_coverage": {"event_metric_start": "20260210"}
        },
    }


def result():
    deterministic = {
        "base_daily": True,
        "base_actions": True,
        "overlay_daily": True,
        "overlay_actions": True,
    }
    gates = {
        "cumulative_return_above_production": True,
        "deterministic_replay": True,
    }
    return {
        "status": "passed_research_candidate",
        "event_overlay_status": "diagnostic_only_no_selection_decision",
        "protocol_id": "p1",
        "validation_boundary": ["20260105", "20260820"],
        "metrics": {
            "production_strategy_reference": metric(0.10),
            "fixed10_current_candidate_without_event_overlay": metric(0.20),
            "fixed10_current_candidate_stress_0_65pct": metric(0.05),
            "fixed10_current_candidate_with_frozen_event_overlay": metric(0.21),
        },
        "event_overlay_coverage_metrics": {
            "fixed10_without_event_tiebreak": metric(0.18, start="20260210"),
            "fixed10_with_severe_event_tiebreak": metric(0.19, start="20260210"),
        },
        "base_candidate_gates": gates,
        "event_overlay_diagnostic_delta_vs_base": {"cumulative_return": 0.01},
        "event_audit": {
            "diagnostic_coverage": {"event_metric_start": "20260210"},
            "precoverage_treated_as_no_event": False,
        },
        "deterministic_replay": deterministic,
        "validation_access": {
            "logical_max_date": "20260820",
            "business_date_predicate_max": "20260820",
        },
        "validation_2026_opened": True,
        "production_modified": False,
        "trading_triggered": False,
    }


def test_passed_result_is_recomputed_from_base_arm_only() -> None:
    actual = audit_result(result(), protocol())
    assert actual["status"] == "readonly_result_audit_passed"
    assert actual["candidate_passed"] is True
    assert actual["candidate_business_classification"] == "risk_adjusted_upgrade"
    assert actual["candidate_role"] == "production_replacement_candidate"


def test_passed_return_with_worse_risk_is_not_called_a_replacement() -> None:
    value = result()
    candidate = value["metrics"][
        "fixed10_current_candidate_without_event_overlay"
    ]
    candidate["sharpe"] = 0.8
    candidate["max_drawdown"] = 0.3
    actual = audit_result(value, protocol())
    assert actual["status"] == "readonly_result_audit_passed"
    assert actual["candidate_passed"] is True
    assert (
        actual["candidate_business_classification"]
        == "return_upgrade_with_risk_tradeoff"
    )
    assert actual["candidate_role"] == (
        "return_superior_candidate_with_disclosed_risk_tradeoff"
    )


def test_business_classification_rejects_failed_gates() -> None:
    value = result()
    gates = dict(value["base_candidate_gates"])
    gates["cumulative_return_above_production"] = False
    assert classify_business_result(value["metrics"], gates) == "reject"


def test_metric_equivalence_is_numeric_and_nested() -> None:
    actual = {"return": 0.1, "annual": {"2026": 0.2}, "count": 10}
    expected = {
        "return": 0.1 + 1e-13,
        "annual": {"2026": 0.2},
        "count": 10,
    }
    assert metrics_equivalent(actual, expected)
    expected["annual"]["2026"] = 0.21
    assert not metrics_equivalent(actual, expected)


def test_output_metric_recomputation_uses_all_four_arms() -> None:
    import research_v260_fixed10_one_shot_2026_result_audit_20260822 as target

    daily = pd.DataFrame(
        {
            "date": ["20260105", "20260820"],
            "return": [0.01, -0.005],
            "turnover": [0.1, 0.2],
            "invested_ratio": [1.0, 1.0],
            "positions": [10, 10],
            "trades": [1, 1],
        }
    )
    actions = pd.DataFrame(
        {
            "buy_date": ["20260105", "20260820"],
            "action": ["BUY", "SELL"],
        }
    )
    scratch_parent = target.REPO / "quant/data_file/runtime"
    with tempfile.TemporaryDirectory(dir=scratch_parent) as directory:
        scratch = Path(directory)
        prefixes = (
            "production",
            "fixed10",
            "fixed10_stress_0_65pct",
            "fixed10_event_overlay",
        )
        for prefix in prefixes:
            daily.to_csv(scratch / f"{prefix}_daily.csv", index=False)
            actions.to_csv(scratch / f"{prefix}_actions.csv", index=False)
        expected = target.round1.evaluate_run(
            daily, actions, "20260105", "20260820"
        )
        value = result()
        value["metrics"] = {name: expected for name in value["metrics"]}
        value["event_overlay_coverage_metrics"] = {
            name: expected for name in value["event_overlay_coverage_metrics"]
        }
        contract = protocol()
        contract["event_overlay_runtime"]["diagnostic_coverage"][
            "event_metric_start"
        ] = "20260105"
        original_root = target.VALIDATION_ROOT
        target.VALIDATION_ROOT = scratch
        try:
            recomputed = recompute_output_metrics(value, contract)
        finally:
            target.VALIDATION_ROOT = original_root
        assert recomputed["all_recomputed_metrics_match"] is True
        assert all(recomputed["artifact_boundary_checks"].values())


def test_event_overlay_cannot_rescue_a_rejected_base_arm() -> None:
    value = result()
    value["metrics"]["fixed10_current_candidate_without_event_overlay"] = metric(0.05)
    value["metrics"]["fixed10_current_candidate_with_frozen_event_overlay"] = metric(0.50)
    value["status"] = "rejected_on_2026"
    value["base_candidate_gates"]["cumulative_return_above_production"] = False
    actual = audit_result(value, protocol())
    assert actual["status"] == "readonly_result_audit_passed"
    assert actual["candidate_passed"] is False


def test_tampered_terminal_status_fails_audit() -> None:
    value = result()
    value["status"] = "rejected_on_2026"
    actual = audit_result(value, protocol())
    assert not actual["checks"]["terminal_status_matches_base_gates"]
    assert actual["status"] == "readonly_result_audit_failed"


def test_metric_or_access_boundary_drift_fails_audit() -> None:
    value = result()
    value["metrics"]["production_strategy_reference"]["start"] = "20251231"
    value["validation_access"]["logical_max_date"] = "20260819"
    actual = audit_result(value, protocol())
    assert actual["status"] == "readonly_result_audit_failed"
    assert actual["checks"]["primary_metric_boundaries_exact"] is False
    assert actual["checks"]["validation_access_endpoint_exact"] is False


def test_production_or_trading_side_effect_fails_audit() -> None:
    value = result()
    value["production_modified"] = True
    value["trading_triggered"] = True
    actual = audit_result(value, protocol())
    assert actual["status"] == "readonly_result_audit_failed"
    assert not actual["checks"]["production_unchanged"]
    assert not actual["checks"]["trading_not_triggered"]


def test_failure_terminal_requires_no_retry_and_complete_exception() -> None:
    value = {
        "status": "failed_after_one_shot_validation_started",
        "protocol_id": "p1",
        "protocol_sha256": "placeholder",
        "validation_boundary": {"start": "20260105", "end": "20260820"},
        "validation_2026_opened": True,
        "retry_allowed": False,
        "production_modified": False,
        "trading_triggered": False,
        "exception_type": "RuntimeError",
        "exception_message": "failed",
        "traceback": "trace",
    }
    expected_sha = "placeholder"
    import research_v260_fixed10_one_shot_2026_result_audit_20260822 as target

    original = target.runner.sha256
    target.runner.sha256 = lambda _: expected_sha
    try:
        actual = audit_failure(value, protocol())
    finally:
        target.runner.sha256 = original
    assert actual["status"] == "readonly_failure_audit_passed"


def test_missing_event_coverage_is_not_a_valid_result() -> None:
    value = copy.deepcopy(result())
    value["event_audit"].pop("diagnostic_coverage")
    assert audit_result(value, protocol())["status"] == "readonly_result_audit_failed"


def test_terminal_binding_requires_the_exact_protocol_and_start_marker() -> None:
    contract = protocol()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        protocol_path = root / "protocol.json"
        claim_path = root / "validation_claim.lock"
        marker_path = root / "validation_started.json"
        protocol_path.write_text(
            json.dumps(contract, sort_keys=True) + "\n", encoding="utf-8"
        )
        claim_path.write_text(
            '{"status":"exclusive_validation_claim_acquired"}\n',
            encoding="utf-8",
        )
        marker = {
            "protocol_id": contract["protocol_id"],
            "protocol_sha256": runner.sha256(protocol_path),
            "validation_boundary": contract["validation_boundary"],
            "exclusive_claim_sha256": runner.sha256(claim_path),
        }
        marker_path.write_text(
            json.dumps(marker, sort_keys=True) + "\n", encoding="utf-8"
        )
        terminal = {
            "protocol_sha256": runner.sha256(protocol_path),
            "start_marker_sha256": runner.sha256(marker_path),
        }
        checks = terminal_binding_checks(
            terminal, contract, protocol_path, marker_path, claim_path
        )
        assert all(checks.values())
        terminal["start_marker_sha256"] = "0" * 64
        assert not terminal_binding_checks(
            terminal, contract, protocol_path, marker_path, claim_path
        )["terminal_start_marker_sha256_exact"]


def test_output_artifact_digests_must_match_the_terminal_declaration() -> None:
    declared = {name: str(index).zfill(64) for index, name in enumerate(
        runner.OUTPUT_ARTIFACT_NAMES, start=1
    )}
    artifacts = {
        name: {"exists": True, "sha256": digest}
        for name, digest in declared.items()
    }
    terminal = {"output_artifact_sha256": declared}
    assert output_artifact_digests_match(terminal, artifacts)
    artifacts[runner.OUTPUT_ARTIFACT_NAMES[0]]["sha256"] = "0" * 64
    assert not output_artifact_digests_match(terminal, artifacts)
